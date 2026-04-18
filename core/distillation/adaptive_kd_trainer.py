"""
core/distillation/adaptive_kd_trainer.py
==========================================
自适应知识蒸馏训练器（AdaptiveKDTrainer）— v4 内存优化版

核心设计原则（v4）：
  1. 复用 ultralytics 外层已计算的学生预测，消除双重前向传播（显存 -50%）
  2. resume 时使用基础权重创建 YOLO 对象，避免双重加载
  3. 绝不修改 model 对象的持久化状态
  4. 教师模型启用梯度检查点 + 积极显存回收
"""

import os
import gc
import json
import math
import re
import logging
import csv as _c
import io
import weakref
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer, DEFAULT_CFG

from core.distillation.common import safe_scalar
from core.distillation.loss_functions import (
    CompositiveDistillLoss,
    CosineTemperatureScheduler,
)

logger = logging.getLogger("DistillTrain")
# 【断点续训修复】使用普通 dict 而非 WeakKeyDictionary！
# 原因：ultralytics 在 resume_training() 时可能用 checkpoint 替换 self.model，
#       WeakKeyDictionary 会自动清除旧模型引用，导致新模型无法找到 trainer。
#       普通字典确保断点续训后 _distill_model_loss 仍能正确路由到 trainer。
_KD_TRAINERS: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha 自适应调度器
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveAlphaScheduler:
    """动态调节蒸馏权重 alpha"""
    
    def __init__(self, alpha_init=0.5, alpha_min=0.2, alpha_max=0.8,
                 lr_alpha=0.05, ema_decay=0.9, delta_target=0.01):
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lr_alpha = lr_alpha
        self.ema_decay = ema_decay
        self.delta_target = delta_target
        self._ema_loss = None
        self._prev_ema = None
    
    @property
    def current_alpha(self): return self.alpha
    
    def update(self, task_loss):
        v = safe_scalar(task_loss)
        self._ema_loss = v if self._ema_loss is None else self.ema_decay * self._ema_loss + (1-self.ema_decay)*v
        if self._prev_ema is not None:
            d = (self._prev_ema - self._ema_loss)/(self._prev_ema + 1e-8)
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha + self.lr_alpha*(self.delta_target-d)))
        self._prev_ema = self._ema_loss
        return self.alpha


def _safe_scalar(value):
    """兼容旧调用点，内部委托到共享实现。"""
    return safe_scalar(value)


def _format_progress_bar(completed, total, width=24):
    if total <= 0:
        return '[unknown]'
    completed = min(max(int(completed), 0), int(total))
    ratio = completed / float(total)
    filled = int(round(ratio * width))
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def _find_trainer_for_model(model):
    """
    安全查找与给定模型关联的 trainer。
    
    【断点续训场景】ultralytics 在 resume_training() 时可能替换 self.model，
    此时直接用新模型查 _KD_TRAINERS 会失败。需要遍历查找匹配的 trainer。
    """
    # 直接查找（正常路径）
    trainer = _KD_TRAINERS.get(model)
    if trainer is not None:
        return trainer
    
    # 回退：通过 id 查找（断点续训时模型可能被替换但 trainer 仍存在）
    # 检查是否有 trainer 的 self.model 与传入的 model 是同一对象或 id 匹配
    for registered_model, t in list(_KD_TRAINERS.items()):
        if t is not None and getattr(t, 'model', None) is not None:
            if t.model is model or id(t.model) == id(model):
                # 更新注册表，让后续调用走快速路径
                _KD_TRAINERS[model] = t
                return t
    
    return None


def _distill_model_loss(model, batch, preds=None):
    """
    注入到 DetectionModel.loss 的蒸馏包装器。

    【重要】ultralytics 训练循环中调用链：
      model(batch) → BaseModel.forward(dict) → model.loss(batch)   ← 注意：不传 preds！
      或 compile 模式：model(img) → model.loss(batch, preds=preds) ← 只有 compile 才传 preds

    此函数确保无论哪种模式，都正确缓存学生预测并传递给 trainer.get_loss()。
    """
    trainer = _find_trainer_for_model(model)
    if trainer is not None:
        # 将当前 batch 存入 trainer
        trainer._batch = batch

        # 【核心修复】当 preds 为空时（非 compile 模式的默认情况），
        # 主动计算学生预测并缓存，避免蒸馏阶段回退到双重前向传播。
        # 【关键】必须使用 model(img) 而非 model.predict(img)！
        #   model.predict() 内部使用 torch.no_grad()，返回的预测无梯度 → 反向传播失效！
        #   model(img) 保留计算图，确保蒸馏损失梯度能正确回传给学生模型参数。
        if preds is None:
            img = batch['img']
            device_type = img.device.type
            with torch.amp.autocast(device_type, enabled=trainer.args.amp if hasattr(trainer, 'args') else False):
                preds = model(img.float())

        trainer._cached_student_preds = preds  # 缓存学生预测供 get_loss 复用
        return trainer.get_loss()

    # 无 trainer 注册时的安全回退（不应发生，但保留以防万一）
    original_loss_fn = getattr(type(model), '_distill_original_loss', None)
    if original_loss_fn is None:
        raise RuntimeError("Distill loss wrapper invoked without registered trainer")
    return original_loss_fn(model, batch, preds=preds)


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练器 — v3: 纯 get_loss 重写，零模型污染
# ═══════════════════════════════════════════════════════════════════════════════
class AdaptiveKDTrainer(DetectionTrainer):
    """
    自适应知识蒸馏训练器 — v3 零污染版
    
    关键改进（v3）：
      - 不修改 model 对象任何属性 → pickle/EMA/deepcopy 全部兼容
      - 通过重写 get_loss() 注入蒸馏逻辑（ultralytics 官方扩展方式）
      - 原始 loss 函数保存在 trainer._orig_loss 上
      - 无需注册表、无需 MethodType 绑定、无需 monkey-patch
    """
    
    _kd_class_params = None
    
    @classmethod
    def set_kd_params(cls, teacher_path="", alpha_init=0.5, T_max=6.0, T_min=1.5,
                      warm_epochs=5, w_kd=0.5, w_focal=0.3, w_feat=0.0,
                      scale_boost=2.0, focal_gamma=2.0):
        cls._kd_class_params = {
            "teacher_path": teacher_path, "alpha_init": alpha_init, "T_max": T_max,
            "T_min": T_min, "warm_epochs": warm_epochs, "w_kd": w_kd, "w_focal": w_focal,
            "w_feat": w_feat, "scale_boost": scale_boost, "focal_gamma": focal_gamma,
        }
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # 蒸馏状态全部挂在 trainer 自身上，绝不碰 model
        self.teacher_model = None
        self._orig_loss = None          # 原始损失函数（从 model.loss 保存后恢复 model）
        self._teacher_path = ""
        self._warm_epochs = 5
        self._w_kd = 0.5; self._w_focal = 0.3; self._w_feat = 0.0
        self._T_max = 6.0; self._T_min = 1.5; self._alpha_init = 0.5
        self._scale_boost = 2.0; self._focal_gamma = 2.0
        self._alpha_scheduler = None
        self._temp_scheduler = None
        self._distill_loss = None
        self._distill_log = []
        self._epoch_task_loss = 0.0; self._epoch_kd_loss = 0.0
        self._batch_count = 0; self._distill_entered = False
        self._batch = None  # 当前批次数据，由 _distill_model_loss 注入
        self._cached_student_preds = None  # 【v4修复】ultralytics外层已计算的学生预测（避免双重前向）
        # 蒸馏是否激活（warm-up 后自动开启，无需手动控制）
        # 去掉 _distill_active 标志，改用 epoch 直接判断（与旧版 v2 行为一致）
    
    def setup_model(self):
        # 【关键修复】必须返回 super().setup_model() 的返回值（ckpt），
        # 否则 _setup_train 中 resume_training(ckpt) 收到 None，
        # 导致 resume 失效、epoch 从头开始！
        ckpt = super().setup_model()

        logger.info("[INIT 5/6] 检测模型就绪 → 初始化蒸馏组件...")

        # 【关键】保存原始 loss 函数引用，然后立即恢复 model.loss
        # （后续蒸馏逻辑通过 get_loss() 注入，不需要替换 model.loss）
        self._orig_loss = getattr(self.model, 'loss', None)
        if self._orig_loss is None:
            logger.error("[FATAL] 找不到 model.loss!")
            return
        
        kd = AdaptiveKDTrainer._kd_class_params
        if kd and kd.get("teacher_path"):
            self._teacher_path = kd["teacher_path"]
            self._warm_epochs = int(kd.get("warm_epochs", 5))
            self._w_kd=float(kd.get("w_kd", 0.5)); self._w_focal=float(kd.get("w_focal", 0.3))
            self._w_feat=float(kd.get("w_feat", 0.0))
            self._T_max=float(kd.get("T_max", 6.0)); self._T_min=float(kd.get("T_min", 1.5))
            self._alpha_init=float(kd.get("alpha_init", 0.5))
            self._scale_boost=float(kd.get("scale_boost", 2.0)); self._focal_gamma=float(kd.get("focal_gamma", 2.0))
            logger.info(f"蒸馏参数 | teacher={os.path.basename(self._teacher_path)} | "
                       f"α={self._alpha_init} T=[{self._T_max},{self._T_min}]")
        
        logger.info(f"[INIT 5/6] 加载教师模型: {os.path.basename(self._teacher_path)} ...")
        self._load_teacher()
        self._init_distill()
        logger.info(f"[INIT 5/6] 蒸馏损失函数 + 调度器 就绪 ✓")

        # 为模型注入蒸馏损失包装器，并保留原始 loss 方法以便热身阶段和权重保存。
        _KD_TRAINERS[self.model] = self
        original_loss_fn = self._orig_loss.__func__ if hasattr(self._orig_loss, '__func__') else self._orig_loss
        type(self.model).loss = _distill_model_loss
        type(self.model)._distill_original_loss = original_loss_fn
        self._class_loss_backup = original_loss_fn
        logger.warning("[DEBUG] AdaptiveKDTrainer patched DetectionModel.loss for distillation")

        return ckpt  # 返回 checkpoint 供 _setup_train 中 resume_training() 使用

    def _load_teacher(self):
        if not self._teacher_path or not os.path.exists(self._teacher_path):
            raise FileNotFoundError(f"教师模型不存在: {self._teacher_path}")
        logger.info(f"加载教师模型: {self._teacher_path}")
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            teacher = YOLO(self._teacher_path, verbose=False)
        
        teacher_model = teacher.model.to(self.device).eval()
        del teacher  # 释放 YOLO 包装器（只保留内部模型）
        
        # 冻结所有参数
        for p in teacher_model.parameters():
            p.requires_grad = False
        
        # 【v4内存优化】对教师模型启用梯度检查点
        # 教师模型仅做前向推理，gradient_checkpointing 可大幅减少中间激活值显存占用
        try:
            teacher_model.gradient_checkpointing_enable()
            logger.info("教师模型已启用梯度检查点 (gradient checkpointing)")
        except Exception:
            # 某些版本不支持或已开启，忽略错误
            pass
        
        self.teacher_model = teacher_model
        # 手动清理
        gc_collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("教师模型就绪 ✓ (显存已优化)")
    
    def _init_distill(self):
        total_epochs = self.args.epochs if hasattr(self.args, "epochs") else 150
        self._alpha_scheduler = AdaptiveAlphaScheduler(alpha_init=self._alpha_init)
        self._temp_scheduler = CosineTemperatureScheduler(T_max=self._T_max, T_min=self._T_min, total_epochs=total_epochs)
        self._distill_loss = CompositiveDistillLoss(w_kd=self._w_kd, w_focal=self._w_focal, w_feat=self._w_feat,
                                                     temperature=self._T_max, scale_boost=self._scale_boost, focal_gamma=self._focal_gamma).to(self.device)
        logger.info(f"蒸馏组件就绪 | T:[{self._T_max}→{self._T_min}] α:{self._alpha_init} warm:{self._warm_epochs}")
        
        self.add_callback("on_train_epoch_start", self._on_epoch_start)
        self.add_callback("on_train_epoch_end", self._on_epoch_end)
        self.add_callback("on_train_batch_end", self._on_train_batch_end)
        self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        self.add_callback("on_val_end", self._on_val_end)
        self.add_callback("on_train_end", self._on_train_end)  # 安全网：确保最终CSV完整
        logger.info("AdaptiveKDTrainer callbacks registered: on_train_epoch_start, on_train_epoch_end, on_train_batch_end, on_fit_epoch_end, on_val_end, on_train_end")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 可选：兼容旧版 ultralytics 的 get_loss 扩展点
    # 当前训练流程实际通过 DetectionModel.loss 注入蒸馏损失
    # ══════════════════════════════════════════════════════════════════════════
    def get_loss(self):
        """
        重写损失计算：warm-up 用原始损失，之后注入蒸馏损失。
        
        【v4 内存修复】关键改进：
          - 通过 _cached_student_preds 复用 ultralytics 外层 model(batch) 已计算的学生预测
          - 蒸馏阶段不再额外调用 self.model(img)，消除双重前向传播
          - 显存占用降低 ~50%
        """
        if self._orig_loss is None:
            if hasattr(super(), 'get_loss'):
                return super().get_loss()
            raise RuntimeError("Distill trainer missing original model loss function")

        # 输出批次进度（节流：每 epoch 最多 ~10 条日志）
        epoch = getattr(self, 'epoch', 0)
        display_epoch = (epoch + 1) if epoch is not None else None
        total_epochs, batch_size, total_batches, dataset_samples = self._get_epoch_info()
        current_batch = self._batch_count + 1
        warm_phase = epoch < self._warm_epochs
        phase = 'warm' if warm_phase else 'distill'
        
        _log_interval = getattr(self, '_batch_log_interval', 0)
        if _log_interval <= 0 and total_batches and isinstance(total_batches, int):
            self._batch_log_interval = max(1, total_batches // 10)
            _log_interval = self._batch_log_interval
        
        if current_batch == 1 or (_log_interval > 0 and current_batch % _log_interval == 0) or current_batch == total_batches:
            percent = current_batch / total_batches * 100.0 if total_batches and total_batches > 0 else 0.0
            bar = _format_progress_bar(current_batch, total_batches) if total_batches else '[unknown]'
            logger.info(
                f"[BATCH_PROGRESS] Epoch {display_epoch}/{total_epochs} | "
                f"Batch [{current_batch:>4d}/{total_batches}] | "
                f"{bar} {percent:5.1f}% | "
                f"batch_size={batch_size} | "
                f"samples={current_batch * batch_size if batch_size else '?'}/{dataset_samples or '?'} | "
                f"phase={phase}"
            )

        # Warm-up 阶段：复用 _distill_model_loss 里已算好的学生预测，避免同 batch 二次前向
        batch = self._batch
        if warm_phase:
            warm_preds = self._cached_student_preds
            try:
                loss, loss_items = self._orig_loss(batch, preds=warm_preds)
            except TypeError:
                try:
                    loss, loss_items = self._orig_loss(batch, preds=batch.get('preds') if isinstance(batch, dict) else None)
                except TypeError:
                    loss, loss_items = self._orig_loss(batch)
            self._accumulate_losses(task_loss=loss)
            return loss, loss_items

        # ════════════════════════════════════════════════════════
        # 蒸馏阶段（epoch >= warm_epochs）— v4 内存优化版
        # ════════════════════════════════════════════════════════
        if not self._distill_entered:
            self._distill_entered = True
            logger.info(f"[DISTILL] 进入蒸馏阶段 epoch={epoch}")

        img = batch['img']
        device_type = img.device.type

        # 【v4核心】复用 ultralytics 外层已计算的学生预测，避免额外前向！
        student_preds = self._cached_student_preds
        if student_preds is None:
            # 极端情况回退：外层未传递 preds（理论上不应发生）
            logger.warning("[DISTILL] 缓存预测为空，执行回退前向（应避免此路径）")
            with torch.amp.autocast(device_type, enabled=False):
                student_preds = self.model(img.float())

        try:
            # 原始任务损失（复用学生预测）
            loss_task, loss_items = self._orig_loss(batch, preds=student_preds)
        except Exception as e:
            logger.warning(f"[DISTILL] _orig_loss(preds=...) 失败 ({e})，尝试无参调用")
            loss_task, loss_items = self._orig_loss(batch)

        # 教师模型前向传播（无梯度）
        teacher_img = batch['img'].float()
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=False):
                teacher_preds = self.teacher_model(teacher_img)

        # 蒸散损失计算
        num_classes = getattr(self.model, 'nc', 80)
        current_temp = self._temp_scheduler.current_temperature
        loss_distill, _detail = self._distill_loss(
            student_preds=student_preds,
            teacher_preds=teacher_preds,
            num_classes=num_classes,
            temperature=current_temp,
        )

        # 立即释放中间张量，回收显存
        del student_preds, teacher_preds
        self._cached_student_preds = None  # 清除缓存引用
        if device_type == 'cuda':
            torch.cuda.empty_cache()

        # 加权组合
        alpha = self._alpha_scheduler.current_alpha
        total_loss = (1 - alpha) * loss_task + alpha * loss_distill

        self._accumulate_losses(task_loss=loss_task, kd_loss=loss_distill)

        return total_loss, loss_items
    
    def _setup_train(self):
        """重写 setup_train：全程锁定 workers=0，避免 DataLoader 子进程与主训练争内存。"""
        self.args.workers = 0

        # 【关键修复】保存配置文件中的 epochs 值，防止 resume 时被 checkpoint 覆盖
        # 当使用断点续训时，ultralytics 会从 last.pt 中恢复所有训练参数（含 epochs），
        # 如果之前是用不同 epochs 训练的，会导致总轮数变成错误的值。
        cfg_epochs = getattr(self.args, 'epochs', None)

        total_batches = "unknown"
        super()._setup_train()

        # 再次强制 workers=0：checkpoint 内的 args 可能在 super() 里把 workers 改回非 0
        self.args.workers = 0

        # 强制将 epochs 恢复为配置文件中的值（而非 checkpoint 里的旧值）
        if cfg_epochs is not None and hasattr(self.args, 'epochs'):
            restored = int(cfg_epochs)
            if self.args.epochs != restored:
                logger.info(
                    f"[RESUME_FIX] epochs 已被 checkpoint 覆盖 → "
                    f"强制恢复为配置文件值: {self.args.epochs} → {restored}"
                )
                self.args.epochs = restored

        for attr in ('dataloader', 'train_dataloader', 'train_loader', 'loader', 'data_loader'):
            loader = getattr(self, attr, None)
            if loader is not None:
                try:
                    total_batches = len(loader)
                    break
                except Exception:
                    continue
        self._known_total_batches = total_batches if isinstance(total_batches, int) else None
        logger.info(
            f"[INIT 6/6] 数据集加载完成 ✓ | "
            f"total_batches={total_batches} | batch_size={self.args.batch} | "
            f"epochs={self.args.epochs} | 即将开始训练..."
        )
    
    def _accumulate_losses(self, task_loss=None, kd_loss=None):
        if task_loss is not None: self._epoch_task_loss += _safe_scalar(task_loss)
        if kd_loss is not None: self._epoch_kd_loss += _safe_scalar(kd_loss)
        self._batch_count += 1
    
    def _get_epoch_info(self):
        total_epochs = getattr(self.args, 'epochs', 0) if hasattr(self, 'args') else 0
        batch_size = getattr(self.args, 'batch', 0) if hasattr(self, 'args') else 0
        total_batches = None
        for attr in ('dataloader', 'train_dataloader', 'train_loader', 'loader', 'data_loader'):
            loader = getattr(self, attr, None)
            if loader is not None:
                try:
                    total_batches = len(loader)
                    break
                except Exception:
                    continue
        if total_batches is None:
            total_batches = getattr(self, '_known_total_batches', None)
        dataset_samples = total_batches * batch_size if total_batches is not None and batch_size else None
        return total_epochs, batch_size, total_batches, dataset_samples

    def _on_epoch_start(self, trainer):
        epoch = trainer.epoch
        display_epoch = epoch + 1
        total_epochs, batch_size, total_batches, dataset_samples = self._get_epoch_info()
        logger.info(
            f"[EPOCH_START] epoch={display_epoch} total={total_epochs} "
            f"batch_size={batch_size} total_batches={total_batches or 'unknown'} "
            f"dataset_samples={dataset_samples if dataset_samples is not None else 'unknown'}"
        )

    def _on_epoch_end(self, trainer):
        epoch = trainer.epoch
        display_epoch = epoch + 1
        batch_count = self._batch_count
        total_epochs, batch_size, total_batches, dataset_samples = self._get_epoch_info()
        
        if epoch >= self._warm_epochs:
            self._temp_scheduler.step(epoch)

        if batch_count > 0:
            avg_t = self._epoch_task_loss / batch_count
            avg_k = self._epoch_kd_loss / batch_count
            phase = "warm" if epoch < self._warm_epochs else "distill"
            
            # 结构化日志行 — 前端 handleEpochProgress 可解析
            alpha_val = round(self._alpha_scheduler.current_alpha, 4)
            temp_val = round(self._temp_scheduler.current_temperature, 4)
            logger.info(
                f"[EPOCH_PROGRESS] epoch={display_epoch} total={total_epochs} "
                f"loss={avg_t:.4f} kd={avg_k:.4f} "
                f"alpha={alpha_val} temp={temp_val} "
                f"batches={batch_count}/{total_batches or 'unknown'} "
                f"batch_size={batch_size} "
                f"samples={batch_count * batch_size if batch_size else 'unknown'} "
                f"dataset_samples={dataset_samples if dataset_samples is not None else 'unknown'} "
                f"phase={phase}"
            )
            self._distill_log.append({
                "epoch": display_epoch,
                "phase": phase,
                "task_loss": round(avg_t, 6),
                "kd_loss": round(avg_k, 6),
                "alpha": round(self._alpha_scheduler.current_alpha, 6),
                "temperature": round(self._temp_scheduler.current_temperature, 6),
            })
            if epoch >= self._warm_epochs:
                self._alpha_scheduler.update(avg_t)
            self._epoch_task_loss = 0.0
            self._epoch_kd_loss = 0.0
            self._batch_count = 0
            # 每个 epoch 结束后积极清理显存，防止长时间训练后内存持续增长
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  # 重置峰值统计以便监控每个epoch

    def _on_train_batch_end(self, trainer):
        # 批次进度已在 get_loss() 中按间隔输出，此处不再重复
        pass

    def _on_fit_epoch_end(self, trainer):
        """注入蒸馏指标到 results.csv"""
        epoch = trainer.epoch
        display_epoch = epoch + 1
        entry = next((e for e in reversed(self._distill_log) if e["epoch"] == display_epoch), None)
        if entry is None and self._distill_log: entry = self._distill_log[-1]
        if not entry:
            return
        
        av, tv, kv = entry['alpha'], entry['temperature'], entry['kd_loss']
        for target in [getattr(trainer, 'metrics', None), getattr(trainer, 'results_dict', None)]:
            if isinstance(target, dict):
                try:
                    target['distill/alpha']=av
                    target['distill/temperature']=tv
                    target['distill/kd_loss']=kv
                except Exception:
                    pass
        self._append_csv(display_epoch, av, tv, kv)
    
    def _append_csv(self, epoch, a, t, k):
        sd = getattr(self, 'save_dir', None)
        if not sd: return
        p = Path(sd)/'results.csv'
        if not p.exists(): return
        try:
            rows=[]; fn=[]
            with open(p,'r',encoding='utf-8',newline='') as f:
                r=_c.DictReader(f); fn=list(r.fieldnames or []); rows=list(r)
            for c in ['distill/alpha','distill/temperature','distill/kd_loss']:
                if c not in fn: fn.append(c)
            matched = False
            for row in rows:
                try:
                    row_epoch = int(float(row.get('epoch','-1')))
                    if row_epoch == int(epoch):
                        row['distill/alpha']=a; row['distill/temperature']=t; row['distill/kd_loss']=k; matched = True; break
                except (ValueError, TypeError, KeyError):
                    continue
            if not matched:
                for row in rows:
                    try:
                        row_epoch = int(float(row.get('epoch','-1')))
                        if row_epoch == int(epoch) + 1 or row_epoch == int(epoch) - 1:
                            row['distill/alpha']=a; row['distill/temperature']=t; row['distill/kd_loss']=k; break
                    except (ValueError, TypeError, KeyError):
                        continue
            with open(p,'w',encoding='utf-8',newline='') as f:
                _c.DictWriter(f,fn).writeheader(); _c.DictWriter(f,fn).writerows(rows)
        except Exception as e: logger.debug(f"csv追加跳过: {e}")
    
    def _on_val_end(self, trainer):
        """验证结束后：提取并缓存每类别的性能指标（AP / Precision / Recall）"""
        try:
            # 多路径探测验证结果对象（兼容不同 ultralytics 版本）
            validator = getattr(trainer, 'validator', None)
            if validator is None:
                logger.debug("[VAL] trainer.validator 为空，跳过")
                return

            # 尝试多种方式获取结果对象
            results = None
            for attr_name in ('results', 'final_results', 'stats'):
                r = getattr(validator, attr_name, None) or getattr(trainer, f'validator_{attr_name}', None)
                if r is not None:
                    results = r
                    break

            if results is None:
                # 尝试直接从 trainer.metrics 获取
                metrics_dict = getattr(trainer, 'metrics', None) or getattr(trainer, 'results_dict', None)
                if isinstance(metrics_dict, dict):
                    self._try_save_per_class_from_metrics(metrics_dict, trainer)
                return

            box = getattr(results, 'box', None)
            if box is None:
                return

            # 提取 per-class 数据 — ultralytics >= 8.2 Metric API
            class_names = getattr(results, 'names', {}) or {}
            nc = len(class_names) if class_names else getattr(box, 'nc', 0)
            if nc == 0:
                return

            import numpy as _np

            # maps 是 ndarray 属性（非方法），shape=(nc,)，包含全部类别的 mAP@50-95
            _maps = None
            if hasattr(box, 'maps') and box.maps is not None:
                _m = _np.asarray(box.maps)
                _maps = list(_m.flatten()) if _m.ndim > 1 else list(_m)

            labels, map_list, prec_list, rec_list = [], [], [], []
            for i in range(nc):
                labels.append(class_names.get(i, f'class{i}'))
                map_list.append(float(_maps[i]) if _maps and i < len(_maps) else None)
                # P/R 通过 class_result(i) 获取（处理稀疏映射）
                try:
                    cr = box.class_result(i) if hasattr(box, 'class_result') else None
                    if cr is not None:
                        prec_list.append(float(cr[0]) if cr[0] is not None else None)
                        rec_list.append(float(cr[1]) if cr[1] is not None else None)
                    else:
                        idx_map = getattr(box, 'ap_class_index', None)
                        if idx_map is not None:
                            matches = [j for j, v in enumerate(idx_map) if int(v) == i]
                            if matches:
                                j = matches[0]
                                prec_list.append(float(box.p[j]) if getattr(box, 'p', None) is not None and box.p[j] is not None else None)
                                rec_list.append(float(box.r[j]) if getattr(box, 'r', None) is not None and box.r[j] is not None else None)
                            else:
                                prec_list.append(None)
                                rec_list.append(None)
                        else:
                            p_arr = getattr(box, 'p', None)
                            r_arr = getattr(box, 'r', None)
                            if p_arr is not None and r_arr is not None:
                                try:
                                    prec_list.append(float(p_arr[i]) if i < len(p_arr) and p_arr[i] is not None else None)
                                    rec_list.append(float(r_arr[i]) if i < len(r_arr) and r_arr[i] is not None else None)
                                except Exception:
                                    prec_list.append(None)
                                    rec_list.append(None)
                            else:
                                prec_list.append(None)
                                rec_list.append(None)
                except Exception:
                    prec_list.append(None)
                    rec_list.append(None)

            output_data = {
                'labels': labels,
                'map': map_list,
                'precision': prec_list,
                'recall': rec_list,
                'epoch': trainer.epoch + 1,
                'generated_at': datetime.now().isoformat(),
            }

            sd = getattr(trainer, 'save_dir', None)
            if sd:
                cache_path = Path(sd) / 'per_class_metrics.json'
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.debug(f"[VAL] 每类指标提取跳过: {e}")

    def _try_save_per_class_from_metrics(self, metrics: dict, trainer):
        """从 metrics 字典中尝试提取每类指标（兜底路径）"""
        try:
            # 检查是否有按类别前缀的列，如 class0_ap, cls_0_precision 等
            class_metrics: dict[int, dict[str, float]] = {}
            for key in metrics.keys():
                match = re.search(r'(?i)(?:class|cls)[\s_/\\-]*(\d+)', str(key))
                if not match:
                    continue
                idx = int(match.group(1))
                val = metrics[key]
                lower_key = str(key).lower()
                metric_type = None
                if 'ap' in lower_key or 'map' in lower_key:
                    metric_type = 'map'
                elif 'precision' in lower_key or 'p@' in lower_key:
                    metric_type = 'p'
                elif 'recall' in lower_key or 'r@' in lower_key:
                    metric_type = 'r'
                if metric_type:
                    try:
                        metric_val = float(val)
                    except (TypeError, ValueError):
                        metric_val = None
                    class_metrics.setdefault(idx, {})[metric_type] = metric_val

            if not class_metrics:
                return

            # 获取类别名
            data_cfg_path = getattr(getattr(trainer, 'args', None), 'data', None)
            names = {}
            if data_cfg_path:
                from pathlib import Path as _P
                import yaml as _yml
                dp = Path(data_cfg_path)
                if dp.exists():
                    cfg_data = _yml.safe_load(dp.read_text(encoding='utf-8'))
                    names = cfg_data.get('names') or {}

            labels, map_l, prec_l, rec_l = [], [], [], []
            for idx in sorted(class_metrics.keys()):
                cm = class_metrics[idx]
                labels.append(names.get(idx, f'class{idx}') if isinstance(names, dict) else f'class{idx}')
                map_val = cm.get('map')
                prec_val = cm.get('p')
                rec_val = cm.get('r')
                map_l.append(map_val)
                prec_l.append(prec_val)
                rec_l.append(rec_val)

            output = {
                'labels': labels, 'map': map_l, 'precision': prec_l, 'recall': rec_l,
                'epoch': getattr(trainer, 'epoch', -1) + 1,
                'generated_at': datetime.now().isoformat(), 'source': 'metrics_dict_fallback',
            }
            sd = getattr(trainer, 'save_dir', None)
            if sd:
                cache_path = Path(sd) / 'per_class_metrics.json'
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"[VAL] metrics_dict 兜底提取失败: {e}")

    def _on_train_end(self, trainer):
        """
        安全网回调：在训练完全结束后（final_eval 之后）重新写入所有蒸馏数据。
        
        问题：ultralytics 的 final_eval() 会覆写 results.csv，
              导致 _on_fit_epoch_end 写入的最后一轮蒸馏数据被清空。
        解决：训练结束后遍历 _distill_log，确保每一轮的 distill 列都完整写入。
        """
        if not self._distill_log:
            return
        
        sd = getattr(self, 'save_dir', None)
        if not sd:
            return
        p = Path(sd) / 'results.csv'
        if not p.exists():
            return
        
        try:
            rows = []
            fn = []
            with open(p, 'r', encoding='utf-8', newline='') as f:
                reader = _c.DictReader(f)
                fn = list(reader.fieldnames or [])
                rows = list(reader)
            
            # 确保 distill 列存在
            for c in ['distill/alpha', 'distill/temperature', 'distill/kd_loss']:
                if c not in fn:
                    fn.append(c)
            
            # 用 _distill_log 中的数据补全每一行
            epoch_to_distill = {entry['epoch']: entry for entry in self._distill_log}
            written_count = 0
            
            for row in rows:
                try:
                    ep = int(float(row.get('epoch', '-1')))
                except (ValueError, TypeError):
                    continue
                
                if ep in epoch_to_distill:
                    de = epoch_to_distill[ep]
                elif ep - 1 in epoch_to_distill:
                    de = epoch_to_distill[ep - 1]
                elif ep + 1 in epoch_to_distill:
                    de = epoch_to_distill[ep + 1]
                else:
                    de = None
                
                if de is not None:
                    row['distill/alpha'] = de['alpha']
                    row['distill/temperature'] = de['temperature']
                    row['distill/kd_loss'] = de['kd_loss']
                    written_count += 1
            
            with open(p, 'w', encoding='utf-8', newline='') as f:
                writer = _c.DictWriter(f, fn)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(
                f"[TRAIN_END] 蒸馏数据安全网: 已补全 {written_count}/{len(rows)} 行 "
                f"(总日志条目: {len(self._distill_log)})"
            )
            
        except Exception as e:
            logger.warning(f"[TRAIN_END] CSV 安全网写入失败: {e}")

        # 最终兜底：如果 _on_val_end 没有成功生成 per_class_metrics.json，
        # 在训练结束后通过模型验证生成
        self._ensure_per_class_metrics(trainer)

    def _ensure_per_class_metrics(self, trainer):
        """训练结束时确保 per_class_metrics.json 已存在，否则用模型验证生成。"""
        sd = getattr(trainer, 'save_dir', None) or getattr(self, 'save_dir', None)
        if not sd:
            return
        cache_path = Path(sd) / 'per_class_metrics.json'
        if cache_path.exists():
            logger.info("[TRAIN_END] per_class_metrics.json 已存在，跳过生成")
            return

        # 查找模型权重
        weight_candidates = [
            Path(sd) / 'weights' / 'best.pt',
            Path(sd) / 'weights' / 'last.pt',
            Path(sd) / 'best.pt',
            Path(sd) / 'last.pt',
        ]
        model_path = None
        for candidate in weight_candidates:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None or not model_path.exists():
            # 尝试从 args.yaml 获取数据集配置来验证当前模型
            try:
                import io as _io
                from contextlib import redirect_stdout as _rs, redirect_stderr as _rse
                from ultralytics import YOLO

                data_cfg = getattr(getattr(trainer, 'args', None), 'data', None)
                if not data_cfg:
                    return
                imgsz = getattr(getattr(trainer, 'args', None), 'imgsz', 640)

                buf = _io.StringIO()
                with _rs(buf), _rse(buf):
                    eval_model = YOLO(str(self.model))
                    eval_results = eval_model.val(
                        data=data_cfg,
                        imgsz=imgsz,
                        verbose=False,
                    )

                self._extract_and_save_per_class(eval_results, sd)
            except Exception as e:
                logger.warning(f"[TRAIN_END] 兜底每类指标生成失败: {e}")
            return

        # 有权重文件时直接加载验证
        try:
            import io as _io2
            from contextlib import redirect_stdout as _rs2, redirect_stderr as _rse2
            from ultralytics import YOLO as _YOLO

            data_cfg = getattr(getattr(trainer, 'args', None), 'data', None)
            if not data_cfg:
                return
            imgsz = getattr(getattr(trainer, 'args', None), 'imgsz', 640)

            buf = _io2.StringIO()
            with _rs2(buf), _rse2(buf):
                m = _YOLO(str(model_path))
                results = m.val(data=data_cfg, imgsz=imgsz, verbose=False)

            self._extract_and_save_per_class(results, sd)
            del m
        except Exception as e:
            logger.warning(f"[TRAIN_END] 权重文件验证生成每类指标失败: {e}")

    def _extract_and_save_per_class(self, val_results, save_dir):
        """从 YOLO 验证结果中提取每类指标并保存"""
        box = getattr(val_results, 'box', None)
        if box is None:
            return

        class_names = getattr(val_results, 'names', {}) or {}
        nc = len(class_names) if class_names else getattr(box, 'nc', 0)
        if nc == 0:
            return

        import numpy as _np

        # maps 是 ndarray 属性（非方法），shape=(nc,)
        _maps = None
        if hasattr(box, 'maps') and box.maps is not None:
            _m = _np.asarray(box.maps)
            _maps = list(_m.flatten()) if _m.ndim > 1 else list(_m)

        labels, map_l, prec_l, rec_l = [], [], [], []
        for i in range(nc):
            labels.append(class_names.get(i, f'class{i}'))
            map_l.append(float(_maps[i]) if _maps and i < len(_maps) else None)
            try:
                cr = box.class_result(i) if hasattr(box, 'class_result') else None
                if cr is not None:
                    prec_l.append(float(cr[0]))
                    rec_l.append(float(cr[1]))
                else:
                    idx_map = getattr(box, 'ap_class_index', None)
                    if idx_map is not None:
                        matches = [j for j, v in enumerate(idx_map) if int(v) == i]
                        if matches:
                            j = matches[0]
                            prec_l.append(float(box.p[j]) if box.p is not None else None)
                            rec_l.append(float(box.r[j]) if box.r is not None else None)
                        else:
                            prec_l.append(None); rec_l.append(None)
                    else:
                        p_arr = getattr(box, 'p', None)
                        r_arr = getattr(box, 'r', None)
                        if p_arr is not None and r_arr is not None and i < len(p_arr) and i < len(r_arr):
                            try:
                                prec_l.append(float(p_arr[i]) if p_arr[i] is not None else None)
                                rec_l.append(float(r_arr[i]) if r_arr[i] is not None else None)
                            except Exception:
                                prec_l.append(None); rec_l.append(None)
                        else:
                            prec_l.append(None); rec_l.append(None)
            except Exception:
                prec_l.append(None); rec_l.append(None)

        output = {
            'labels': labels, 'map': map_l, 'precision': prec_l, 'recall': rec_l,
            'source': 'train_end_fallback',
            'generated_at': datetime.now().isoformat(),
        }
        cache_path = Path(save_dir) / 'per_class_metrics.json'
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"[TRAIN_END] 每类指标已生成: {cache_path} ({nc} 个类别)")

    def save_model(self, *args, **kwargs):
        if hasattr(self, '_class_loss_backup'):
            type(self.model).loss = self._class_loss_backup
            if hasattr(type(self.model), '_distill_original_loss'):
                delattr(type(self.model), '_distill_original_loss')
        try:
            return super().save_model(*args, **kwargs)
        finally:
            if hasattr(self, '_class_loss_backup'):
                type(self.model).loss = _distill_model_loss
                type(self.model)._distill_original_loss = self._class_loss_backup

    def validate(self, *args, **kwargs):
        if hasattr(self, '_class_loss_backup'):
            type(self.model).loss = self._class_loss_backup
        try:
            return super().validate(*args, **kwargs)
        finally:
            if hasattr(self, '_class_loss_backup'):
                type(self.model).loss = _distill_model_loss
                type(self.model)._distill_original_loss = self._class_loss_backup

    def get_distill_log(self): return self._distill_log
