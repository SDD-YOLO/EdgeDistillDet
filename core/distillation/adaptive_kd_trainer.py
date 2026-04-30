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

from core.distillation.common import safe_scalar, w_feat_to_scalar
from core.distillation.loss_functions import (
    CompositiveDistillLoss,
    CosineTemperatureScheduler,
)

logger = logging.getLogger("DistillTrain")


def _scatter_pr_from_ultralytics_box(box, nc: int):
    """
    将 ultralytics Metric.box 中按紧凑下标排列的 P/R 映射到全局类别 id 0..nc-1。
    class_result(i) 与 box.p[i] 中的 i 均为「第 i 个有 AP 条目的类别」，与验证打印循环一致，
    不能与 range(nc) 的全局类别下标混用。
    """
    import numpy as _np

    out_p = [None] * nc
    out_r = [None] * nc
    ap_idx = getattr(box, 'ap_class_index', None)
    p_arr = getattr(box, 'p', None)
    r_arr = getattr(box, 'r', None)
    if ap_idx is None or p_arr is None or r_arr is None:
        return out_p, out_r
    p_arr = _np.asarray(p_arr).ravel()
    r_arr = _np.asarray(r_arr).ravel()
    for j, cid in enumerate(ap_idx):
        cid = int(cid)
        if not (0 <= cid < nc):
            continue
        if j < len(p_arr):
            pv = p_arr[j]
            if pv is not None and not (isinstance(pv, (float, _np.floating)) and _np.isnan(pv)):
                out_p[cid] = float(pv)
        if j < len(r_arr):
            rv = r_arr[j]
            if rv is not None and not (isinstance(rv, (float, _np.floating)) and _np.isnan(rv)):
                out_r[cid] = float(rv)
    return out_p, out_r


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

def _resolve_original_model_loss(model):
    """
    解析 DetectionModel 的真实原始 loss 函数。
    """
    current_loss = getattr(model, "loss", None)
    if current_loss is None:
        return None
    
    # 直接从 type 上获取备份的原始 loss
    original_fn = getattr(type(model), "_distill_original_loss", None)
    if original_fn is not None:
        return original_fn.__get__(model, type(model)) if hasattr(original_fn, '__get__') else original_fn
    
    # 没有备份，就返回当前的（首次 setup 时）
    return current_loss

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
                      scale_boost=2.0, focal_gamma=2.0, **extra_kwargs):
        cls._kd_class_params = {
            "teacher_path": teacher_path, "alpha_init": alpha_init, "T_max": T_max,
            "T_min": T_min, "warm_epochs": warm_epochs, "w_kd": w_kd, "w_focal": w_focal,
            "w_feat": w_feat, "scale_boost": scale_boost, "focal_gamma": focal_gamma,
        }
        if extra_kwargs:
            cls._kd_class_params.update(extra_kwargs)
    
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
        self._resume_prev_kd_loss = None
        self._resume_prev_epoch = None
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
        self._orig_loss = _resolve_original_model_loss(self.model)
        if self._orig_loss is None:
            logger.error("[FATAL] 找不到 model.loss!")
            return
        
        kd = AdaptiveKDTrainer._kd_class_params
        if kd and kd.get("teacher_path"):
            self._teacher_path = kd["teacher_path"]
            self._warm_epochs = int(kd.get("warm_epochs", 5))
            self._w_kd=float(kd.get("w_kd", 0.5)); self._w_focal=float(kd.get("w_focal", 0.3))
            self._w_feat = w_feat_to_scalar(kd.get("w_feat", 0.0))
            self._T_max=float(kd.get("T_max", 6.0)); self._T_min=float(kd.get("T_min", 1.5))
            self._alpha_init=float(kd.get("alpha_init", 0.5))
            self._scale_boost=float(kd.get("scale_boost", 2.0)); self._focal_gamma=float(kd.get("focal_gamma", 2.0))
            logger.info(f"蒸馏参数 | teacher={os.path.basename(self._teacher_path)} | "
                       f"α={self._alpha_init} T=[{self._T_max},{self._T_min}]")
        
        logger.info(f"[INIT 5/6] 加载教师模型: {os.path.basename(self._teacher_path)} ...")
        self._load_teacher()
        self._init_distill()
        logger.info(f"[INIT 5/6] 蒸馏损失函数 + 调度器 就绪 ✓")

        # 不再替换 model.loss — Ultralytics 8.4.19 使用 criterion 计算损失，
        # 替换 model.loss 会导致 criterion 内部拿到的 preds 格式错误。
        # 蒸馏逻辑完全通过重写的 get_loss() 注入。
        logger.info("[DEBUG] AdaptiveKDTrainer 使用 get_loss() 注入蒸馏逻辑（不修改 model.loss）")

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

    def _distill_state_path(self) -> Path | None:
        sd = getattr(self, "save_dir", None)
        if sd:
            return Path(sd) / "distill_state.json"
        return None

    def _save_distill_state(self, epoch: int) -> None:
        state_path = self._distill_state_path()
        if state_path is None or self._alpha_scheduler is None:
            return
        last_kd = None
        current_temp = None
        if self._distill_log:
            try:
                last_kd = float(self._distill_log[-1].get("kd_loss"))
            except Exception:
                last_kd = None
        if self._temp_scheduler is not None:
            try:
                current_temp = float(self._temp_scheduler.current_temperature)
            except Exception:
                current_temp = None
        payload = {
            "epoch": int(epoch),
            "alpha_scheduler": {
                "alpha": float(self._alpha_scheduler.alpha),
                "alpha_min": float(self._alpha_scheduler.alpha_min),
                "alpha_max": float(self._alpha_scheduler.alpha_max),
                "lr_alpha": float(self._alpha_scheduler.lr_alpha),
                "ema_decay": float(self._alpha_scheduler.ema_decay),
                "delta_target": float(self._alpha_scheduler.delta_target),
                "ema_loss": None if self._alpha_scheduler._ema_loss is None else float(self._alpha_scheduler._ema_loss),
                "prev_ema": None if self._alpha_scheduler._prev_ema is None else float(self._alpha_scheduler._prev_ema),
            },
            "last_kd_loss": last_kd,
            "temperature": current_temp,
        }
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"[DISTILL_STATE] 保存失败: {e}")

    def _restore_alpha_from_results_csv(self) -> float | None:
        sd = getattr(self, "save_dir", None)
        if not sd:
            return None
        csv_path = Path(sd) / "results.csv"
        if not csv_path.exists():
            return None
        last_alpha = None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = _c.DictReader(f)
                for row in reader:
                    raw = row.get("distill/alpha")
                    if raw is None:
                        continue
                    text = str(raw).strip()
                    if not text:
                        continue
                    try:
                        last_alpha = float(text)
                    except Exception:
                        continue
        except Exception:
            return None
        return last_alpha

    def _restore_distill_baseline_from_results_csv(self) -> dict | None:
        sd = getattr(self, "save_dir", None)
        if not sd:
            return None
        csv_path = Path(sd) / "results.csv"
        if not csv_path.exists():
            return None
        last = None
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = _c.DictReader(f)
                for row in reader:
                    last = row
        except Exception:
            return None
        if not last:
            return None

        def _safe_float(v):
            try:
                text = str(v).strip()
                return float(text) if text else None
            except Exception:
                return None

        def _safe_int(v):
            try:
                text = str(v).strip()
                return int(float(text)) if text else None
            except Exception:
                return None

        return {
            "epoch": _safe_int(last.get("epoch")),
            "alpha": _safe_float(last.get("distill/alpha")),
            "kd_loss": _safe_float(last.get("distill/kd_loss")),
            "temperature": _safe_float(last.get("distill/temperature")),
        }

    def _restore_distill_state_if_needed(self) -> None:
        if not bool(getattr(self.args, "resume", False)):
            return
        if self._alpha_scheduler is None:
            return
        restored_from = None
        restored_alpha = None
        restored_temp_from = None
        state_path = self._distill_state_path()
        if state_path is not None and state_path.exists():
            try:
                disk = json.loads(state_path.read_text(encoding="utf-8"))
                alpha_state = (disk or {}).get("alpha_scheduler", {}) or {}
                alpha = alpha_state.get("alpha")
                if alpha is not None:
                    self._alpha_scheduler.alpha = float(alpha)
                    self._alpha_scheduler._ema_loss = alpha_state.get("ema_loss")
                    self._alpha_scheduler._prev_ema = alpha_state.get("prev_ema")
                    restored_alpha = float(self._alpha_scheduler.alpha)
                    restored_from = "distill_state.json"
                saved_temp = (disk or {}).get("temperature")
                if self._temp_scheduler is not None and saved_temp is not None:
                    try:
                        self._temp_scheduler._current_T = float(saved_temp)
                        restored_temp_from = "distill_state.json"
                    except Exception:
                        pass
                try:
                    self._resume_prev_kd_loss = float((disk or {}).get("last_kd_loss"))
                except Exception:
                    self._resume_prev_kd_loss = None
                try:
                    self._resume_prev_epoch = int((disk or {}).get("epoch"))
                except Exception:
                    self._resume_prev_epoch = None
            except Exception as e:
                logger.debug(f"[DISTILL_STATE] 读取状态失败: {e}")
        if restored_alpha is None:
            csv_baseline = self._restore_distill_baseline_from_results_csv()
            csv_alpha = csv_baseline.get("alpha") if isinstance(csv_baseline, dict) else None
            if csv_alpha is not None:
                self._alpha_scheduler.alpha = float(csv_alpha)
                restored_alpha = float(self._alpha_scheduler.alpha)
                restored_from = "results.csv"
            if isinstance(csv_baseline, dict):
                if self._resume_prev_kd_loss is None and csv_baseline.get("kd_loss") is not None:
                    self._resume_prev_kd_loss = float(csv_baseline["kd_loss"])
                if self._resume_prev_epoch is None and csv_baseline.get("epoch") is not None:
                    self._resume_prev_epoch = int(csv_baseline["epoch"])
                if self._temp_scheduler is not None and csv_baseline.get("temperature") is not None:
                    try:
                        self._temp_scheduler._current_T = float(csv_baseline["temperature"])
                        restored_temp_from = "results.csv"
                    except Exception:
                        pass
        else:
            # 兼容旧版 state：若缺少 last_kd_loss，回退从 results.csv 获取续训边界基线
            if self._resume_prev_kd_loss is None or self._resume_prev_epoch is None:
                csv_baseline = self._restore_distill_baseline_from_results_csv()
                if isinstance(csv_baseline, dict):
                    if self._resume_prev_kd_loss is None and csv_baseline.get("kd_loss") is not None:
                        self._resume_prev_kd_loss = float(csv_baseline["kd_loss"])
                    if self._resume_prev_epoch is None and csv_baseline.get("epoch") is not None:
                        self._resume_prev_epoch = int(csv_baseline["epoch"])
            # 即使 alpha/kd 基线完整，旧版 state 也可能没有 temperature，需独立回退。
            if self._temp_scheduler is not None and restored_temp_from is None:
                csv_baseline = self._restore_distill_baseline_from_results_csv()
                if isinstance(csv_baseline, dict) and csv_baseline.get("temperature") is not None:
                    try:
                        self._temp_scheduler._current_T = float(csv_baseline["temperature"])
                        restored_temp_from = "results.csv"
                    except Exception:
                        pass
        if restored_alpha is not None:
            logger.info(f"[RESUME_ALPHA] 恢复 alpha={restored_alpha:.6f} 来源={restored_from}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 可选：兼容旧版 ultralytics 的 get_loss 扩展点
    # 当前训练流程实际通过 DetectionModel.loss 注入蒸馏损失
    # ══════════════════════════════════════════════════════════════════════════
    def get_loss(self, batch, preds=None):
        """
        重写损失计算：warm-up 用原始损失，之后注入蒸馏损失。
        """
        if self._orig_loss is None:
            if hasattr(super(), 'get_loss'):
                return super().get_loss(batch, preds)
            raise RuntimeError("Distill trainer missing original model loss function")

        # 确保 preds 已计算（从 train_one_epoch 传入）
        if preds is None:
            preds = self.model(batch['img'])

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

        # Warm-up 阶段：直接使用传入的 preds 计算原始损失
        if warm_phase:
            try:
                loss, loss_items = self._orig_loss(batch, preds=preds)
            except TypeError:
                try:
                    loss, loss_items = self._orig_loss(batch, preds=batch.get('preds') if isinstance(batch, dict) else None)
                except TypeError:
                    loss, loss_items = self._orig_loss(batch)
            self._accumulate_losses(task_loss=loss)
            return loss, loss_items

        # 蒸馏阶段（epoch >= warm_epochs）
        if not self._distill_entered:
            self._distill_entered = True
            logger.info(f"[DISTILL] 进入蒸馏阶段 epoch={epoch}")

        img = batch['img']
        device_type = img.device.type

        # 计算原始任务损失（使用传入的 preds）
        try:
            loss_task, loss_items = self._orig_loss(batch, preds=preds)
        except Exception as e:
            logger.warning(f"[DISTILL] _orig_loss(preds=...) 失败 ({e})，尝试无参调用")
            loss_task, loss_items = self._orig_loss(batch)

        # 教师模型前向传播（无梯度）
        teacher_img = batch['img'].float()
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=False):
                teacher_preds = self.teacher_model(teacher_img)

        # 蒸馏损失计算
        num_classes = getattr(self.model, 'nc', 80)
        current_temp = self._temp_scheduler.current_temperature
        loss_distill, _detail = self._distill_loss(
            student_preds=preds,
            teacher_preds=teacher_preds,
            num_classes=num_classes,
            temperature=current_temp,
        )

        # 立即释放中间张量
        del teacher_preds
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

        # resume 时恢复 alpha 调度状态（优先 distill_state.json，回退 results.csv）
        self._restore_distill_state_if_needed()

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
            # logger 在部分运行环境不会输出到子进程 stdout，前端将收不到 alpha 进度；
            # 同步 print 到 stdout，保证 /api/train/logs 能稳定拿到结构化进度行。
            print(
                f"[EPOCH_PROGRESS] epoch={display_epoch} total={total_epochs} "
                f"loss={avg_t:.4f} kd={avg_k:.4f} "
                f"alpha={alpha_val} temp={temp_val} "
                f"batches={batch_count}/{total_batches or 'unknown'} "
                f"batch_size={batch_size} "
                f"samples={batch_count * batch_size if batch_size else 'unknown'} "
                f"dataset_samples={dataset_samples if dataset_samples is not None else 'unknown'} "
                f"phase={phase}",
                flush=True,
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
            self._save_distill_state(display_epoch)
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

            labels, map_list = [], []
            for i in range(nc):
                labels.append(class_names.get(i, f'class{i}'))
                map_list.append(float(_maps[i]) if _maps and i < len(_maps) else None)
            prec_list, rec_list = _scatter_pr_from_ultralytics_box(box, nc)

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

        labels, map_l = [], []
        for i in range(nc):
            labels.append(class_names.get(i, f'class{i}'))
            map_l.append(float(_maps[i]) if _maps and i < len(_maps) else None)
        prec_l, rec_l = _scatter_pr_from_ultralytics_box(box, nc)

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
