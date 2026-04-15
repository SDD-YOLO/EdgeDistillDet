"""
core/distillation/adaptive_kd_trainer.py
==========================================
自适应知识蒸馏训练器（AdaptiveKDTrainer）— 最终修复版 v3

核心设计原则（v3 彻底重构）：
  1. 绝不修改 model 对象的持久化状态（避免 checkpoint 和 EMA 兼容性问题）
  2. 通过 DetectionModel.loss 全局注入蒸馏损失，训练结束后恢复原始 loss
  3. 原始损失函数存储在 trainer 实例属性上（不污染模型）
  4. 蒸馏指标通过多路注入写入 results.csv
"""

import os
import json
import math
import logging
import csv as _c
import io
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer, DEFAULT_CFG

from core.distillation.loss_functions import (
    CompositiveDistillLoss,
    CosineTemperatureScheduler,
)

logger = logging.getLogger("DistillTrainer")
_KD_TRAINERS = weakref.WeakKeyDictionary()


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
        from .adaptive_kd_trainer import _safe_scalar
        v = _safe_scalar(task_loss)
        self._ema_loss = v if self._ema_loss is None else self.ema_decay * self._ema_loss + (1-self.ema_decay)*v
        if self._prev_ema is not None:
            d = (self._prev_ema - self._ema_loss)/(self._prev_ema + 1e-8)
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha + self.lr_alpha*(self.delta_target-d)))
        self._prev_ema = self._ema_loss
        return self.alpha


def _safe_scalar(value):
    """安全标量转换 — 处理任意维度张量"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0: return 0.0
        elif value.numel() == 1: return value.detach().item()
        else: return value.detach().mean().item()
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def _distill_model_loss(model, batch, preds=None):
    trainer = _KD_TRAINERS.get(model)
    if trainer is None:
        original_loss_fn = getattr(type(model), '_distill_original_loss', None)
        if original_loss_fn is None:
            raise RuntimeError("Distill loss wrapper invoked without registered trainer")
        return original_loss_fn(model, batch, preds=preds)

    logger.warning(f"[DEBUG] _distill_model_loss invoked epoch={getattr(trainer, 'epoch', None)} warm={getattr(trainer, 'epoch', 0) < trainer._warm_epochs}")
    if preds is None:
        preds = model.forward(batch["img"])

    loss_task, loss_items = trainer._orig_loss(batch, preds=preds)
    num_classes = getattr(model, 'nc', 80)
    current_temp = trainer._temp_scheduler.current_temperature
    teacher_model = trainer.teacher_model
    teacher_device = next(teacher_model.parameters()).device
    teacher_dtype = next(teacher_model.parameters()).dtype
    teacher_img = batch["img"].to(teacher_device).to(teacher_dtype)
    device_type = teacher_img.device.type
    with torch.no_grad():
        with torch.amp.autocast(device_type, enabled=False):
            teacher_preds = teacher_model(teacher_img)

    loss_distill, _detail = trainer._distill_loss(
        student_preds=preds,
        teacher_preds=teacher_preds,
        num_classes=num_classes,
        temperature=current_temp,
    )
    alpha = trainer._alpha_scheduler.current_alpha
    total_loss = (1 - alpha) * loss_task + alpha * loss_distill

    trainer._accumulate_losses(task_loss=loss_task, kd_loss=loss_distill)
    return total_loss, loss_items


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
        # 蒸馏是否激活（warm-up 后自动开启，无需手动控制）
        # 去掉 _distill_active 标志，改用 epoch 直接判断（与旧版 v2 行为一致）
    
    def setup_model(self):
        super().setup_model()
        
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
        
        self._load_teacher()
        self._init_distill()

        # 为模型注入蒸馏损失包装器，并保留原始 loss 方法以便热身阶段和权重保存。
        _KD_TRAINERS[self.model] = self
        original_loss_fn = self._orig_loss.__func__ if hasattr(self._orig_loss, '__func__') else self._orig_loss
        type(self.model).loss = _distill_model_loss
        type(self.model)._distill_original_loss = original_loss_fn
        self._class_loss_backup = original_loss_fn
        logger.warning("[DEBUG] AdaptiveKDTrainer patched DetectionModel.loss for distillation")

    def _load_teacher(self):
        if not self._teacher_path or not os.path.exists(self._teacher_path):
            raise FileNotFoundError(f"教师模型不存在: {self._teacher_path}")
        logger.info(f"加载教师模型: {self._teacher_path}")
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            teacher = YOLO(self._teacher_path, verbose=False)
        self.teacher_model = teacher.model.to(self.device).eval()
        for p in self.teacher_model.parameters(): p.requires_grad = False
        logger.info("教师模型就绪 ✓")
    
    def _init_distill(self):
        total_epochs = self.args.epochs if hasattr(self.args, "epochs") else 150
        self._alpha_scheduler = AdaptiveAlphaScheduler(alpha_init=self._alpha_init)
        self._temp_scheduler = CosineTemperatureScheduler(T_max=self._T_max, T_min=self._T_min, total_epochs=total_epochs)
        self._distill_loss = CompositiveDistillLoss(w_kd=self._w_kd, w_focal=self._w_focal, w_feat=self._w_feat,
                                                     temperature=self._T_max, scale_boost=self._scale_boost, focal_gamma=self._focal_gamma).to(self.device)
        logger.info(f"蒸馏组件就绪 | T:[{self._T_max}→{self._T_min}] α:{self._alpha_init} warm:{self._warm_epochs}")
        
        self.add_callback("on_train_epoch_end", self._on_epoch_end)
        self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        self.add_callback("on_val_end", self._on_val_end)
        logger.warning("[DEBUG] AdaptiveKDTrainer callbacks registered: on_train_epoch_end, on_fit_epoch_end, on_val_end")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 可选：兼容旧版 ultralytics 的 get_loss 扩展点
    # 当前训练流程实际通过 DetectionModel.loss 注入蒸馏损失
    # ══════════════════════════════════════════════════════════════════════════
    def get_loss(self):
        """
        重写损失计算：warm-up 用原始损失，之后注入蒸馏损失。
        
        这是唯一需要重写的方法。ultralytics 的 train loop 中：
          batch forward → model(batch) → self.get_loss() 
        因此在这里注入蒸馏逻辑是最干净的方案。
        """
        if self._orig_loss is None:
            return super().get_loss()

        if not getattr(self, '_loss_called', False):
            self._loss_called = True

        epoch = getattr(self, 'epoch', 0)
        
        # Warm-up 阶段：纯原始损失
        if epoch < self._warm_epochs:
            loss, loss_items = super().get_loss()
            self._accumulate_losses(task_loss=loss)
            return loss, loss_items
        
        # 蒸馏阶段（epoch >= warm_epochs 后自动激活）
        if not self._distill_entered:
            self._distill_entered = True
            logger.info(f"[DISTILL] 进入蒸馏阶段 epoch={epoch}")
        
        # 1) 计算原始任务损失
        # 注意：不能直接调 super().get_loss() 因为它会触发前向传播，
        # 我们需要手动控制前向传播来获取学生预测
        batch = self.last_batch
        img = batch['img']
        if img.dtype == torch.float16:
            img = img.float()
        
        device_type = img.device.type
        
        # 学生模型前向传播
        with torch.amp.autocast(device_type, enabled=False):
            student_preds = self.model(img)
        
        # 原始损失（带学生预测）
        loss_task, loss_items = self._orig_loss(batch, preds=student_preds)
        
        # 2) 教师模型前向传播（无梯度）
        teacher_img = batch['img'].float()
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=False):
                teacher_preds = self.teacher_model(teacher_img)
        
        # 3) 计算蒸馏损失
        num_classes = getattr(self.model, 'nc', 80)
        current_temp = self._temp_scheduler.current_temperature
        loss_distill, _detail = self._distill_loss(
            student_preds=student_preds,
            teacher_preds=teacher_preds,
            num_classes=num_classes,
            temperature=current_temp,
        )
        
        # 4) 加权组合
        alpha = self._alpha_scheduler.current_alpha
        total_loss = (1 - alpha) * loss_task + alpha * loss_distill
        
        self._accumulate_losses(task_loss=loss_task, kd_loss=loss_distill)
        
        return total_loss, loss_items
    
    def _setup_train(self):
        """重写 setup_train：强制 workers=0 避免 Windows 多进程问题"""
        orig_workers = getattr(self.args, 'workers', 0)
        self.args.workers = 0
        try: super()._setup_train()
        finally: self.args.workers = orig_workers
    
    def _accumulate_losses(self, task_loss=None, kd_loss=None):
        if task_loss is not None: self._epoch_task_loss += _safe_scalar(task_loss)
        if kd_loss is not None: self._epoch_kd_loss += _safe_scalar(kd_loss)
        self._batch_count += 1
    
    def _on_epoch_end(self, trainer):
        epoch = trainer.epoch
        batch_count = self._batch_count
        logger.warning(f"[DEBUG] _on_epoch_end triggered epoch={epoch} batch_count={batch_count}")
        if epoch >= self._warm_epochs:
            self._temp_scheduler.step(epoch)

        if batch_count > 0:
            avg_t = self._epoch_task_loss / batch_count
            avg_k = self._epoch_kd_loss / batch_count
            phase = "warm" if epoch < self._warm_epochs else "distill"
            logger.info(
                f"Epoch[{epoch:4d}] {phase:>6} | t={avg_t:.4f} kd={avg_k:.4f} "
                f"a={self._alpha_scheduler.current_alpha:.3f} T={self._temp_scheduler.current_temperature:.2f}"
            )
            self._distill_log.append({
                "epoch": epoch,
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
        else:
            logger.warning(f"[DEBUG] _on_epoch_end skipped append because batch_count=0")
    
    def _on_fit_epoch_end(self, trainer):
        """注入蒸馏指标到 results.csv"""
        epoch = trainer.epoch
        logger.warning(f"[DEBUG] _on_fit_epoch_end triggered epoch={epoch} distill_log_len={len(self._distill_log)}")
        entry = next((e for e in reversed(self._distill_log) if e["epoch"] == epoch), None)
        if entry is None and self._distill_log: entry = self._distill_log[-1]
        if not entry:
            logger.warning(f"[DEBUG] _on_fit_epoch_end no entry found for epoch={epoch}")
            return
        
        av, tv, kv = entry['alpha'], entry['temperature'], entry['kd_loss']
        for target in [getattr(trainer, 'metrics', None), getattr(trainer, 'results_dict', None)]:
            if isinstance(target, dict):
                try:
                    target['distill/alpha']=av
                    target['distill/temperature']=tv
                    target['distill/kd_loss']=kv
                except Exception as e:
                    logger.warning(f"[DEBUG] failed to inject distill metrics into target dict: {e}")
        self._append_csv(epoch, av, tv, kv)
    
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
            for row in rows:
                try:
                    if int(float(row.get('epoch','-1')))==int(epoch):
                        row['distill/alpha']=a; row['distill/temperature']=t; row['distill/kd_loss']=k; break
                except (ValueError, TypeError, KeyError): continue
            with open(p,'w',encoding='utf-8',newline='') as f:
                _c.DictWriter(f,fn).writeheader(); _c.DictWriter(f,fn).writerows(rows)
        except Exception as e: logger.debug(f"csv追加跳过: {e}")
    
    def _on_val_end(self, trainer): pass

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
