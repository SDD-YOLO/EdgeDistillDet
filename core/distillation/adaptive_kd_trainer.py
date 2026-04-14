"""
core/distillation/adaptive_kd_trainer.py
==========================================
自适应知识蒸馏训练器（AdaptiveKDTrainer）

原创功能：
  1. 动态蒸馏权重 alpha  —— 基于学生 EMA 损失曲率自动调节蒸馏强度
     · 学生下降快 → 减小 alpha，让任务损失主导
     · 学生停滞   → 提升 alpha，加大教师引导
  2. 余弦退火温度调度     —— 见 loss_functions.CosineTemperatureScheduler
  3. 复合蒸馏损失集成     —— 自适应 KD + 焦点 KD + 特征对齐
  4. 详细蒸馏日志         —— 每 epoch 打印 alpha/T/各子损失，便于过程监控
  5. 阶段化训练（Phase）  —— 支持 warm 阶段（纯任务损失预热）+ distill 阶段
"""

import os
import math
import time
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from core.distillation.loss_functions import (
    CompositiveDistillLoss,
    CosineTemperatureScheduler,
)

logger = logging.getLogger("EdgeDistillDet.Trainer")


# ─────────────────────────────────────────────────────────────────────────────
# 动态 Alpha 调度器
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveAlphaScheduler:
    """
    根据学生任务损失的 EMA 变化率动态调节蒸馏权重 alpha。

    机制：
      ema_loss 每 epoch 更新一次。
      delta = (ema_prev - ema_cur) / (ema_prev + eps)  —— 相对下降率
      delta 大（下降快）→ 学生自学能力强，降低 alpha
      delta 小（停滞）  → 需要更多教师引导，提升 alpha

    alpha_new = clip(alpha + lr_alpha * (delta_target - delta), alpha_min, alpha_max)
    """

    def __init__(
        self,
        alpha_init: float = 0.5,
        alpha_min: float = 0.2,
        alpha_max: float = 0.8,
        lr_alpha: float = 0.05,
        ema_decay: float = 0.9,
        delta_target: float = 0.01,
    ):
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lr_alpha = lr_alpha
        self.ema_decay = ema_decay
        self.delta_target = delta_target
        self._ema_loss: Optional[float] = None
        self._prev_ema: Optional[float] = None

    def update(self, task_loss: float) -> float:
        # EMA 更新
        if self._ema_loss is None:
            self._ema_loss = task_loss
        else:
            self._ema_loss = self.ema_decay * self._ema_loss + (1 - self.ema_decay) * task_loss

        if self._prev_ema is not None:
            delta = (self._prev_ema - self._ema_loss) / (self._prev_ema + 1e-8)
            self.alpha -= self.lr_alpha * (delta - self.delta_target)
            self.alpha = float(max(self.alpha_min, min(self.alpha_max, self.alpha)))

        self._prev_ema = self._ema_loss
        return self.alpha

    @property
    def current_alpha(self) -> float:
        return self.alpha


# ─────────────────────────────────────────────────────────────────────────────
# 主训练器
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveKDTrainer(DetectionTrainer):
    """
    面向边缘计算小目标检测的自适应异构知识蒸馏训练器。

    参数（通过 overrides 字典传入，或调用 configure() 方法设置）：
      teacher_path    : 教师模型权重路径
      kd_alpha_init   : 初始蒸馏权重（默认 0.5）
      T_max / T_min   : 温度范围（默认 6.0 / 1.5）
      warm_epochs     : 纯任务损失预热轮数（默认 5）
      w_kd / w_focal  : 复合损失各子项权重
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_model = None

        # 默认值（会在 setup_model 中从 model._kd_params 更新）
        self._teacher_path   = ""
        self._warm_epochs    = 5
        self._w_kd           = 0.5
        self._w_focal        = 0.3
        self._w_feat         = 0.0
        self._T_max          = 6.0
        self._T_min          = 1.5
        self._alpha_init     = 0.5
        self._scale_boost    = 2.0
        self._focal_gamma    = 2.0

        self._alpha_scheduler: Optional[AdaptiveAlphaScheduler] = None
        self._temp_scheduler:  Optional[CosineTemperatureScheduler] = None
        self._distill_loss:    Optional[CompositiveDistillLoss] = None

        # 蒸馏过程日志（每 epoch 记录一条）
        self._distill_log: list = []

        self._current_epoch_task_loss: float = 0.0
        self._batch_count: int = 0

    # ── 模型初始化 ──────────────────────────────────────────────────────────
    def setup_model(self):
        super().setup_model()

        # 从模型对象上读取蒸馏参数（新版 ultralytics 兼容）
        kd_source = getattr(self.model, '_kd_params', None) if hasattr(self, 'model') else None
        if kd_source and isinstance(kd_source, dict):
            self._teacher_path   = kd_source.get("teacher_path", self._teacher_path)
            self._warm_epochs    = int(kd_source.get("warm_epochs", self._warm_epochs))
            self._w_kd           = float(kd_source.get("w_kd", self._w_kd))
            self._w_focal        = float(kd_source.get("w_focal", self._w_focal))
            self._w_feat         = float(kd_source.get("w_feat", self._w_feat))
            self._T_max          = float(kd_source.get("T_max", self._T_max))
            self._T_min          = float(kd_source.get("T_min", self._T_min))
            self._alpha_init     = float(kd_source.get("kd_alpha_init", self._alpha_init))
            self._scale_boost    = float(kd_source.get("scale_boost", self._scale_boost))
            self._focal_gamma    = float(kd_source.get("focal_gamma", self._focal_gamma))

        self._load_teacher()
        self._init_distill_components()

    def _load_teacher(self):
        if not self._teacher_path or not os.path.exists(self._teacher_path):
            raise FileNotFoundError(
                f"[AdaptiveKDTrainer] 教师权重不存在: {self._teacher_path}\n"
                "请在 distill_config.yaml 中配置正确的 teacher_path。"
            )
        logger.info(f"正在加载教师模型: {self._teacher_path}")
        teacher = YOLO(self._teacher_path)
        self.teacher_model = teacher.model.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        logger.info("教师模型加载完毕，参数已冻结。")

    def _init_distill_components(self):
        total_epochs = self.args.epochs if hasattr(self.args, "epochs") else 150
        self._alpha_scheduler = AdaptiveAlphaScheduler(
            alpha_init=self._alpha_init,
            alpha_min=0.2,
            alpha_max=0.8,
        )
        self._temp_scheduler = CosineTemperatureScheduler(
            T_max=self._T_max,
            T_min=self._T_min,
            total_epochs=total_epochs,
        )
        self._distill_loss = CompositiveDistillLoss(
            w_kd=self._w_kd,
            w_focal=self._w_focal,
            w_feat=self._w_feat,
            temperature=self._T_max,
            scale_boost=self._scale_boost,
            focal_gamma=self._focal_gamma,
        )
        logger.info(
            f"蒸馏组件初始化完成 | "
            f"T: [{self._T_max}→{self._T_min}] | "
            f"alpha_init={self._alpha_init} | "
            f"warm_epochs={self._warm_epochs}"
        )

    # ── 损失计算核心 ─────────────────────────────────────────────────────────
    def loss(self, batch):
        preds_student = self.model(batch["img"])
        loss_task, loss_items = self.criterion(preds_student, batch)

        epoch = getattr(self, "epoch", 0)

        # 预热阶段：仅任务损失
        if epoch < self._warm_epochs:
            self._accumulate_task_loss(loss_task)
            return loss_task, loss_items

        # 蒸馏阶段
        current_T     = self._temp_scheduler.current_temperature
        current_alpha = self._alpha_scheduler.current_alpha
        num_cls       = self.model.nc

        with torch.no_grad():
            preds_teacher = self.teacher_model(batch["img"])

        loss_distill, distill_detail = self._distill_loss(
            student_preds=preds_student,
            teacher_preds=preds_teacher,
            num_classes=num_cls,
            temperature=current_T,
        )

        total_loss = (1.0 - current_alpha) * loss_task + current_alpha * loss_distill

        self._accumulate_task_loss(loss_task)

        loss_items_out = loss_items.clone()
        loss_items_out[0] = total_loss.detach()
        return total_loss, loss_items_out

    def _accumulate_task_loss(self, loss_task):
        val = loss_task.item() if isinstance(loss_task, torch.Tensor) else float(loss_task)
        self._current_epoch_task_loss += val
        self._batch_count += 1

    # ── Epoch 结束钩子：更新调度器 + 打印日志 ────────────────────────────────
    def on_train_epoch_end(self):
        super().on_train_epoch_end() if hasattr(super(), "on_train_epoch_end") else None
        epoch = getattr(self, "epoch", 0)

        avg_task = (
            self._current_epoch_task_loss / max(self._batch_count, 1)
        )
        self._current_epoch_task_loss = 0.0
        self._batch_count = 0

        if epoch >= self._warm_epochs:
            new_alpha = self._alpha_scheduler.update(avg_task)
            new_T     = self._temp_scheduler.step(epoch - self._warm_epochs)
        else:
            new_alpha = self._alpha_init
            new_T     = self._T_max

        log_entry = {
            "epoch":       epoch,
            "phase":       "distill" if epoch >= self._warm_epochs else "warm",
            "avg_task_loss": round(avg_task, 5),
            "alpha":       round(new_alpha, 4),
            "temperature": round(new_T, 3),
        }
        self._distill_log.append(log_entry)
        logger.info(
            f"[Epoch {epoch:4d}] phase={log_entry['phase']} | "
            f"task_loss={avg_task:.4f} | alpha={new_alpha:.3f} | T={new_T:.2f}"
        )

    def get_distill_log(self) -> list:
        """返回完整的蒸馏过程记录（可供后处理或可视化）。"""
        return self._distill_log
