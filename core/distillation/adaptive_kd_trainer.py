"""
core/distillation/adaptive_kd_trainer.py
==========================================
自适应知识蒸馏训练器（AdaptiveKDTrainer）— v4 内存优化版

核心设计原则（v4）:
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
from core.distillation.loss_functions import CompositiveDistillLoss, CosineTemperatureScheduler
from core.logging import get_logger
from web.services.cache.csv_cache import invalidate_csv_rows
logger = logging.getLogger('DistillTrain')

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
        return (out_p, out_r)
    p_arr = _np.asarray(p_arr).ravel()
    r_arr = _np.asarray(r_arr).ravel()
    for j, cid in enumerate(ap_idx):
        cid = int(cid)
        if not 0 <= cid < nc:
            continue
        if j < len(p_arr):
            pv = p_arr[j]
            if pv is not None and (not (isinstance(pv, (float, _np.floating)) and _np.isnan(pv))):
                out_p[cid] = float(pv)
        if j < len(r_arr):
            rv = r_arr[j]
            if rv is not None and (not (isinstance(rv, (float, _np.floating)) and _np.isnan(rv))):
                out_r[cid] = float(rv)
    return (out_p, out_r)
_KD_TRAINERS: dict = {}

class AdaptiveAlphaScheduler:
    """动态调节蒸馏权重 alpha"""

    def __init__(self, alpha_init=0.5, alpha_min=0.2, alpha_max=0.8, lr_alpha=0.05, ema_decay=0.9, delta_target=0.01):
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lr_alpha = lr_alpha
        self.ema_decay = ema_decay
        self.delta_target = delta_target
        self._ema_loss = None
        self._prev_ema = None

    @property
    def current_alpha(self):
        return self.alpha

    def update(self, task_loss):
        v = safe_scalar(task_loss)
        self._ema_loss = v if self._ema_loss is None else self.ema_decay * self._ema_loss + (1 - self.ema_decay) * v
        if self._prev_ema is not None:
            d = (self._prev_ema - self._ema_loss) / (self._prev_ema + 1e-08)
            self.alpha = max(self.alpha_min, min(self.alpha_max, self.alpha + self.lr_alpha * (self.delta_target - d)))
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
    current_loss = getattr(model, 'loss', None)
    if current_loss is None:
        return None
    original_fn = getattr(type(model), '_distill_original_loss', None)
    if original_fn is not None:
        return original_fn.__get__(model, type(model)) if hasattr(original_fn, '__get__') else original_fn
    return current_loss


def _move_detection_loss_to_device(loss_obj, device):
    """把 Ultralytics 检测损失内部缓冲迁移到目标设备。"""
    if loss_obj is None:
        return None
    try:
        loss_obj.device = device
    except Exception:
        pass
    proj = getattr(loss_obj, 'proj', None)
    if hasattr(proj, 'to'):
        try:
            loss_obj.proj = proj.to(device)
        except Exception:
            pass
    bbox_loss = getattr(loss_obj, 'bbox_loss', None)
    if hasattr(bbox_loss, 'to'):
        try:
            loss_obj.bbox_loss = bbox_loss.to(device)
        except Exception:
            pass
    return loss_obj


def _resolve_trainer_device(trainer, model):
    device = getattr(trainer, 'device', None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device('cpu')


def _loss_to_float(value):
    if value is None:
        return 0.0
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'numel') and value.numel() > 1:
        value = value.sum()
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


class _DistillCriterionProxy:
    """将原始检测损失包装为蒸馏损失，同时保持原始接口行为。"""

    def __init__(self, trainer, base_criterion):
        self._trainer_ref = weakref.ref(trainer)
        self._base_criterion = base_criterion

    def __call__(self, preds, batch):
        trainer_ref = getattr(self, '_trainer_ref', None)
        trainer = trainer_ref() if callable(trainer_ref) else None
        if trainer is None:
            return self._base_criterion(preds, batch)
        return trainer._criterion_impl(preds, batch, self._base_criterion)

    def update(self, *args, **kwargs):
        updater = getattr(self._base_criterion, 'update', None)
        if callable(updater):
            return updater(*args, **kwargs)
        return None

    def __getattr__(self, name):
        try:
            base_criterion = object.__getattribute__(self, '_base_criterion')
        except AttributeError:
            raise AttributeError(name) from None
        if base_criterion is None:
            raise AttributeError(name)
        return getattr(base_criterion, name)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._base_criterion!r})'

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_trainer_ref'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class AdaptiveKDTrainer(DetectionTrainer):
    """
    自适应知识蒸馏训练器 — v3 零污染版
    
    关键改进（v3）：
      - 不修改 model 对象任何属性 → pickle/EMA/deepcopy 全部兼容
      - 通过重写 get_loss() 注入蒸馏逻辑（ultralytics 官方扩展方式）
      - 原始 loss 函数保存在 trainer._orig_loss 上
            - 通过 model.criterion 代理保持训练主路径稳定
    """
    _kd_class_params = None

    @classmethod
    def set_kd_params(cls, teacher_path='', alpha_init=0.5, T_max=6.0, T_min=1.5, warm_epochs=5, w_kd=0.5, w_focal=0.3, w_feat=0.0, scale_boost=2.0, focal_gamma=2.0, **extra_kwargs):
        cls._kd_class_params = {'teacher_path': teacher_path, 'alpha_init': alpha_init, 'T_max': T_max, 'T_min': T_min, 'warm_epochs': warm_epochs, 'w_kd': w_kd, 'w_focal': w_focal, 'w_feat': w_feat, 'scale_boost': scale_boost, 'focal_gamma': focal_gamma}
        if extra_kwargs:
            cls._kd_class_params.update(extra_kwargs)

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides and 'trainer' in overrides:
            del overrides['trainer']
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_model = None
        self._distill_loss = None
        self._temp_scheduler = None
        self._alpha_scheduler = None
        self._teacher_path = ''
        self._warm_epochs = 5
        self._w_kd = 0.5
        self._w_focal = 0.3
        self._w_feat = 0.0
        self._T_max = 6.0
        self._T_min = 1.5
        self._alpha_init = 0.5
        self._scale_boost = 2.0
        self._focal_gamma = 2.0
        self._distill_entered = False
        self._orig_loss = None
        self._base_criterion = None
        self._criterion_proxy = None
        self._accum_task_loss = []
        self._accum_kd_loss = []
        self._epoch_task_loss = 0.0
        self._epoch_kd_loss = 0.0
        self._batch_count = 0
        self._distill_log = []
        self._last_log_batch = 0
        self._setup_distill_from_class_params()
        logger.info('[INIT] 蒸馏训练器初始化完成，等待模型就绪后绑定 model.criterion')

    def _setup_distill_from_class_params(self):
        """从类参数加载教师模型和蒸馏组件"""
        kd = AdaptiveKDTrainer._kd_class_params
        if not kd:
            return
        teacher_path = str(kd.get('teacher_path', '')).strip()
        if not teacher_path or not os.path.exists(teacher_path):
            return
        self._teacher_path = teacher_path
        self._warm_epochs = int(kd.get('warm_epochs', 5))
        self._w_kd = float(kd.get('w_kd', 0.5))
        self._w_focal = float(kd.get('w_focal', 0.3))
        self._w_feat = w_feat_to_scalar(kd.get('w_feat', 0.0))
        self._T_max = float(kd.get('T_max', 6.0))
        self._T_min = float(kd.get('T_min', 1.5))
        self._alpha_init = float(kd.get('alpha_init', 0.5))
        self._scale_boost = float(kd.get('scale_boost', 2.0))
        self._focal_gamma = float(kd.get('focal_gamma', 2.0))
        logger.info(f'[INIT] 蒸馏参数 | teacher={os.path.basename(self._teacher_path)} | α={self._alpha_init} T=[{self._T_max},{self._T_min}]')
        self._load_teacher()
        self._init_distill()
        logger.info('[INIT] 蒸馏损失函数 + 调度器 就绪 ✓')

    def setup_model(self):
        ckpt = super().setup_model()
        logger.info('[INIT] 检测模型就绪')
        self._orig_loss = _resolve_original_model_loss(self.model)
        if self._orig_loss is None:
            logger.error('[FATAL] 找不到 model.loss!')
        self._install_distill_criterion()
        if self.teacher_model is not None:
            logger.info('[INIT] 蒸馏组件已在 __init__ 中加载，跳过重复初始化')
        elif self._teacher_path and os.path.exists(self._teacher_path):
            logger.info(f'[INIT] 补加载教师模型: {os.path.basename(self._teacher_path)}')
            self._load_teacher()
            self._init_distill()
            self._install_distill_criterion()
        return ckpt

    def _install_distill_criterion(self):
        """把模型自身的 criterion 包装成蒸馏代理，确保训练循环走真实调用路径。"""
        model = getattr(self, 'model', None)
        if model is None or getattr(self, 'teacher_model', None) is None:
            return
        if getattr(model, 'args', None) is None and getattr(self, 'args', None) is not None:
            model.args = self.args
        current = getattr(model, 'criterion', None)
        if isinstance(current, _DistillCriterionProxy):
            self._base_criterion = current._base_criterion
            self._criterion_proxy = current
            self.criterion = current
            return
        if current is None:
            current = model.init_criterion()
        current = _move_detection_loss_to_device(current, _resolve_trainer_device(self, model))
        self._base_criterion = current
        proxy = _DistillCriterionProxy(self, current)
        model.criterion = proxy
        self._criterion_proxy = proxy
        self.criterion = proxy
        logger.info('[INIT] model.criterion 已绑定到蒸馏代理')

    def _load_teacher(self):
        if not self._teacher_path:
            logger.info('[TEACHER] 教师路径为空，跳过加载')
            return
        if not os.path.exists(self._teacher_path):
            raise FileNotFoundError(f'教师模型不存在: {self._teacher_path}')
        logger.info(f'加载教师模型: {self._teacher_path}')
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            teacher = YOLO(self._teacher_path, verbose=False)
        teacher_model = teacher.model.to(self.device).eval()
        del teacher
        for p in teacher_model.parameters():
            p.requires_grad = False
        try:
            teacher_model.gradient_checkpointing_enable()
            logger.info('教师模型已启用梯度检查点')
        except Exception:
            pass
        self.teacher_model = teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info('教师模型就绪 ✓ (显存已优化)')

    def _init_distill(self):
        total_epochs = self.args.epochs if hasattr(self.args, 'epochs') else 150
        self._alpha_scheduler = AdaptiveAlphaScheduler(alpha_init=self._alpha_init)
        self._temp_scheduler = CosineTemperatureScheduler(T_max=self._T_max, T_min=self._T_min, total_epochs=total_epochs)
        self._distill_loss = CompositiveDistillLoss(w_kd=self._w_kd, w_focal=self._w_focal, w_feat=self._w_feat, temperature=self._T_max, scale_boost=self._scale_boost, focal_gamma=self._focal_gamma).to(self.device)
        logger.info(f'蒸馏组件就绪 | T:[{self._T_max}→{self._T_min}] α:{self._alpha_init} warm:{self._warm_epochs}')
        self.add_callback('on_train_epoch_start', self._on_epoch_start)
        self.add_callback('on_train_epoch_end', self._on_epoch_end)
        self.add_callback('on_train_batch_end', self._on_train_batch_end)
        self.add_callback('on_fit_epoch_end', self._on_fit_epoch_end)
        self.add_callback('on_val_end', self._on_val_end)
        self.add_callback('on_train_end', self._on_train_end)
        logger.info('AdaptiveKDTrainer callbacks registered: on_train_epoch_start, on_train_epoch_end, on_train_batch_end, on_fit_epoch_end, on_val_end, on_train_end')

    def _distill_state_path(self) -> Path | None:
        sd = getattr(self, 'save_dir', None)
        if sd:
            return Path(sd) / 'distill_state.json'
        return None

    def _save_distill_state(self, epoch: int) -> None:
        state_path = self._distill_state_path()
        if state_path is None or self._alpha_scheduler is None:
            return
        last_kd = None
        current_temp = None
        if self._distill_log:
            try:
                last_kd = float(self._distill_log[-1].get('kd_loss'))
            except Exception:
                last_kd = None
        if self._temp_scheduler is not None:
            try:
                current_temp = float(self._temp_scheduler.current_temperature)
            except Exception:
                current_temp = None
        payload = {'epoch': int(epoch), 'alpha_scheduler': {'alpha': float(self._alpha_scheduler.alpha), 'alpha_min': float(self._alpha_scheduler.alpha_min), 'alpha_max': float(self._alpha_scheduler.alpha_max), 'lr_alpha': float(self._alpha_scheduler.lr_alpha), 'ema_decay': float(self._alpha_scheduler.ema_decay), 'delta_target': float(self._alpha_scheduler.delta_target), 'ema_loss': None if self._alpha_scheduler._ema_loss is None else float(self._alpha_scheduler._ema_loss), 'prev_ema': None if self._alpha_scheduler._prev_ema is None else float(self._alpha_scheduler._prev_ema)}, 'last_kd_loss': last_kd, 'temperature': current_temp}
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f'[DISTILL_STATE] 保存失败: {e}')

    def _restore_alpha_from_results_csv(self) -> float | None:
        sd = getattr(self, 'save_dir', None)
        if not sd:
            return None
        csv_path = Path(sd) / 'results.csv'
        if not csv_path.exists():
            return None
        last_alpha = None
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = _c.DictReader(f)
                for row in reader:
                    raw = row.get('distill/alpha')
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
        sd = getattr(self, 'save_dir', None)
        if not sd:
            return None
        csv_path = Path(sd) / 'results.csv'
        if not csv_path.exists():
            return None
        last = None
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as f:
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
        return {'epoch': _safe_int(last.get('epoch')), 'alpha': _safe_float(last.get('distill/alpha')), 'kd_loss': _safe_float(last.get('distill/kd_loss')), 'temperature': _safe_float(last.get('distill/temperature'))}

    def _restore_distill_state_if_needed(self) -> None:
        if not bool(getattr(self.args, 'resume', False)):
            return
        if self._alpha_scheduler is None:
            return
        restored_from = None
        restored_alpha = None
        restored_temp_from = None
        state_path = self._distill_state_path()
        if state_path is not None and state_path.exists():
            try:
                disk = json.loads(state_path.read_text(encoding='utf-8'))
                alpha_state = (disk or {}).get('alpha_scheduler', {}) or {}
                alpha = alpha_state.get('alpha')
                if alpha is not None:
                    self._alpha_scheduler.alpha = float(alpha)
                    self._alpha_scheduler._ema_loss = alpha_state.get('ema_loss')
                    self._alpha_scheduler._prev_ema = alpha_state.get('prev_ema')
                    restored_alpha = float(self._alpha_scheduler.alpha)
                    restored_from = 'distill_state.json'
                saved_temp = (disk or {}).get('temperature')
                if self._temp_scheduler is not None and saved_temp is not None:
                    try:
                        self._temp_scheduler._current_T = float(saved_temp)
                        restored_temp_from = 'distill_state.json'
                    except Exception:
                        pass
                try:
                    self._resume_prev_kd_loss = float((disk or {}).get('last_kd_loss'))
                except Exception:
                    self._resume_prev_kd_loss = None
                try:
                    self._resume_prev_epoch = int((disk or {}).get('epoch'))
                except Exception:
                    self._resume_prev_epoch = None
            except Exception as e:
                logger.debug(f'[DISTILL_STATE] 读取状态失败: {e}')
        if restored_alpha is None:
            csv_baseline = self._restore_distill_baseline_from_results_csv()
            csv_alpha = csv_baseline.get('alpha') if isinstance(csv_baseline, dict) else None
            if csv_alpha is not None:
                self._alpha_scheduler.alpha = float(csv_alpha)
                restored_alpha = float(self._alpha_scheduler.alpha)
                restored_from = 'results.csv'
            if isinstance(csv_baseline, dict):
                if self._resume_prev_kd_loss is None and csv_baseline.get('kd_loss') is not None:
                    self._resume_prev_kd_loss = float(csv_baseline['kd_loss'])
                if self._resume_prev_epoch is None and csv_baseline.get('epoch') is not None:
                    self._resume_prev_epoch = int(csv_baseline['epoch'])
                if self._temp_scheduler is not None and csv_baseline.get('temperature') is not None:
                    try:
                        self._temp_scheduler._current_T = float(csv_baseline['temperature'])
                        restored_temp_from = 'results.csv'
                    except Exception:
                        pass
        else:
            if self._resume_prev_kd_loss is None or self._resume_prev_epoch is None:
                csv_baseline = self._restore_distill_baseline_from_results_csv()
                if isinstance(csv_baseline, dict):
                    if self._resume_prev_kd_loss is None and csv_baseline.get('kd_loss') is not None:
                        self._resume_prev_kd_loss = float(csv_baseline['kd_loss'])
                    if self._resume_prev_epoch is None and csv_baseline.get('epoch') is not None:
                        self._resume_prev_epoch = int(csv_baseline['epoch'])
            if self._temp_scheduler is not None and restored_temp_from is None:
                csv_baseline = self._restore_distill_baseline_from_results_csv()
                if isinstance(csv_baseline, dict) and csv_baseline.get('temperature') is not None:
                    try:
                        self._temp_scheduler._current_T = float(csv_baseline['temperature'])
                        restored_temp_from = 'results.csv'
                    except Exception:
                        pass
        if restored_alpha is not None:
            logger.info(f'[RESUME_ALPHA] 恢复 alpha={restored_alpha:.6f} 来源={restored_from}')

    def get_loss(self, batch, preds=None):
        """
        兼容层：将 get_loss 调用转发到 model.loss().
        ultralytics 训练循环最终走 model.loss(batch, preds)，
        但保留 get_loss() 以兼容其他扩展方式。
        """
        if preds is None:
            preds = self.model(batch['img'])
        return self.model.loss(batch, preds)

    def _criterion_impl(self, preds, batch, base_criterion):
        """执行标准检测损失 + 蒸馏损失，并维护 epoch 统计。"""
        if base_criterion is None:
            model = getattr(self, 'model', None)
            if model is None:
                raise RuntimeError('model 还未初始化，无法计算损失')
            current = getattr(model, 'criterion', None)
            if isinstance(current, _DistillCriterionProxy):
                base_criterion = current._base_criterion
            else:
                base_criterion = current or model.init_criterion()
            self._base_criterion = base_criterion
        loss_task, loss_items = base_criterion(preds, batch)
        if getattr(self, 'teacher_model', None) is None:
            return (loss_task, loss_items)
        epoch = getattr(self, 'epoch', 0)
        warm_phase = epoch < self._warm_epochs
        try:
            self._epoch_task_loss += _loss_to_float(loss_task)
            self._batch_count += 1
        except Exception as e:
            logger.warning(f'[CRITERION] 累加损失失败: {e}')
        if warm_phase:
            return (loss_task, loss_items)
        if not self._distill_entered:
            self._distill_entered = True
            logger.info(f'[DISTILL] 进入蒸馏阶段 epoch={epoch}')
        img = batch['img']
        device_type = img.device.type
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=False):
                teacher_preds = self.teacher_model(img.float())
        num_classes = getattr(self.model, 'nc', 80)
        current_temp = self._temp_scheduler.current_temperature
        loss_distill, _ = self._distill_loss(student_preds=preds, teacher_preds=teacher_preds, num_classes=num_classes, temperature=current_temp)
        try:
            self._epoch_kd_loss += _loss_to_float(loss_distill)
        except Exception as e:
            logger.warning(f'[CRITERION] 累加蒸馏损失失败: {e}')
        del teacher_preds
        if device_type == 'cuda':
            torch.cuda.empty_cache()
        alpha = self._alpha_scheduler.current_alpha
        total_loss = (1 - alpha) * loss_task + alpha * loss_distill
        return (total_loss, loss_items)

    def criterion(self, preds, batch):
        """
        兼容旧调用点：直接复用蒸馏代理的损失逻辑。
        """
        return self._criterion_impl(preds, batch, self._base_criterion)

    def _setup_train(self):
        """重写 setup_train：全程锁定 workers=0"""
        self.args.workers = 0
        super()._setup_train()
        self.args.workers = 0
        logger.info('[INIT 6/6] 数据集加载完成 ✓ | ...')

    def _accumulate_losses(self, task_loss=None, kd_loss=None):
        """累加损失并递增 batch 计数"""
        if task_loss is not None:
            self._epoch_task_loss += task_loss.item() if hasattr(task_loss, 'item') else float(task_loss)
        if kd_loss is not None:
            self._epoch_kd_loss += kd_loss.item() if hasattr(kd_loss, 'item') else float(kd_loss)
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
        return (total_epochs, batch_size, total_batches, dataset_samples)

    def _on_epoch_start(self, trainer):
        epoch = trainer.epoch
        display_epoch = epoch + 1
        total_epochs, batch_size, total_batches, dataset_samples = self._get_epoch_info()
        logger.info(f"[EPOCH_START] epoch={display_epoch} total={total_epochs} batch_size={batch_size} total_batches={total_batches or 'unknown'} dataset_samples={(dataset_samples if dataset_samples is not None else 'unknown')}")

    def _on_epoch_end(self, trainer):
        epoch = trainer.epoch
        display_epoch = epoch + 1
        batch_count = self._batch_count
        total_epochs, batch_size, total_batches, dataset_samples = self._get_epoch_info()
        if epoch >= self._warm_epochs:
            self._temp_scheduler.step(epoch)
        if batch_count > 0:
            try:
                avg_t = self._epoch_task_loss / batch_count
                avg_k = self._epoch_kd_loss / batch_count
            except Exception as e:
                logger.warning(f'[EPOCH_END] 计算平均损失失败: {e}')
                return
            phase = 'warm' if epoch < self._warm_epochs else 'distill'
            alpha_val = round(self._alpha_scheduler.current_alpha, 4)
            temp_val = round(self._temp_scheduler.current_temperature, 4)
            logger.info(f"[EPOCH_PROGRESS] epoch={display_epoch} total={total_epochs} loss={avg_t:.4f} kd={avg_k:.4f} alpha={alpha_val} temp={temp_val} batches={batch_count}/{total_batches or 'unknown'} batch_size={batch_size} samples={(batch_count * batch_size if batch_size else 'unknown')} dataset_samples={(dataset_samples if dataset_samples is not None else 'unknown')} phase={phase}")
            self._distill_log.append({'epoch': display_epoch, 'phase': phase, 'task_loss': round(avg_t, 6), 'kd_loss': round(avg_k, 6), 'alpha': round(self._alpha_scheduler.current_alpha, 6), 'temperature': round(self._temp_scheduler.current_temperature, 6)})
            if epoch >= self._warm_epochs:
                self._alpha_scheduler.update(avg_t)
            self._save_distill_state(display_epoch)
            self._epoch_task_loss = 0.0
            self._epoch_kd_loss = 0.0
            self._batch_count = 0
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    def _on_train_batch_end(self, trainer):
        pass

    def _save_distill_log_json(self):
        """将蒸馏日志持久化到 distill_log.json（作为备用数据源）"""
        sd = getattr(self, 'save_dir', None)
        if not sd or not self._distill_log:
            return
        try:
            log_path = Path(sd) / 'distill_log.json'
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self._distill_log, f, indent=2, ensure_ascii=False)
            logger.debug(f'[DISTILL_LOG] 已保存 {len(self._distill_log)} 条蒸馏日志到 {log_path}')
        except Exception as e:
            logger.warning(f'[DISTILL_LOG] 保存 distill_log.json 失败: {e}')

    def _on_fit_epoch_end(self, trainer):
        """注入蒸馏指标到 results.csv（同时保存备用 distill_log.json）"""
        epoch = trainer.epoch
        display_epoch = epoch + 1
        entry = next((e for e in reversed(self._distill_log) if e['epoch'] == display_epoch), None)
        if entry is None and self._distill_log:
            entry = self._distill_log[-1]
        if not entry:
            return
        av, tv, kv = (entry['alpha'], entry['temperature'], entry['kd_loss'])
        for target in [getattr(trainer, 'metrics', None), getattr(trainer, 'results_dict', None)]:
            if isinstance(target, dict):
                try:
                    target['distill/alpha'] = av
                    target['distill/temperature'] = tv
                    target['distill/kd_loss'] = kv
                except Exception as e:
                    logger.debug(f'[DISTILL] 注入 metrics 失败: {e}')
        self._save_distill_log_json()

    def _append_csv(self, epoch, a, t, k):
        """安全地追加蒸馏指标到 results.csv"""
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
                r = _c.DictReader(f)
                fn = list(r.fieldnames or [])
                rows = list(r)
            for c in ['distill/alpha', 'distill/temperature', 'distill/kd_loss']:
                if c not in fn:
                    fn.append(c)
            matched = False
            for row in rows:
                try:
                    row_epoch = int(float(row.get('epoch', '-1')))
                    if row_epoch == int(epoch):
                        row['distill/alpha'] = a
                        row['distill/temperature'] = t
                        row['distill/kd_loss'] = k
                        matched = True
                        break
                except (ValueError, TypeError, KeyError):
                    continue
            if matched:
                with open(p, 'w', encoding='utf-8', newline='') as f:
                    writer = _c.DictWriter(f, fn)
                    writer.writeheader()
                    writer.writerows(rows)
                logger.debug(f'[DISTILL_CSV] 已更新 epoch {int(epoch)} 的蒸馏数据')
                # 清除SQLite缓存，下次查询会从磁盘重新读取
                try:
                    invalidate_csv_rows(p)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f'[DISTILL_CSV] 追加蒸馏数据失败: {e}')

    def _on_val_end(self, trainer):
        """验证结束后：提取并缓存每类别的性能指标（AP / Precision / Recall）"""
        try:
            validator = getattr(trainer, 'validator', None)
            if validator is None:
                logger.debug('[VAL] trainer.validator 为空，跳过')
                return
            results = None
            for attr_name in ('results', 'final_results', 'stats'):
                r = getattr(validator, attr_name, None) or getattr(trainer, f'validator_{attr_name}', None)
                if r is not None:
                    results = r
                    break
            if results is None:
                metrics_dict = getattr(trainer, 'metrics', None) or getattr(trainer, 'results_dict', None)
                if isinstance(metrics_dict, dict):
                    self._try_save_per_class_from_metrics(metrics_dict, trainer)
                return
            box = getattr(results, 'box', None)
            if box is None:
                return
            class_names = getattr(results, 'names', {}) or {}
            nc = len(class_names) if class_names else getattr(box, 'nc', 0)
            if nc == 0:
                return
            import numpy as _np
            _maps = None
            if hasattr(box, 'maps') and box.maps is not None:
                _m = _np.asarray(box.maps)
                _maps = list(_m.flatten()) if _m.ndim > 1 else list(_m)
            labels, map_list = ([], [])
            for i in range(nc):
                labels.append(class_names.get(i, f'class{i}'))
                map_list.append(float(_maps[i]) if _maps and i < len(_maps) else None)
            prec_list, rec_list = _scatter_pr_from_ultralytics_box(box, nc)
            output_data = {'labels': labels, 'map': map_list, 'precision': prec_list, 'recall': rec_list, 'epoch': trainer.epoch + 1, 'generated_at': datetime.now().isoformat()}
            sd = getattr(trainer, 'save_dir', None)
            if sd:
                cache_path = Path(sd) / 'per_class_metrics.json'
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f'[VAL] 每类指标提取跳过: {e}')

    def _try_save_per_class_from_metrics(self, metrics: dict, trainer):
        """从 metrics 字典中尝试提取每类指标（兜底路径）"""
        try:
            class_metrics: dict[int, dict[str, float]] = {}
            for key in metrics.keys():
                match = re.search('(?i)(?:class|cls)[\\s_/\\\\-]*(\\d+)', str(key))
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
            data_cfg_path = getattr(getattr(trainer, 'args', None), 'data', None)
            names = {}
            if data_cfg_path:
                from pathlib import Path as _P
                import yaml as _yml
                dp = Path(data_cfg_path)
                if dp.exists():
                    cfg_data = _yml.safe_load(dp.read_text(encoding='utf-8'))
                    names = cfg_data.get('names') or {}
            labels, map_l, prec_l, rec_l = ([], [], [], [])
            for idx in sorted(class_metrics.keys()):
                cm = class_metrics[idx]
                labels.append(names.get(idx, f'class{idx}') if isinstance(names, dict) else f'class{idx}')
                map_val = cm.get('map')
                prec_val = cm.get('p')
                rec_val = cm.get('r')
                map_l.append(map_val)
                prec_l.append(prec_val)
                rec_l.append(rec_val)
            output = {'labels': labels, 'map': map_l, 'precision': prec_l, 'recall': rec_l, 'epoch': getattr(trainer, 'epoch', -1) + 1, 'generated_at': datetime.now().isoformat(), 'source': 'metrics_dict_fallback'}
            sd = getattr(trainer, 'save_dir', None)
            if sd:
                cache_path = Path(sd) / 'per_class_metrics.json'
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f'[VAL] metrics_dict 兜底提取失败: {e}')

    def _on_train_end(self, trainer):
        """
        【安全网回调】训练完全结束后重新写入所有蒸馏数据。
        
        处理流程：
        1. 持久化蒸馏日志到 distill_log.json
        2. 修复 results.csv 中的蒸馏列（恢复被 final_eval 覆写的数据）
        3. 处理异常中断的情况
        """
        if not self._distill_log:
            logger.info('[TRAIN_END] 无蒸馏日志，跳过')
            return
        sd = getattr(self, 'save_dir', None)
        if not sd:
            logger.warning('[TRAIN_END] save_dir 为空')
            return
        self._save_distill_log_json()
        p = Path(sd) / 'results.csv'
        if not p.exists():
            logger.warning(f'[TRAIN_END] results.csv 不存在: {p}')
            return
        try:
            rows = []
            fn = []
            with open(p, 'r', encoding='utf-8', newline='') as f:
                reader = _c.DictReader(f)
                fn = list(reader.fieldnames or [])
                rows = list(reader)
            logger.info(f'[TRAIN_END] 读取 results.csv: {len(rows)} 行, {len(fn)} 列')
            distill_cols = ['distill/alpha', 'distill/temperature', 'distill/kd_loss']
            for c in distill_cols:
                if c not in fn:
                    fn.append(c)
            epoch_to_distill = {entry['epoch']: entry for entry in self._distill_log}
            written_count = 0
            missing_epochs = set()
            for row in rows:
                try:
                    ep = int(float(row.get('epoch', '-1')))
                except (ValueError, TypeError):
                    continue
                de = epoch_to_distill.get(ep) or epoch_to_distill.get(ep - 1) or epoch_to_distill.get(ep + 1)
                if de is not None:
                    row['distill/alpha'] = de['alpha']
                    row['distill/temperature'] = de['temperature']
                    row['distill/kd_loss'] = de['kd_loss']
                    written_count += 1
                else:
                    missing_epochs.add(ep)
            with open(p, 'w', encoding='utf-8', newline='') as f:
                writer = _c.DictWriter(f, fn)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f'[TRAIN_END] CSV 修复完成: {written_count}/{len(rows)} 行已补全蒸馏数据 (缺失 {len(missing_epochs)} 轮: {sorted(missing_epochs)[:5]}...)')
            # 清除SQLite缓存，下次查询会从磁盘重新读取
            try:
                invalidate_csv_rows(p)
            except Exception:
                pass
        except Exception as e:
            logger.error(f'[TRAIN_END] CSV 修复失败: {type(e).__name__}: {e}')
            logger.info('[TRAIN_END] 备用数据已保存到 distill_log.json，可通过该文件恢复')
        try:
            self._ensure_per_class_metrics(trainer)
        except Exception as e:
            logger.warning(f'[TRAIN_END] 生成 per_class_metrics.json 失败: {e}')

    def _ensure_per_class_metrics(self, trainer):
        """训练结束时确保 per_class_metrics.json 已存在，否则用模型验证生成。"""
        sd = getattr(trainer, 'save_dir', None) or getattr(self, 'save_dir', None)
        if not sd:
            return
        cache_path = Path(sd) / 'per_class_metrics.json'
        if cache_path.exists():
            logger.info('[TRAIN_END] per_class_metrics.json 已存在，跳过生成')
            return
        weight_candidates = [Path(sd) / 'weights' / 'best.pt', Path(sd) / 'weights' / 'last.pt', Path(sd) / 'best.pt', Path(sd) / 'last.pt']
        model_path = None
        for candidate in weight_candidates:
            if candidate.exists():
                model_path = candidate
                break
        if model_path is None or not model_path.exists():
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
                    eval_results = eval_model.val(data=data_cfg, imgsz=imgsz, verbose=False)
                self._extract_and_save_per_class(eval_results, sd)
            except Exception as e:
                logger.warning(f'[TRAIN_END] 兜底每类指标生成失败: {e}')
            return
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
            logger.warning(f'[TRAIN_END] 权重文件验证生成每类指标失败: {e}')

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
        _maps = None
        if hasattr(box, 'maps') and box.maps is not None:
            _m = _np.asarray(box.maps)
            _maps = list(_m.flatten()) if _m.ndim > 1 else list(_m)
        labels, map_l = ([], [])
        for i in range(nc):
            labels.append(class_names.get(i, f'class{i}'))
            map_l.append(float(_maps[i]) if _maps and i < len(_maps) else None)
        prec_l, rec_l = _scatter_pr_from_ultralytics_box(box, nc)
        output = {'labels': labels, 'map': map_l, 'precision': prec_l, 'recall': rec_l, 'source': 'train_end_fallback', 'generated_at': datetime.now().isoformat()}
        cache_path = Path(save_dir) / 'per_class_metrics.json'
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f'[TRAIN_END] 每类指标已生成: {cache_path} ({nc} 个类别)')

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

    def get_distill_log(self):
        return self._distill_log