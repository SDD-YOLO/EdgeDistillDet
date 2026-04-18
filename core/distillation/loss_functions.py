"""
core/distillation/loss_functions.py
=====================================
自适应蒸馏损失函数模块 — 增强鲁棒性版

改进：
  1. 所有张量操作增加维度检查和设备对齐
  2. 空预测时安全返回零张量
  3. 特征提取兼容更多 YOLO 输出格式
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from core.distillation.common import safe_scalar


def _safe_scalar(value):
    """兼容旧调用点，内部委托到共享实现。"""
    return safe_scalar(value)


# ═══════════════════════════════════════════════════════════════════════════════


def _align_and_flatten(s_cls: torch.Tensor, t_cls: torch.Tensor,
                       num_classes: int) -> tuple:
    """
    对齐学生和教师的类别特征张量，然后展平为 [N, C]
    
    核心功能：
      - 处理不同模型（如 YOLOv8n vs YOLOv8m）的空间维度差异
      - 使用自适应池化将空间尺寸统一到较小值
      - 确保返回的两个张量行数相同
    
    Args:
        s_cls: 学生类别特征 [B, C_s, H_s, W_s] 或其他形状
        t_cls: 教师类别特征 [B, C_t, H_t, W_t] 或其他形状
        num_classes: 类别数量
    
    Returns:
        (s_flat, t_flat): 展平后的对齐张量，形状均为 [N_aligned, C]
    """
    # 如果已经是 2D，尝试直接对齐
    if s_cls.dim() <= 2 and t_cls.dim() <= 2:
        if s_cls.dim() == 2 and t_cls.dim() == 2:
            min_n = min(s_cls.shape[0], t_cls.shape[0])
            if min_n > 0:
                return s_cls[:min_n], t_cls[:min_n]
        return s_cls.reshape(-1, s_cls.shape[-1]) if s_cls.dim() >= 2 else s_cls.reshape(-1, 1), \
               t_cls.reshape(-1, t_cls.shape[-1]) if t_cls.dim() >= 2 else t_cls.reshape(-1, 1)
    
    # 4D 特征图对齐：使用自适应池化统一空间尺寸
    if s_cls.dim() == 4 and t_cls.dim() == 4:
        B = s_cls.shape[0]
        
        # 取较小的空间尺寸作为目标（避免上采样引入伪影）
        target_h = min(s_cls.shape[2], t_cls.shape[2])
        target_w = min(s_cls.shape[3], t_cls.shape[3])
        
        # 确保至少有 1x1 的空间分辨率
        target_h = max(target_h, 1)
        target_w = max(target_w, 1)
        
        s_pooled = F.adaptive_avg_pool2d(s_cls, (target_h, target_w))
        t_pooled = F.adaptive_avg_pool2d(t_cls, (target_h, target_w))
        
        # 展平为 [B * target_h * target_w, C]
        s_flat = s_pooled.permute(0, 2, 3, 1).contiguous().reshape(-1, s_pooled.shape[1])
        t_flat = t_pooled.permute(0, 2, 3, 1).contiguous().reshape(-1, t_pooled.shape[1])
        
        return s_flat, t_flat
    
    # 混合维度情况：分别处理
    def _flatten_3d_or_4d(x):
        if x.dim() == 4:
            return x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
        elif x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            return x
        else:
            return x.reshape(-1, 1)
    
    s_flat = _flatten_3d_or_4d(s_cls)
    t_flat = _flatten_3d_or_4d(t_cls)
    
    # 行数不一致时截断到较小值
    min_rows = min(s_flat.shape[0], t_flat.shape[0])
    if min_rows > 0:
        return s_flat[:min_rows], t_flat[:min_rows]
    
    return s_flat, t_flat


def _safe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """安全地将张量移到指定设备"""
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _normalize_preds(preds, num_classes: int) -> list:
    """
    将模型预测统一为 List[Tensor] 格式。
    
    支持的输入格式：
      - list/tuple of Tensor
      - dict (按 key 名查找分类输出)
      - 单个 Tensor
    """
    # 情况1：list/tuple — 过滤空张量和非张量
    if isinstance(preds, (list, tuple)):
        items = [p for p in preds if isinstance(p, torch.Tensor) and p.numel() > 0]
        if items:
            return items
    
    # 情况2：dict — 按 key 查找
    if isinstance(preds, dict):
        for k in ('cls', 'cls_pred', 'classification'):
            if k in preds and isinstance(preds[k], torch.Tensor):
                return [preds[k]]
        items = [v for v in preds.values() if isinstance(v, torch.Tensor) and v.numel() > 0]
        if items:
            return items
    
    # 情况3：单个张量
    if isinstance(preds, torch.Tensor) and preds.numel() > 0:
        return [preds]
    
    return []


def _extract_cls_tensor(feat: torch.Tensor, num_classes: int) -> torch.Tensor:
    """从特征图中提取类别预测部分"""
    try:
        if feat.dim() == 4:
            # [B, C, H, W] → 取最后 num_classes 个通道作为分类分量
            if feat.shape[1] >= num_classes:
                return feat[:, -num_classes:, :, :]
            else:
                # 通道不足时，取所有通道并警告
                logger = __import__('logging').getLogger('EdgeDistillDet.Loss')
                logger.warning(f"特征图通道({feat.shape[1]}) < 类别数({num_classes})，使用全部通道")
                return feat
        elif feat.dim() == 3:
            if feat.shape[-1] == num_classes:
                return feat
            elif feat.shape[1] == num_classes:
                return feat.permute(0, 2, 1).contiguous()
            else:
                return feat
        
        raise ValueError(f"无法提取类别分量: shape={tuple(feat.shape)}, num_classes={num_classes}")
    except Exception as e:
        logger = __import__('logging').getLogger('EdgeDistillDet.Loss')
        logger.warning(f"类别提取异常，回退到原始张量: {e}")
        return feat


class CosineTemperatureScheduler:
    """余弦退火温度调度器 — 从 T_max 线性衰减至 T_min"""
    
    def __init__(self, T_max: float = 6.0, T_min: float = 1.5, total_epochs: int = 150):
        self.T_max = max(T_max, 1.0)
        self.T_min = max(T_min, 0.5)
        self.total_epochs = max(total_epochs, 1)
        self._current_T = self.T_max
    
    def step(self, epoch: int) -> float:
        progress = min(epoch / max(self.total_epochs, 1), 1.0)
        self._current_T = self.T_min + 0.5 * (self.T_max - self.T_min) * (1.0 + math.cos(math.pi * progress))
        return self._current_T
    
    @property
    def current_temperature(self) -> float:
        return self._current_T


class AdaptiveTemperatureKDLoss(nn.Module):
    """自适应温度 + 小目标加权的 KL 散度蒸馏损失"""
    
    def __init__(self, temperature: float = 4.0, small_area_thresh: float = 0.005,
                 scale_boost: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.small_area_thresh = small_area_thresh
        self.scale_boost = scale_boost
    
    def forward(self, student_preds: List[torch.Tensor], teacher_preds: List[torch.Tensor],
                num_classes: int, temperature: Optional[float] = None) -> torch.Tensor:
        T = temperature if temperature is not None else self.temperature
        s_list = _normalize_preds(student_preds, num_classes)
        t_list = _normalize_preds(teacher_preds, num_classes)
        
        if not s_list or not t_list:
            device = torch.device('cpu')
            if student_preds:
                try:
                    device = next(iter(student_preds)).device if hasattr(student_preds, '__iter__') else student_preds.device
                except (StopIteration, TypeError, RuntimeError):
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
            return torch.tensor(0.0, device=device)
        
        device = s_list[0].device
        t_list = [_safe_to_device(t, device) for t in t_list]
        
        total_kd = torch.tensor(0.0, device=device)
        valid_pairs = 0
        
        for s_feat, t_feat in zip(s_list, t_list):
            try:
                s_cls = _extract_cls_tensor(s_feat, num_classes)
                t_cls = _extract_cls_tensor(t_feat, num_classes)
                
                # 使用对齐函数处理不同模型的空间维度差异（关键修复）
                # YOLOv8n(学生) vs YOLOv8m(教师) 检测头输出空间尺寸不同
                s_flat, t_flat = _align_and_flatten(s_cls, t_cls, num_classes)
                
                if s_flat.shape[0] == 0 or t_flat.shape[0] == 0:
                    continue
                
                # 小目标加权：基于教师模型置信度
                with torch.no_grad():
                    t_score = t_flat.softmax(dim=-1).max(dim=-1).values
                    small_mask = t_score > self.small_area_thresh
                
                weights = torch.ones(s_flat.shape[0], device=device)
                weights[small_mask] = self.scale_boost
                
                # KL 散度计算
                kl = F.kl_div(
                    F.log_softmax(s_flat / T, dim=-1),
                    F.softmax(t_flat / T, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                
                weighted_kl = (kl * weights).mean() * (T ** 2)
                total_kd = total_kd + weighted_kl
                valid_pairs += 1
                
            except Exception as e:
                # 单对特征失败不影响整体
                __import__('logging').getLogger('EdgeDistillDet.Loss') \
                    .warning(f"KL 散度计算跳过一对特征: {e}")
                continue
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        return total_kd / valid_pairs  # 平均化


class SmallTargetFocalKDLoss(nn.Module):
    """焦点蒸馏损失 — 对难样本给予更高权重"""
    
    def __init__(self, temperature: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
    
    def forward(self, student_preds: List[torch.Tensor], teacher_preds: List[torch.Tensor],
                num_classes: int, temperature: Optional[float] = None) -> torch.Tensor:
        T = temperature if temperature is not None else self.temperature
        s_list = _normalize_preds(student_preds, num_classes)
        t_list = _normalize_preds(teacher_preds, num_classes)
        
        if not s_list or not t_list:
            device = torch.device('cpu')
            if student_preds:
                try:
                    device = next(iter(student_preds)).device if hasattr(student_preds, '__iter__') else getattr(student_preds, 'device', torch.device('cpu'))
                except (StopIteration, TypeError, RuntimeError):
                    pass
            return torch.tensor(0.0, device=device)
        
        device = s_list[0].device
        t_list = [_safe_to_device(t, device) for t in t_list]
        
        total = torch.tensor(0.0, device=device)
        valid_pairs = 0
        
        for s_feat, t_feat in zip(s_list, t_list):
            try:
                s_cls = _extract_cls_tensor(s_feat, num_classes)
                t_cls = _extract_cls_tensor(t_feat, num_classes)
                
                # 使用对齐函数处理不同模型的空间维度差异（关键修复）
                s_flat, t_flat = _align_and_flatten(s_cls, t_cls, num_classes)
                
                s_prob = F.softmax(s_flat / T, dim=-1)
                t_prob = F.softmax(t_flat / T, dim=-1)
                
                # 协议度权重：师生预测越不一致，权重越高
                s_max = s_prob.max(dim=-1).values
                t_max = t_prob.max(dim=-1).values
                agreement = 2 * s_max * t_max / (s_max + t_max + 1e-8)
                focal_weight = (1.0 - agreement) ** self.gamma
                
                kl = F.kl_div(
                    F.log_softmax(s_flat / T, dim=-1),
                    t_prob,
                    reduction="none",
                ).sum(dim=-1)
                
                total = total + (focal_weight * kl).mean() * (T ** 2)
                valid_pairs += 1
                
            except Exception as e:
                __import__('logging').getLogger('EdgeDistillDet.Loss') \
                    .warning(f"Focal KL 计算跳过一对特征: {e}")
                continue
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        return total / valid_pairs


class FeatureAlignmentLoss(nn.Module):
    """特征对齐损失（MSE）— 可选组件"""
    
    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        super().__init__()
        assert len(student_channels) == len(teacher_channels), \
            f"学生/教师通道数不匹配: {student_channels} vs {teacher_channels}"
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sc, tc, kernel_size=1, bias=False),
                nn.BatchNorm2d(tc),
            )
            for sc, tc in zip(student_channels, teacher_channels)
        ])
    
    def forward(self, student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor]) -> torch.Tensor:
        if not student_feats or not teacher_feats:
            device = torch.device('cpu')
            if student_feats:
                try: device = student_feats[0].device
                except (IndexError, AttributeError, RuntimeError): pass
            return torch.tensor(0.0, device=device)
        
        device = student_feats[0].device
        teacher_feats = [_safe_to_device(t, device) for t in teacher_feats]
        
        total = torch.tensor(0.0, device=device)
        count = 0
        for proj, s_f, t_f in zip(self.projectors, student_feats, teacher_feats):
            try:
                s_proj = proj(s_f)
                if s_proj.shape[-2:] != t_f.shape[-2:]:
                    s_proj = F.adaptive_avg_pool2d(s_proj, t_f.shape[-2:])
                s_norm = F.normalize(s_proj.reshape(s_proj.size(0), -1), dim=-1)
                t_norm = F.normalize(t_f.reshape(t_f.size(0), -1), dim=-1)
                total = total + F.mse_loss(s_norm, t_norm)
                count += 1
            except Exception as e:
                __import__('logging').getLogger('EdgeDistillDet.Loss') \
                    .warning(f"特征对齐计算跳过: {e}")
                continue
        
        if count == 0:
            return torch.tensor(0.0, device=device)
        return total / count


class CompositiveDistillLoss(nn.Module):
    """复合蒸馏损失 — 组合多种蒸馏策略"""
    
    def __init__(self, w_kd: float = 0.5, w_focal: float = 0.3, w_feat: float = 0.2,
                 student_channels: Optional[List[int]] = None,
                 teacher_channels: Optional[List[int]] = None,
                 temperature: float = 4.0, small_area_thresh: float = 0.005,
                 scale_boost: float = 2.0, focal_gamma: float = 2.0):
        super().__init__()
        self.w_kd = w_kd
        self.w_focal = w_focal
        self.w_feat = w_feat
        self.use_feat = (student_channels is not None and teacher_channels is not None and w_feat > 0)
        
        self.kd_loss = AdaptiveTemperatureKDLoss(
            temperature=temperature, small_area_thresh=small_area_thresh, scale_boost=scale_boost
        )
        self.focal_kd_loss = SmallTargetFocalKDLoss(temperature=temperature, gamma=focal_gamma)
        
        if self.use_feat:
            self.feat_loss = FeatureAlignmentLoss(student_channels, teacher_channels)
    
    def forward(self, student_preds: List[torch.Tensor], teacher_preds: List[torch.Tensor],
                num_classes: int, temperature: float,
                student_feats: Optional[List[torch.Tensor]] = None,
                teacher_feats: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, dict]:
        s_list = _normalize_preds(student_preds, num_classes)
        t_list = _normalize_preds(teacher_preds, num_classes)
        
        if not s_list:
            device = torch.device('cpu')
            if student_preds:
                try:
                    device = next(iter(student_preds)).device if hasattr(student_preds, '__iter__') else getattr(student_preds, 'device', torch.device('cpu'))
                except (StopIteration, TypeError, RuntimeError):
                    pass
            total = torch.tensor(0.0, device=device)
            return total, {"loss_kd_adaptive": 0.0, "loss_kd_focal": 0.0,
                          "loss_feat_align": 0.0, "loss_distill_total": 0.0}
        
        device = s_list[0].device
        
        l_kd = self.kd_loss(s_list, t_list, num_classes, temperature)
        l_focal = self.focal_kd_loss(s_list, t_list, num_classes, temperature)
        l_feat = torch.tensor(0.0, device=device)
        
        if self.use_feat and student_feats and teacher_feats:
            l_feat = self.feat_loss(student_feats, teacher_feats)
        
        total = self.w_kd * l_kd + self.w_focal * l_focal + self.w_feat * l_feat
        
        detail = {
            "loss_kd_adaptive": _safe_scalar(l_kd),
            "loss_kd_focal": _safe_scalar(l_focal),
            "loss_feat_align": _safe_scalar(l_feat),
            "loss_distill_total": _safe_scalar(total),
        }
        return total, detail


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级工具函数（单一定义，供 adaptive_kd_trainer.py 导入）
# ═══════════════════════════════════════════════════════════════════════════════
# _safe_scalar 已在模块顶部定义（第19行），此处不再重复定义
# ═══════════════════════════════════════════════════════════════════════════════
