"""
core/distillation/loss_functions.py
=====================================
自适应蒸馏损失函数模块

原创功能：
  1. AdaptiveTemperatureKDLoss  —— 余弦退火温度调度 + 目标尺度加权 KL 散度
  2. SmallTargetFocalKDLoss     —— 对小目标区域施以更高蒸馏权重的焦点蒸馏
  3. FeatureAlignmentLoss       —— 异构师生中间层特征对齐（通道自适应投影）
  4. CompositiveDistillLoss     —— 三者组合的可配置复合损失
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. 余弦退火温度调度器
# ─────────────────────────────────────────────────────────────────────────────
class CosineTemperatureScheduler:
    """
    在蒸馏训练过程中动态调节软化温度 T。

    策略：早期高温（平滑软标签，稳定梯度）→ 后期低温（增强决策边界锐度）。
    T(epoch) = T_min + 0.5*(T_max-T_min)*(1 + cos(π * epoch/total))
    """

    def __init__(self, T_max: float = 6.0, T_min: float = 1.5, total_epochs: int = 150):
        self.T_max = T_max
        self.T_min = T_min
        self.total_epochs = total_epochs
        self._current_T = T_max

    def step(self, epoch: int) -> float:
        progress = min(epoch / max(self.total_epochs, 1), 1.0)
        self._current_T = self.T_min + 0.5 * (self.T_max - self.T_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        return self._current_T

    @property
    def current_temperature(self) -> float:
        return self._current_T


# ─────────────────────────────────────────────────────────────────────────────
# 2. 目标尺度加权 KL 散度蒸馏损失
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveTemperatureKDLoss(nn.Module):
    """
    自适应温度 + 小目标加权的 KL 散度蒸馏损失。

    核心思想：
      - 对预测框面积 < small_area_thresh 的特征格点，施加 scale_boost 倍蒸馏权重。
      - 温度由外部 CosineTemperatureScheduler 动态传入，无需固定。
    """

    def __init__(
        self,
        temperature: float = 4.0,
        small_area_thresh: float = 0.005,   # 相对于特征图总格点的比例阈值
        scale_boost: float = 2.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.small_area_thresh = small_area_thresh
        self.scale_boost = scale_boost

    def forward(
        self,
        student_preds: List[torch.Tensor],
        teacher_preds: List[torch.Tensor],
        num_classes: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        T = temperature if temperature is not None else self.temperature
        total_kd = torch.tensor(0.0, device=student_preds[0].device)

        for s_feat, t_feat in zip(student_preds, teacher_preds):
            # 取分类通道：最后 num_classes 个通道
            s_cls = s_feat[:, -num_classes:, :, :]
            t_cls = t_feat[:, -num_classes:, :, :]

            B, C, H, W = s_cls.shape
            N = H * W

            s_flat = s_cls.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*N, C]
            t_flat = t_cls.permute(0, 2, 3, 1).contiguous().view(-1, C)

            # ── 小目标格点权重 ──────────────────────────────────────────
            # 用教师置信度最大值作为目标存在性的代理
            t_score = t_flat.softmax(dim=-1).max(dim=-1).values  # [B*N]
            small_mask = t_score > self.small_area_thresh
            weights = torch.ones(B * N, device=s_flat.device)
            weights[small_mask] = self.scale_boost

            kl = F.kl_div(
                F.log_softmax(s_flat / T, dim=-1),
                F.softmax(t_flat / T, dim=-1),
                reduction="none",
            ).sum(dim=-1)   # [B*N]

            weighted_kl = (kl * weights).mean() * (T ** 2)
            total_kd = total_kd + weighted_kl

        return total_kd


# ─────────────────────────────────────────────────────────────────────────────
# 3. 小目标焦点蒸馏损失（Focal KD）
# ─────────────────────────────────────────────────────────────────────────────
class SmallTargetFocalKDLoss(nn.Module):
    """
    焦点蒸馏损失：对师生分类分布差异大的格点赋予更高权重，
    避免"容易格点"主导蒸馏方向，集中优化师生差距最大的小目标区域。

    权重 w_i = (1 - p_agreement_i)^gamma
    其中 p_agreement 为师生 top-1 类别概率的调和均值。
    """

    def __init__(self, temperature: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma

    def forward(
        self,
        student_preds: List[torch.Tensor],
        teacher_preds: List[torch.Tensor],
        num_classes: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        T = temperature if temperature is not None else self.temperature
        total = torch.tensor(0.0, device=student_preds[0].device)

        for s_feat, t_feat in zip(student_preds, teacher_preds):
            s_cls = s_feat[:, -num_classes:, :, :].permute(0,2,3,1).contiguous()
            t_cls = t_feat[:, -num_classes:, :, :].permute(0,2,3,1).contiguous()
            B, H, W, C = s_cls.shape
            s_flat = s_cls.view(-1, C)
            t_flat = t_cls.view(-1, C)

            s_prob = F.softmax(s_flat / T, dim=-1)
            t_prob = F.softmax(t_flat / T, dim=-1)

            # 协议度：师生最高类别概率的调和均值
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

        return total


# ─────────────────────────────────────────────────────────────────────────────
# 4. 异构中间层特征对齐损失
# ─────────────────────────────────────────────────────────────────────────────
class FeatureAlignmentLoss(nn.Module):
    """
    通道自适应投影层将学生特征空间映射至教师特征空间后，
    计算归一化 L2 距离（cosine distance），适应师生通道数不同的异构场景。

    投影层仅含 1×1 卷积，参数量极小，不会主导训练。
    """

    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        super().__init__()
        assert len(student_channels) == len(teacher_channels), \
            "师生特征层数必须对齐"
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sc, tc, kernel_size=1, bias=False),
                nn.BatchNorm2d(tc),
            )
            for sc, tc in zip(student_channels, teacher_channels)
        ])

    def forward(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=student_feats[0].device)
        for proj, s_f, t_f in zip(self.projectors, student_feats, teacher_feats):
            s_proj = proj(s_f)
            # 若空间分辨率不匹配，自适应对齐
            if s_proj.shape[-2:] != t_f.shape[-2:]:
                s_proj = F.adaptive_avg_pool2d(s_proj, t_f.shape[-2:])
            # 按通道归一化后计算 MSE（等价于 cosine distance）
            s_norm = F.normalize(s_proj.view(s_proj.size(0), -1), dim=-1)
            t_norm = F.normalize(t_f.view(t_f.size(0), -1), dim=-1)
            total = total + F.mse_loss(s_norm, t_norm)
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 5. 复合蒸馏损失（对外统一接口）
# ─────────────────────────────────────────────────────────────────────────────
class CompositiveDistillLoss(nn.Module):
    """
    组合三种蒸馏损失，权重可在配置中独立控制：
        L_total = w_kd * L_adaptive_kd
                + w_focal * L_focal_kd
                + w_feat * L_feat_align
    """

    def __init__(
        self,
        w_kd: float = 0.5,
        w_focal: float = 0.3,
        w_feat: float = 0.2,
        student_channels: Optional[List[int]] = None,
        teacher_channels: Optional[List[int]] = None,
        temperature: float = 4.0,
        small_area_thresh: float = 0.005,
        scale_boost: float = 2.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_kd = w_kd
        self.w_focal = w_focal
        self.w_feat = w_feat
        self.use_feat = (student_channels is not None and
                         teacher_channels is not None and w_feat > 0)

        self.kd_loss = AdaptiveTemperatureKDLoss(
            temperature=temperature,
            small_area_thresh=small_area_thresh,
            scale_boost=scale_boost,
        )
        self.focal_kd_loss = SmallTargetFocalKDLoss(
            temperature=temperature,
            gamma=focal_gamma,
        )
        if self.use_feat:
            self.feat_loss = FeatureAlignmentLoss(student_channels, teacher_channels)

    def forward(
        self,
        student_preds: List[torch.Tensor],
        teacher_preds: List[torch.Tensor],
        num_classes: int,
        temperature: float,
        student_feats: Optional[List[torch.Tensor]] = None,
        teacher_feats: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        l_kd    = self.kd_loss(student_preds, teacher_preds, num_classes, temperature)
        l_focal = self.focal_kd_loss(student_preds, teacher_preds, num_classes, temperature)
        l_feat  = torch.tensor(0.0, device=l_kd.device)

        if self.use_feat and student_feats and teacher_feats:
            l_feat = self.feat_loss(student_feats, teacher_feats)

        total = self.w_kd * l_kd + self.w_focal * l_focal + self.w_feat * l_feat

        detail = {
            "loss_kd_adaptive": l_kd.item(),
            "loss_kd_focal":    l_focal.item(),
            "loss_feat_align":  l_feat.item(),
            "loss_distill_total": total.item(),
        }
        return total, detail
