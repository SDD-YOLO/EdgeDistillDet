"""
core/distillation/common.py
===========================
蒸馏模块共享的轻量工具函数。
"""

from __future__ import annotations

from typing import Any


def safe_scalar(value) -> float:
    """安全标量转换，兼容标量/任意维度张量/异常值。"""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().mean().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def w_feat_to_scalar(v: Any) -> float:
    """
    将配置中的 w_feat 统一为标量。历史上曾使用多元素列表，读入时取算术平均，与训练端 CompositiveDistillLoss 一致。
    """
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, list | tuple):
        if not v:
            return 0.0
        return float(sum(float(x) for x in v) / len(v))
    return safe_scalar(v)
