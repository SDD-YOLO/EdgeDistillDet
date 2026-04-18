"""
core/distillation/common.py
===========================
蒸馏模块共享的轻量工具函数。
"""

from __future__ import annotations

import torch


def safe_scalar(value) -> float:
    """安全标量转换，兼容标量/任意维度张量/异常值。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return value.detach().item()
        return value.detach().mean().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
