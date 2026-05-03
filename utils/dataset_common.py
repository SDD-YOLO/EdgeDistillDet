"""
utils/dataset_common.py
=======================
数据集分析与可视化共享的纯函数与常量。
"""

from __future__ import annotations

import os
from pathlib import Path

SIZE_BINS = {
    "tiny": (0, 16 * 16),
    "small": (16 * 16, 32 * 32),
    "medium": (32 * 32, 96 * 96),
    "large": (96 * 96, float("inf")),
}

V_NIGHT = 70
V_DUSK = 130
V_SUNNY = 150
S_SUNNY = 40


def classify_scene(mean_v: float, mean_s: float) -> str:
    if mean_v < V_NIGHT:
        return "night"
    if mean_v < V_DUSK:
        return "dusk"
    if mean_v >= V_SUNNY and mean_s >= S_SUNNY:
        return "sunny"
    return "overcast"


def size_category(area_px: float) -> str:
    for cat, (low, high) in SIZE_BINS.items():
        if low <= area_px < high:
            return cat
    return "large"


def parse_label(label_path: str, width: int, height: int) -> list[dict]:
    items: list[dict] = []
    if not os.path.exists(label_path):
        return items
    with open(label_path, encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            is_oob = not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1)
            area_px = bw * width * bh * height
            items.append(
                {
                    "cls": cls_id,
                    "cx": cx,
                    "cy": cy,
                    "bw": bw,
                    "bh": bh,
                    "area_px": area_px,
                    "size_cat": size_category(area_px),
                    "is_oob": is_oob,
                    "is_zero": area_px < 1.0,
                }
            )
    return items


def find_label(img_path: str, img_dir: str, lbl_dir: str) -> str:
    stem = Path(img_path).stem
    direct = Path(lbl_dir) / f"{stem}.txt"
    if direct.exists():
        return str(direct)
    alt = Path(img_path).parent.parent / "labels" / f"{stem}.txt"
    return str(alt) if alt.exists() else str(direct)
