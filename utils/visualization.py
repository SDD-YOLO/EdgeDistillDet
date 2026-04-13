"""
utils/visualization.py
========================
数据集可视化工具（DatasetVisualizer）

在原 chart.py 基础上重构，新增：
  1. 尺寸分布直方图（面积分布 log 轴）
  2. 场景构成饼图
  3. 样本拼图（多样性导向采样，自动放大小目标 Inset）
  4. 标注热力图（目标中心点密度）
  5. 统一输出到指定目录，支持 DPI 参数
"""

import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from utils.dataset_analyzer import (
    DatasetAnalyzer, _classify_scene, _parse_label, _find_label,
    SIZE_BINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 配色方案
# ─────────────────────────────────────────────────────────────────────────────
SCENE_COLORS = {
    "sunny":    "#F5A623",
    "overcast": "#7B8FA1",
    "dusk":     "#E05C5C",
    "night":    "#4A90D9",
}
SCENE_LABELS = {
    "sunny":    "Sunny / Day",
    "overcast": "Overcast / Hazy",
    "dusk":     "Dusk / Golden Hr",
    "night":    "Night / Low-light",
}
SIZE_COLORS = {
    "tiny":   "#E74C3C",
    "small":  "#E67E22",
    "medium": "#3498DB",
    "large":  "#2ECC71",
}

SMALL_AREA_THRESH = 32 * 32   # px²


class DatasetVisualizer:
    """
    数据集全面可视化生成器。

    用法：
        viz = DatasetVisualizer(dataset_root, output_dir)
        viz.run_all(sample_limit=500)
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        dataset_root: str,
        output_dir: str,
        dpi: int = 200,
        seed: int = 42,
    ):
        self.root       = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi  = dpi
        random.seed(seed)
        np.random.seed(seed)

    # ──────────────────────────────────────────────────────────────────────────
    def run_all(self, sample_limit: int = 500):
        print("\n[可视化] 开始生成数据集图表...")
        metas = self._collect_meta(sample_limit)
        if not metas:
            print("  ⚠ 未找到有效图像，跳过可视化")
            return

        self._plot_size_histogram(metas)
        self._plot_scene_pie(metas)
        self._plot_sample_grid(metas)
        self._plot_center_heatmap(metas)
        print(f"\n  ✔ 所有图表已保存至 → {self.output_dir}")

    # ── 元数据收集 ────────────────────────────────────────────────────────────
    def _collect_meta(self, limit: int) -> List[dict]:
        all_meta = []
        splits = {
            "train": ("train/images", "train/labels"),
            "val":   ("valid/images", "valid/labels"),
        }
        for split, (img_rel, lbl_rel) in splits.items():
            img_dir = self.root / img_rel
            lbl_dir = self.root / lbl_rel
            if not img_dir.exists():
                continue
            paths = []
            for ext in self.IMG_EXTS:
                paths += glob.glob(str(img_dir / f"*{ext}"))
            random.shuffle(paths)

            for ip in tqdm(paths[:limit // 2], desc=f"  收集 {split}", ncols=72):
                img = cv2.imread(ip)
                if img is None:
                    continue
                H, W = img.shape[:2]
                hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
                mean_v = hsv[:,:,2].mean()
                mean_s = hsv[:,:,1].mean()
                hue_std = hsv[:,:,0].std()
                scene  = _classify_scene(mean_v, mean_s)

                lp     = _find_label(ip, str(img_dir), str(lbl_dir))
                annos  = _parse_label(lp, W, H)
                areas  = [a["area_px"] for a in annos]
                small_boxes = [
                    (a["cx"]*W - a["bw"]*W/2,
                     a["cy"]*H - a["bh"]*H/2,
                     a["cx"]*W + a["bw"]*W/2,
                     a["cy"]*H + a["bh"]*H/2)
                    for a in annos if a["area_px"] < SMALL_AREA_THRESH
                ]
                all_boxes = [
                    (a["cx"]*W - a["bw"]*W/2,
                     a["cy"]*H - a["bh"]*H/2,
                     a["cx"]*W + a["bw"]*W/2,
                     a["cy"]*H + a["bh"]*H/2,
                     a["area_px"])
                    for a in annos
                ]
                centers = [(a["cx"], a["cy"]) for a in annos]

                if not annos:
                    continue

                all_meta.append({
                    "path": ip, "H": H, "W": W,
                    "scene": scene,
                    "areas": areas,
                    "small_boxes": small_boxes,
                    "all_boxes": all_boxes,
                    "centers": centers,
                    "hue_std": hue_std,
                    "mean_v": mean_v,
                })
        return all_meta

    # ── 1. 尺寸分布直方图 ────────────────────────────────────────────────────
    def _plot_size_histogram(self, metas: List[dict]):
        all_areas = [a for m in metas for a in m["areas"]]
        if not all_areas:
            return

        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor("white")
        bins = np.logspace(np.log10(max(min(all_areas), 1)), np.log10(max(all_areas)+1), 40)
        ax.hist(all_areas, bins=bins, color="#3498DB", edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("BBox Area (px²)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Target Size Distribution (Log Scale)", fontsize=11, fontweight="bold")

        for cat, (lo, hi) in SIZE_BINS.items():
            if hi == float("inf"):
                hi = max(all_areas) * 1.5
            ax.axvspan(lo, min(hi, ax.get_xlim()[1]),
                       alpha=0.12, color=SIZE_COLORS[cat], label=cat)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", ls="--", alpha=0.4)
        fig.tight_layout()
        out = self.output_dir / "fig_size_distribution.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✔ 尺寸分布图 → {out.name}")

    # ── 2. 场景构成饼图 ──────────────────────────────────────────────────────
    def _plot_scene_pie(self, metas: List[dict]):
        cnt = defaultdict(int)
        for m in metas:
            cnt[m["scene"]] += 1
        if not cnt:
            return

        labels = [SCENE_LABELS.get(k, k) for k in cnt]
        colors = [SCENE_COLORS.get(k, "#999") for k in cnt]
        sizes  = list(cnt.values())

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title("Scene Distribution", fontsize=11, fontweight="bold")
        fig.tight_layout()
        out = self.output_dir / "fig_scene_distribution.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✔ 场景分布图 → {out.name}")

    # ── 3. 样本拼图 ──────────────────────────────────────────────────────────
    def _plot_sample_grid(self, metas: List[dict], rows: int = 3):
        scenes = list(SCENE_COLORS.keys())
        groups: Dict[str, List[dict]] = defaultdict(list)
        for m in metas:
            if m["small_boxes"]:  # 只展示含小目标的图
                groups[m["scene"]].append(m)

        THUMB_H, THUMB_W = 180, 220
        BBOX_SMALL = (0, 255, 80)
        BBOX_OTHER = (160, 160, 160)
        ZOOM = 3
        ZOOM_CROP = 40

        fig, axes = plt.subplots(rows, len(scenes), figsize=(7.2, 4.8))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(wspace=0.03, hspace=0.04, top=0.90,
                            bottom=0.06, left=0.005, right=0.995)

        for j, scene in enumerate(scenes):
            pool = sorted(groups[scene], key=lambda x: -x["hue_std"])
            step = max(len(pool) // rows, 1)
            chosen = [pool[min(i * step, len(pool)-1)] for i in range(rows)]
            axes[0, j].set_title(SCENE_LABELS[scene], fontsize=7.5,
                                  fontweight="bold", pad=4, color=SCENE_COLORS[scene])

            for i in range(rows):
                ax = axes[i, j]
                ax.axis("off")
                if i >= len(chosen):
                    continue
                m = chosen[i]
                img = cv2.imread(m["path"])
                if img is None:
                    continue
                H, W = img.shape[:2]

                # 画框
                for bx1,by1,bx2,by2,area in m["all_boxes"]:
                    color = BBOX_SMALL if area < SMALL_AREA_THRESH else BBOX_OTHER
                    thick = 2 if area < SMALL_AREA_THRESH else 1
                    cv2.rectangle(img, (int(max(0,bx1)),int(max(0,by1))),
                                       (int(min(W,bx2)),int(min(H,by2))), color, thick)

                # 放大 inset
                if m["small_boxes"]:
                    sx1,sy1,sx2,sy2 = m["small_boxes"][0]
                    cx_t = int((sx1+sx2)/2); cy_t = int((sy1+sy2)/2)
                    half = ZOOM_CROP // 2
                    rx1 = max(0,cx_t-half); rx2 = min(W,cx_t+half)
                    ry1 = max(0,cy_t-half); ry2 = min(H,cy_t+half)
                    crop = img[ry1:ry2, rx1:rx2]
                    if crop.size > 0:
                        zh, zw = crop.shape[0]*ZOOM, crop.shape[1]*ZOOM
                        zoomed = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_NEAREST)
                        ey2 = H-4; ey1 = max(0, ey2-zh)
                        ex2 = W-4; ex1 = max(0, ex2-zw)
                        clip_h = ey2-ey1; clip_w = ex2-ex1
                        if clip_h > 0 and clip_w > 0:
                            cv2.rectangle(img, (ex1-2,ey1-2),(ex2+2,ey2+2),(255,255,255),2)
                            img[ey1:ey2, ex1:ex2] = zoomed[:clip_h, :clip_w]

                # 缩放居中裁剪
                scale = THUMB_H / H
                thumb = cv2.resize(img, (int(W*scale), THUMB_H), interpolation=cv2.INTER_AREA)
                th, tw = thumb.shape[:2]
                if tw > THUMB_W:
                    x0 = (tw-THUMB_W)//2; thumb = thumb[:, x0:x0+THUMB_W]
                elif tw < THUMB_W:
                    pad = np.zeros((th, THUMB_W-tw, 3), dtype=np.uint8)
                    thumb = np.concatenate([thumb, pad], axis=1)

                ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), aspect="auto")
                n_s = len(m["small_boxes"])
                ax.text(0.03, 0.04, f"{n_s} small target{'s' if n_s>1 else ''}",
                        transform=ax.transAxes, fontsize=5.5, color="white",
                        va="bottom", fontweight="bold",
                        bbox=dict(fc="#000", alpha=0.5, pad=1, lw=0,
                                  boxstyle="round,pad=0.2"))

        legend_elems = [
            mpatches.Patch(fc="none", ec="#00FF50", lw=1.5, label="Small target BBox"),
            mpatches.Patch(fc="none", ec="#aaa",    lw=1.0, label="Other BBox"),
            mpatches.Patch(fc="white", ec="white",  lw=1.5, label="Zoomed inset (3×)"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", ncol=3,
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

        out = self.output_dir / "fig_sample_grid.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✔ 样本拼图   → {out.name}")

    # ── 4. 标注中心点热力图 ──────────────────────────────────────────────────
    def _plot_center_heatmap(self, metas: List[dict], bins: int = 64):
        all_cx, all_cy = [], []
        for m in metas:
            for cx, cy in m["centers"]:
                all_cx.append(cx)
                all_cy.append(cy)
        if not all_cx:
            return

        heatmap, _, _ = np.histogram2d(all_cx, all_cy, bins=bins,
                                        range=[[0,1],[0,1]])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor("white")
        im = ax.imshow(heatmap.T, origin="lower", aspect="equal",
                       extent=[0,1,0,1], cmap="YlOrRd", interpolation="bilinear")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Annotation Density")
        ax.set_xlabel("Normalized X", fontsize=9)
        ax.set_ylabel("Normalized Y", fontsize=9)
        ax.set_title("Target Center Heatmap", fontsize=11, fontweight="bold")
        fig.tight_layout()
        out = self.output_dir / "fig_center_heatmap.png"
        fig.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ✔ 热力图     → {out.name}")
