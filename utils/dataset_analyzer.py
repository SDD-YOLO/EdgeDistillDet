"""
utils/dataset_analyzer.py
===========================
数据集统计分析器（DatasetAnalyzer）

原创功能：
  1. 小目标占比统计  —— 按 tiny/small/medium/large 分级，输出各级数量与占比
  2. 图像亮度自动场景分类  —— 晴天/阴天/黄昏/夜间四类，无需元数据标注
  3. 样本多样性评分  —— 基于色相标准差衡量视觉多样性（0~100分）
  4. 标注质量检查    —— 发现越界框、零面积框、重复框等标注异常
  5. 汇总报告导出    —— JSON + 控制台表格
"""

import glob
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils.dataset_common import (
    classify_scene,
    find_label,
    parse_label,
    size_category,
)

logger = logging.getLogger("EdgeDistillDet.DatasetAnalyzer")
_classify_scene = classify_scene
_size_category = size_category
_parse_label = parse_label
_find_label = find_label


class DatasetAnalyzer:
    """
    数据集多维统计分析器。

    用法：
        analyzer = DatasetAnalyzer(dataset_root="/path/to/DroneSOD-30K")
        report = analyzer.run(sample_limit=2000)
        analyzer.save_report("outputs/dataset_report.json")
    """

    SPLITS = {
        "train": ("train/images", "train/labels"),
        "val": ("valid/images", "valid/labels"),
        "test": ("test/images", "test/labels"),
    }
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, dataset_root: str, sample_limit: int = 2000, seed: int = 42):
        self.root = Path(dataset_root)
        self.sample_limit = sample_limit
        random.seed(seed)
        self._report: dict | None = None

    def run(self) -> dict:
        report = {
            "dataset_root": str(self.root),
            "splits": {},
            "overall": {},
            "annotation_qc": {},
        }
        overall_size_count = defaultdict(int)
        overall_scene_count = defaultdict(int)
        overall_anno_count = 0
        overall_img_count = 0
        diversity_scores = []
        qc_oob = 0
        qc_zero = 0
        for split, (img_rel, lbl_rel) in self.SPLITS.items():
            img_dir = self.root / img_rel
            lbl_dir = self.root / lbl_rel
            if not img_dir.exists():
                logger.warning(f"Split '{split}' 不存在，跳过: {img_dir}")
                continue
            img_paths = []
            for ext in self.IMG_EXTS:
                img_paths += glob.glob(str(img_dir / f"*{ext}"))
                img_paths += glob.glob(str(img_dir / f"*{ext.upper()}"))
            random.shuffle(img_paths)
            sampled = img_paths[: self.sample_limit]
            split_stats = self._analyze_split(sampled, str(img_dir), str(lbl_dir))
            report["splits"][split] = split_stats
            for k, v in split_stats["size_distribution"].items():
                overall_size_count[k] += v
            for k, v in split_stats["scene_distribution"].items():
                overall_scene_count[k] += v
            overall_anno_count += split_stats["total_annotations"]
            overall_img_count += split_stats["total_images"]
            diversity_scores.append(split_stats["diversity_score"])
            qc_oob += split_stats["qc"]["oob_boxes"]
            qc_zero += split_stats["qc"]["zero_area_boxes"]
        total_anno = max(overall_anno_count, 1)
        report["overall"] = {
            "total_images": overall_img_count,
            "total_annotations": overall_anno_count,
            "size_distribution": dict(overall_size_count),
            "size_ratio": {k: round(v / total_anno * 100, 2) for k, v in overall_size_count.items()},
            "scene_distribution": dict(overall_scene_count),
            "avg_diversity_score": round(sum(diversity_scores) / max(len(diversity_scores), 1), 2),
        }
        report["annotation_qc"] = {
            "total_oob_boxes": qc_oob,
            "total_zero_area_boxes": qc_zero,
            "quality_level": "良好" if qc_oob + qc_zero == 0 else "可接受" if qc_oob + qc_zero < overall_img_count * 0.01 else "需修复",
        }
        self._report = report
        self._print_summary(report)
        return report

    def _analyze_split(self, img_paths: list[str], img_dir: str, lbl_dir: str) -> dict:
        size_cnt = defaultdict(int)
        scene_cnt = defaultdict(int)
        hue_stds = []
        total_anno = 0
        oob_cnt = 0
        zero_cnt = 0
        for ip in tqdm(img_paths, desc=f"  分析 {Path(img_dir).parent.name}", ncols=72):
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2hSV).astype(float)
            mean_v = hsv[:, :, 2].mean()
            mean_s = hsv[:, :, 1].mean()
            hue_stds.append(hsv[:, :, 0].std())
            scene_cnt[classify_scene(mean_v, mean_s)] += 1
            lp = find_label(ip, img_dir, lbl_dir)
            annos = parse_label(lp, w, h)
            total_anno += len(annos)
            for a in annos:
                size_cnt[a["size_cat"]] += 1
                if a["is_oob"]:
                    oob_cnt += 1
                if a["is_zero"]:
                    zero_cnt += 1
        div_score = round(min(float(np.mean(hue_stds)) / 90.0 * 100, 100), 1) if hue_stds else 0.0
        return {
            "total_images": len(img_paths),
            "total_annotations": total_anno,
            "size_distribution": dict(size_cnt),
            "scene_distribution": dict(scene_cnt),
            "diversity_score": div_score,
            "qc": {"oob_boxes": oob_cnt, "zero_area_boxes": zero_cnt},
        }

    def save_report(self, output_path: str):
        if self._report is None:
            raise RuntimeError("请先调用 run() 生成报告")
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集报告已保存: {output_path}")

    @staticmethod
    def _print_summary(report: dict):
        ov = report["overall"]
        logger.info(f"\n{'─' * 50}")
        logger.info("  数据集分析报告")
        logger.info(f"{'─' * 50}")
        logger.info(f"  总图像数   : {ov['total_images']:,}")
        logger.info(f"  总标注数   : {ov['total_annotations']:,}")
        logger.info("  目标尺寸分布：")
        for k, v in ov["size_distribution"].items():
            ratio = ov["size_ratio"].get(k, 0)
            bar = "█" * int(ratio / 3)
            logger.info(f"    {k:<8}: {v:6,}  ({ratio:5.1f}%)  {bar}")
        logger.info("  场景分布：")
        for k, v in ov["scene_distribution"].items():
            logger.info(f"    {k:<10}: {v:,}")
        logger.info(f"  数据多样性评分 : {ov['avg_diversity_score']} / 100")
        qc = report["annotation_qc"]
        logger.info(f"  标注质量   : {qc['quality_level']}  (越界:{qc['total_oob_boxes']}  零面积:{qc['total_zero_area_boxes']})")
        logger.info(f"{'─' * 50}\n")
