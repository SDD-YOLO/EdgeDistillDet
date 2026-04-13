"""
scripts/visualize_dataset.py
==============================
数据集分析与可视化脚本（CLI 入口）
"""

import logging
from utils.dataset_analyzer import DatasetAnalyzer
from utils.visualization import DatasetVisualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] — %(message)s")


def run_dataset_analysis(dataset_root: str, output_dir: str = "outputs/figures"):
    # 1. 统计分析
    analyzer = DatasetAnalyzer(dataset_root, sample_limit=2000)
    report = analyzer.run()
    analyzer.save_report(f"{output_dir}/dataset_report.json")

    # 2. 可视化
    viz = DatasetVisualizer(dataset_root, output_dir, dpi=200)
    viz.run_all(sample_limit=500)

    return report
