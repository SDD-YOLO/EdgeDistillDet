"""
scripts/evaluate.py
====================
多设备统一性能评估脚本
"""

import logging
import yaml
from core.evaluation.benchmark import UnifiedBenchmark

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("EdgeDistillDet.Evaluate")


def run_evaluation(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_cfg   = cfg.get("evaluation", {})
    output_cfg = cfg.get("output", {})

    bench = UnifiedBenchmark(
        weight_paths=eval_cfg.get("weight_paths", []),
        test_yaml=eval_cfg.get("test_yaml", "data.yaml"),
        imgsz=eval_cfg.get("imgsz", 640),
        gpu_batch=eval_cfg.get("gpu_batch", 4),
        cpu_batch=eval_cfg.get("cpu_batch", 1),
        gpu_repeat=eval_cfg.get("gpu_repeat", 50),
        cpu_repeat=eval_cfg.get("cpu_repeat", 10),
        fps_ref=eval_cfg.get("fps_ref", 30.0),
        run_gpu=eval_cfg.get("run_gpu", True),
        run_cpu=eval_cfg.get("run_cpu", True),
        sample_count=eval_cfg.get("sample_count", 10),
        output_csv=output_cfg.get("csv_path", "outputs/eval_result.csv"),
    )

    df = bench.run()
    logger.info("评估完成")
    return df
