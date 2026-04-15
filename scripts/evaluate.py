"""
scripts/evaluate.py
====================
多设备统一性能评估脚本
"""

import sys
import logging
import argparse
import yaml

# 确保项目根目录在路径中（scripts/ -> 项目根目录）
_SCRIPT_DIR = str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0, _SCRIPT_DIR)
_ROOT = __import__('pathlib').Path(_SCRIPT_DIR).parent.resolve()

from core.evaluation.benchmark import UnifiedBenchmark

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    stream=sys.stdout)  # 输出到 stdout 以便前端捕获
logger = logging.getLogger("EdgeDistillDet.Evaluate")


def run_evaluation(config_path: str):
    # 引用 main.py 的统一 BANNER
    try:
        from main import BANNER
        print(BANNER)
    except ImportError:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║          🎯  EdgeDistillDet  Model Evaluation (v1.0)               ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    eval_cfg   = cfg.get("evaluation", {})
    output_cfg = cfg.get("output", {})

    print(f"[评估] 配置文件: {config_path}")

    # ── 数据集 YAML 解析（优先级：evaluation.test_yaml > training.data_yaml > coco128.yaml）──
    test_yaml = eval_cfg.get("test_yaml")
    if not test_yaml:
        test_yaml = cfg.get("training", {}).get("data_yaml")
    if not test_yaml:
        test_yaml = "coco128.yaml"

    from pathlib import Path
    _ty_path = Path(test_yaml)
    if not _ty_path.is_absolute():
        # 先尝试相对于项目根目录
        _candidate = _ROOT / test_yaml
        if not _candidate.exists():
            # 再尝试 configs/ 目录
            _candidate = _ROOT / 'configs' / test_yaml
        if _candidate.exists():
            test_yaml = str(_candidate)
            print(f"[评估] 数据集配置: {test_yaml}")
        else:
            print(f"[警告] 数据集文件未找到: {test_yaml} (将使用默认)")
    else:
        print(f"[评估] 数据集配置: {test_yaml}")

    print(f"[评估] 图像尺寸: imgsz={eval_cfg.get('imgsz', 640)}, GPU batch={eval_cfg.get('gpu_batch', 4)}")

    weight_paths = eval_cfg.get("weight_paths", [])
    if not weight_paths:
        student_weight = (cfg.get("distillation") or {}).get("student_weight")
        teacher_weight = (cfg.get("distillation") or {}).get("teacher_weight")
        if student_weight:
            weight_paths = [student_weight]
        if teacher_weight:
            if not isinstance(weight_paths, list):
                weight_paths = []
            weight_paths.append(teacher_weight)
    print(f"[评估] 权重模型 ({len(weight_paths)} 个): {[Path(w).name for w in weight_paths]}")

    bench = UnifiedBenchmark(
        weight_paths=weight_paths,
        test_yaml=test_yaml,
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

    print("[评估] 开始运行 benchmark...")
    df = bench.run()
    logger.info("评估完成")
    if df is not None and not df.empty:
        print("\n" + "=" * 60 + "\n[评估结果]\n" + "=" * 60)
        print(df.to_string(index=False))
        csv_path = output_cfg.get("csv_path", "outputs/eval_result.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 结果已保存至: {csv_path}")
        print("=" * 60)
    else:
        print("⚠️ 未生成有效结果数据（可能所有评估均失败）")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeDistillDet 模型评估工具")
    parser.add_argument("config", nargs="?", default="configs/distill_config.yaml",
                        help="配置文件路径 (默认: configs/distill_config.yaml)")
    args = parser.parse_args()

    config_path = args.config
    from pathlib import Path
    if not Path(config_path).is_absolute():
        config_path = str(_ROOT / config_path)

    try:
        run_evaluation(config_path)
    except Exception as e:
        print(f"[错误] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
