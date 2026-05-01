"""
scripts/evaluate.py
====================
多设备统一性能评估脚本
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from core.evaluation.benchmark import UnifiedBenchmark
from core.logging import get_logger
from utils import expand_env_vars

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SCRIPT_DIR)
_ROOT = Path(_SCRIPT_DIR).parent.resolve()

logger = get_logger('EdgeDistillDet.Evaluate')

def run_evaluation(config_path: str):
    try:
        from main import __version__
        logger.info(f'EdgeDistillDet evaluation started | version={__version__}')
    except ImportError:
        logger.info('EdgeDistillDet evaluation started')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = expand_env_vars(yaml.safe_load(f) or {})
    eval_cfg = cfg.get('evaluation', {})
    output_cfg = cfg.get('output', {})
    logger.info(f'[评估] 配置文件: {config_path}')
    test_yaml = eval_cfg.get('test_yaml')
    if not test_yaml:
        test_yaml = cfg.get('training', {}).get('data_yaml')
    if not test_yaml:
        test_yaml = 'coco128.yaml'
    config_dir = Path(config_path).resolve().parent

    def _resolve_relative_path(path_str: str, base_dir: Path) -> str:
        if not isinstance(path_str, str) or not path_str:
            return path_str
        path_obj = Path(path_str)
        if path_obj.is_absolute():
            return str(path_obj)
        candidate = (base_dir / path_obj).resolve()
        if candidate.exists():
            return str(candidate)
        candidate = (_ROOT / path_obj).resolve()
        if candidate.exists():
            return str(candidate)
        return str(path_obj)
    test_yaml = _resolve_relative_path(test_yaml, config_dir)
    if Path(test_yaml).exists():
        logger.info(f'[评估] 数据集配置: {test_yaml}')
    else:
        logger.warning(f'[评估] 数据集文件未找到: {test_yaml} (将使用默认)')
    logger.info(f"[评估] 图像尺寸: imgsz={eval_cfg.get('imgsz', 640)}, GPU batch={eval_cfg.get('gpu_batch', 4)}")
    weight_paths = eval_cfg.get('weight_paths', [])
    if not weight_paths:
        student_weight = (cfg.get('distillation') or {}).get('student_weight')
        teacher_weight = (cfg.get('distillation') or {}).get('teacher_weight')
        if student_weight:
            weight_paths = [student_weight]
        if teacher_weight:
            if not isinstance(weight_paths, list):
                weight_paths = []
            weight_paths.append(teacher_weight)
    weight_paths = [_resolve_relative_path(w, config_dir) for w in weight_paths]
    logger.info(f'[评估] 权重模型 ({len(weight_paths)} 个): {[Path(w).name for w in weight_paths]}')
    bench = UnifiedBenchmark(weight_paths=weight_paths, test_yaml=test_yaml, imgsz=eval_cfg.get('imgsz', 640), gpu_batch=eval_cfg.get('gpu_batch', 4), cpu_batch=eval_cfg.get('cpu_batch', 1), gpu_repeat=eval_cfg.get('gpu_repeat', 50), cpu_repeat=eval_cfg.get('cpu_repeat', 10), fps_ref=eval_cfg.get('fps_ref', 30.0), run_gpu=eval_cfg.get('run_gpu', True), run_cpu=eval_cfg.get('run_cpu', True), sample_count=eval_cfg.get('sample_count', 10), output_csv=output_cfg.get('csv_path', 'outputs/eval_result.csv'))
    logger.info('[评估] 开始运行 benchmark...')
    df = bench.run()
    logger.info('评估完成')
    if df is not None and (not df.empty):
        logger.info('\n' + '=' * 60 + '\n[评估结果]\n' + '=' * 60)
        logger.info(df.to_string(index=False))
        csv_path = output_cfg.get('csv_path', 'outputs/eval_result.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f'结果已保存至: {csv_path}')
        logger.info('=' * 60)
    else:
        logger.warning('未生成有效结果数据（可能所有评估均失败）')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EdgeDistillDet 模型评估工具')
    parser.add_argument('config', nargs='?', default='configs/distill_config.yaml', help='配置文件路径 (默认: configs/distill_config.yaml)')
    args = parser.parse_args()
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(_ROOT / config_path)
    try:
        run_evaluation(config_path)
    except Exception as e:
        logger.exception(f'[错误] 评估失败: {e}')
        sys.exit(1)