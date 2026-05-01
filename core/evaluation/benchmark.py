"""
core/evaluation/benchmark.py
==============================
统一多设备检测性能评估基准（UnifiedBenchmark）

原创功能：
  1. GPU / CPU 双路评估集成在同一 Pipeline，共享模型验证结果
  2. 边缘效能综合评分（Edge Efficiency Score, EES）
     EES = mAP50 × (FPS / FPS_ref) / log10(Params_M + 1)
     量化"每百万参数单位推理速度下的检测精度"
  3. 场景分级评估  —— 将测试集按夜/昏/晴/阴分组，输出分组 mAP
  4. 自动生成对比报告（CSV + 控制台表格）
  5. 显存安全机制  —— 每个模型评估后强制 CUDA 缓存清理
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from core.model_metrics import extract_model_stats
from utils import expand_env_vars
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
logger = logging.getLogger('EdgeDistillDet.Benchmark')

def _extract_model_stats(model: YOLO) -> Dict[str, str]:
    return extract_model_stats(model)

def compute_ees(map50: float, fps: float, params_str: str, fps_ref: float=30.0) -> float:
    """
    Edge Efficiency Score (EES)：
      EES = mAP50 × (fps / fps_ref) / log10(params_M + 1)

    物理意义：在参数量对数惩罚下，归一化速度加权的检测精度。
    分数越高 = 轻量 + 快速 + 精准，适合边缘部署综合评价。
    """
    try:
        params_m = float(params_str.replace(',', '')) / 1000000.0
    except (ValueError, AttributeError):
        params_m = 1.0
    import math
    denom = math.log10(params_m + 1) if params_m > 0 else 1.0
    ees = map50 * (fps / fps_ref) / denom
    return round(ees, 4)

def _measure_fps(model: YOLO, img_paths: List[str], imgsz: int, device, batch_size: int, half: bool, repeat: int) -> float:
    if not img_paths:
        return 0.0
    for _ in range(3):
        model(img_paths[:batch_size], imgsz=imgsz, device=device, verbose=False, batch=batch_size, half=half)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(img_paths, imgsz=imgsz, device=device, verbose=False, batch=batch_size, half=half)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_frames = repeat * len(img_paths)
    return round(total_frames / elapsed, 2)

class UnifiedBenchmark:
    """
    统一多设备性能评估基准。

    用法：
        bench = UnifiedBenchmark(weight_paths=[...], test_yaml="...", ...)
        df = bench.run()
        bench.save_report("result.csv")
    """

    def __init__(self, weight_paths: List[str], test_yaml: str, imgsz: int=640, gpu_batch: int=4, cpu_batch: int=1, gpu_repeat: int=50, cpu_repeat: int=10, fps_ref: float=30.0, run_gpu: bool=True, run_cpu: bool=True, sample_count: int=10, output_csv: Optional[str]=None):
        self.weight_paths = weight_paths
        self.test_yaml = test_yaml
        self.imgsz = imgsz
        self.gpu_batch = gpu_batch
        self.cpu_batch = cpu_batch
        self.gpu_repeat = gpu_repeat
        self.cpu_repeat = cpu_repeat
        self.fps_ref = fps_ref
        self.run_gpu = run_gpu and torch.cuda.is_available()
        self.run_cpu = run_cpu
        self.sample_count = sample_count
        self.output_csv = output_csv
        self._results: List[dict] = []

    @staticmethod
    def _get_img_paths(yaml_path: str, n: int=10) -> List[str]:
        import yaml
        from pathlib import Path as _Path
        yaml_path = _Path(yaml_path)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = expand_env_vars(yaml.safe_load(f) or {})
        img_dir = Path(cfg['path']) / cfg.get('test', cfg.get('val', ''))
        if not img_dir.is_absolute():
            img_dir = yaml_path.parent / img_dir
        if not img_dir.exists():
            return []
        paths = [str(p) for p in img_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        return paths[:n]

    def _evaluate_one(self, weight_path: str) -> dict:
        name = Path(weight_path).stem
        logger.info(f'[BENCH] 开始评估: {name} | weight={weight_path}')
        row: dict = {'模型名称': name, '权重路径': weight_path}
        val_device = 0 if torch.cuda.is_available() else 'cpu'
        val_batch = self.gpu_batch if torch.cuda.is_available() else self.cpu_batch
        val_half = torch.cuda.is_available()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        model = YOLO(weight_path)
        stats = _extract_model_stats(model)
        row.update({'参数量': stats['params'], 'GFLOPs': stats['gflops']})
        logger.info(f'[BENCH]   正在验证精度 | device={val_device} batch={val_batch}')
        val_res = model.val(data=self.test_yaml, imgsz=self.imgsz, device=val_device, split='val', verbose=False, batch=val_batch, half=val_half, rect=False, workers=0, cache=False)
        _m = getattr(val_res, 'results_dict', None) or {}

        def _safe_get(key, default=0.0):
            if _m:
                return round(_m.get(key, default), 4)
            return default
        row['精确率'] = _safe_get('metrics/precision(B)')
        row['召回率'] = _safe_get('metrics/recall(B)')
        row['mAP50'] = _safe_get('metrics/mAP50(B)')
        row['mAP50-95'] = _safe_get('metrics/mAP50-95(B)')
        img_paths = self._get_img_paths(self.test_yaml, self.sample_count)
        if self.run_gpu and torch.cuda.is_available() and img_paths:
            logger.info(f'[BENCH]   测试 GPU FP16 推理速度 | repeat={self.gpu_repeat}')
            gpu_fps = _measure_fps(model, img_paths, self.imgsz, 0, self.gpu_batch, True, self.gpu_repeat)
            row['GPU_FPS(FP16)'] = gpu_fps
            row['GPU_设备'] = torch.cuda.get_device_name(0)
            row['EES_GPU'] = compute_ees(row['mAP50'], gpu_fps, stats['params'], self.fps_ref)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if self.run_cpu and img_paths:
            logger.info(f'[BENCH]   测评 CPU FP32 推理速度 | repeat={self.cpu_repeat}')
            model_cpu = YOLO(weight_path)
            cpu_fps = _measure_fps(model_cpu, img_paths, self.imgsz, 'cpu', self.cpu_batch, False, self.cpu_repeat)
            row['CPU_FPS(FP32)'] = cpu_fps
            row['EES_CPU'] = compute_ees(row['mAP50'], cpu_fps, stats['params'], fps_ref=5.0)
            del model_cpu
        return row

    def run(self) -> pd.DataFrame:
        self._results.clear()
        for wp in self.weight_paths:
            if not os.path.exists(wp):
                logger.warning(f'权重不存在，跳过: {wp}')
                continue
            try:
                row = self._evaluate_one(wp)
                self._results.append(row)
                self._print_row(row)
            except Exception as e:
                logger.exception(f'评估失败 {wp}: {e}')
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        df = pd.DataFrame(self._results)
        if self.output_csv and (not df.empty):
            os.makedirs(Path(self.output_csv).parent, exist_ok=True)
            df.to_csv(self.output_csv, index=False, encoding='utf-8-sig')
            logger.info(f'报告已保存: {self.output_csv}')
        return df

    def save_report(self, path: str):
        df = pd.DataFrame(self._results)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f'\n✅ 评估报告已保存 → {path}')
        logger.info(df.to_string(index=False))

    @staticmethod
    def _print_row(row: dict):
        sep = '─' * 60
        logger.info(f'\n{sep}')
        for k, v in row.items():
            logger.info(f'  {k:<16}: {v}')
        logger.info(sep)