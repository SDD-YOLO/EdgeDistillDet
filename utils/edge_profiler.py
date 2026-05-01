"""
utils/edge_profiler.py
=======================
边缘设备部署剖析器（EdgeProfiler）

原创功能：
  1. 设备规格数据库  —— 内置 RK3588 / Ascend310 / CPU / GPU 理论参数
  2. 理论吞吐量估算  —— 基于 GFLOPs 与设备算力上限，估算理论 FPS 上限
  3. 内存占用分析    —— 分别估算 FP32 / FP16 / INT8 三种精度下的模型显存需求
  4. 量化收益预测    —— 对比 FP32→INT8 的参数压缩比、精度风险分级
  5. 部署可行性报告  —— 综合算力/内存/功耗给出 PASS / WARN / FAIL 评级
"""
import math
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from core.model_metrics import estimate_gflops_from_weight, estimate_params_m_from_checkpoint
from core.logging import get_logger
logger = logging.getLogger('EdgeDistillDet.EdgeProfiler')

@dataclass
class DeviceSpec:
    name: str
    tops_fp16: float
    tops_int8: float
    memory_mb: int
    tdp_watt: float
    supports_fp16: bool = True
    supports_int8: bool = True
    note: str = ''
DEVICE_DB: Dict[str, DeviceSpec] = {'ascend310': DeviceSpec(name='Ascend 310', tops_fp16=8.0, tops_int8=16.0, memory_mb=8192, tdp_watt=8.0, note='华为昇腾310，需通过 ATC 工具链转换为 OM 模型'), 'rk3588': DeviceSpec(name='RK3588 NPU', tops_fp16=3.0, tops_int8=6.0, memory_mb=4096, tdp_watt=10.0, note='瑞芯微 RK3588，需通过 RKNN Toolkit2 转换'), 'cpu': DeviceSpec(name='通用 x86 CPU', tops_fp16=0.5, tops_int8=1.0, memory_mb=8192, tdp_watt=65.0, supports_fp16=False, note='通用 CPU 推理（OpenVINO / ONNX Runtime）'), 'gpu': DeviceSpec(name='桌面级 GPU (RTX 3060)', tops_fp16=13.0, tops_int8=26.0, memory_mb=12288, tdp_watt=170.0, note='参考 NVIDIA RTX 3060 规格')}

@dataclass
class ProfileResult:
    model_name: str
    target_device: str
    params_m: float
    gflops: float
    mem_fp32_mb: float
    mem_fp16_mb: float
    mem_int8_mb: float
    theoretical_fps_fp16: float
    theoretical_fps_int8: float
    quant_compress_ratio: float
    deployability: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class EdgeProfiler:
    """
    边缘部署剖析器。

    用法：
        profiler = EdgeProfiler("model.pt", target_device="rk3588")
        profiler.run_full_profile()
    """

    def __init__(self, weight_path: str, target_device: str='rk3588'):
        self.weight_path = weight_path
        self.target_device = target_device.lower()
        if self.target_device not in DEVICE_DB:
            raise ValueError(f'不支持的设备: {target_device}。可选: {list(DEVICE_DB.keys())}')
        self.device_spec = DEVICE_DB[self.target_device]

    @staticmethod
    def _count_params_from_checkpoint(path: str) -> float:
        params_m = estimate_params_m_from_checkpoint(path)
        if params_m is None or params_m <= 0:
            raise RuntimeError('模型参数计数结果为 0')
        return params_m

    def _get_model_metrics(self) -> Dict[str, float]:
        try:
            params_m = self._count_params_from_checkpoint(self.weight_path)
            gflops = estimate_gflops_from_weight(self.weight_path)
            if gflops is None:
                gflops = max(params_m * 4.5, 1.0)
            return {'params_m': params_m, 'gflops': gflops}
        except Exception as e:
            logger.warning(f'模型指标提取失败: {e}')
            raise

    @staticmethod
    def _estimate_memory(params_m: float) -> Dict[str, float]:
        """
        估算不同精度下的静态内存占用（参数 + 激活缓冲区 × 1.5 系数）。
        """
        param_bytes_fp32 = params_m * 1000000.0 * 4
        return {'fp32_mb': round(param_bytes_fp32 * 1.5 / 1024 / 1024, 1), 'fp16_mb': round(param_bytes_fp32 * 0.75 / 1024 / 1024, 1), 'int8_mb': round(param_bytes_fp32 * 0.375 / 1024 / 1024, 1)}

    @staticmethod
    def _estimate_fps(gflops: float, device_tops: float, efficiency: float=0.6) -> float:
        """
        FPS_theoretical = (device_tops × 1e12 × efficiency) / (gflops × 1e9)
        efficiency: 实际算力利用率（典型值 0.5~0.7）
        """
        if gflops <= 0:
            return 0.0
        fps = device_tops * 1000000000000.0 * efficiency / (gflops * 1000000000.0)
        return round(fps, 1)

    def _assess_deployability(self, mem_fp16_mb: float, fps_fp16: float, fps_int8: float) -> tuple:
        spec = self.device_spec
        issues, recs, status = ([], [], 'PASS')
        if mem_fp16_mb > spec.memory_mb * 0.9:
            issues.append(f'FP16 内存需求 {mem_fp16_mb}MB 超过设备可用内存 {spec.memory_mb}MB 的 90%')
            recs.append('建议使用 INT8 量化或模型剪枝')
            status = 'FAIL'
        elif mem_fp16_mb > spec.memory_mb * 0.6:
            issues.append(f'FP16 内存占比较高（{mem_fp16_mb / spec.memory_mb * 100:.0f}%）')
            recs.append('建议评估 INT8 量化可行性')
            if status == 'PASS':
                status = 'WARN'
        best_fps = fps_int8 if spec.supports_int8 else fps_fp16
        if best_fps < 10:
            issues.append(f'估算最高 FPS {best_fps} 低于实时阈值（10 fps）')
            recs.append('考虑模型压缩或降低输入分辨率')
            status = 'FAIL'
        elif best_fps < 25:
            issues.append(f'估算 FPS {best_fps} 仅达勉强实时水平')
            if status == 'PASS':
                status = 'WARN'
        if self.target_device == 'ascend310':
            recs.append('需使用 ATC 工具：atc --model=model.onnx --framework=5 ...')
        elif self.target_device == 'rk3588':
            recs.append('需使用 RKNN Toolkit2：rknn.load_onnx() → rknn.build(do_quantization=True)')
        if not issues:
            issues.append('所有检查项通过，适合在目标设备上部署')
        return (status, issues, recs)

    def run_full_profile(self) -> ProfileResult:
        model_name = Path(self.weight_path).stem
        logger.info(f"\n{'=' * 60}")
        logger.info(f'  边缘部署剖析：{model_name}  →  {self.device_spec.name}')
        logger.info(f"{'=' * 60}")
        metrics = self._get_model_metrics()
        params_m = metrics['params_m']
        gflops = metrics['gflops']
        mem = self._estimate_memory(params_m)
        fps_fp16 = self._estimate_fps(gflops, self.device_spec.tops_fp16)
        fps_int8 = self._estimate_fps(gflops, self.device_spec.tops_int8)
        compress = round(mem['fp32_mb'] / max(mem['int8_mb'], 0.01), 2)
        status, issues, recs = self._assess_deployability(mem['fp16_mb'], fps_fp16, fps_int8)
        result = ProfileResult(model_name=model_name, target_device=self.target_device, params_m=round(params_m, 3), gflops=gflops, mem_fp32_mb=mem['fp32_mb'], mem_fp16_mb=mem['fp16_mb'], mem_int8_mb=mem['int8_mb'], theoretical_fps_fp16=fps_fp16, theoretical_fps_int8=fps_int8, quant_compress_ratio=compress, deployability=status, issues=issues, recommendations=recs)
        self._print_report(result)
        return result

    @staticmethod
    def _print_report(r: ProfileResult):
        STATUS_ICON = {'PASS': '✅', 'WARN': '⚠️ ', 'FAIL': '❌'}
        icon = STATUS_ICON.get(r.deployability, '?')
        logger.info(f'\n  模型参数量     : {r.params_m:.2f} M')
        logger.info(f'  计算量         : {r.gflops} GFLOPs')
        logger.info(f'\n  内存占用估算：')
        logger.info(f'    FP32         : {r.mem_fp32_mb} MB')
        logger.info(f'    FP16         : {r.mem_fp16_mb} MB')
        logger.info(f'    INT8         : {r.mem_int8_mb} MB  (压缩比 {r.quant_compress_ratio}×)')
        logger.info(f'\n  理论推理速度：')
        logger.info(f'    FP16 模式    : {r.theoretical_fps_fp16} FPS（理论上限）')
        logger.info(f'    INT8 模式    : {r.theoretical_fps_int8} FPS（理论上限）')
        logger.info(f'\n  部署评级       : {icon} {r.deployability}')
        logger.info(f'\n  问题分析：')
        for issue in r.issues:
            logger.info(f'    · {issue}')
        logger.info(f'\n  部署建议：')
        for rec in r.recommendations:
            logger.info(f'    → {rec}')
        logger.info('')