"""
自动设备检测模块
- 优先检测 NVIDIA GPU
- CPU 模式下自动设置 DNNL 环境变量避免 bf16 反向传播错误
"""
import os
import warnings
import subprocess
from typing import Optional, List, Dict, Any
from core.logging import get_logger
logger = get_logger(__name__)

def _fix_dnnl_for_cpu():
    """
    修复 Intel Arrow Lake+ 等 CPU 的 DNNL bf16/fp16 反向传播不兼容问题。
    必须在 torch 导入前设置环境变量。
    """
    os.environ.setdefault('ONEDNN_DEFAULT_FPMATH_MODE', 'FP32')
    os.environ.setdefault('ONEDNN_ENABLE_PRIMITIVE_CACHE', '0')

def _get_available_cuda_devices() -> List[int]:
    """获取实际可用的 CUDA GPU 索引列表"""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return []

def detect_best_device(preferred: Optional[str]=None) -> str:
    """
    自动检测最优可用设备。

    Args:
        preferred: 用户偏好设备，None/'auto' 表示自动检测

    Returns:
        设备字符串: 'cpu', '0', '0,1' 等
    """
    cuda_devices = _get_available_cuda_devices()
    if preferred and preferred.strip().lower() not in ('', 'auto', 'none'):
        p = preferred.strip().lower()
        if p == 'cpu':
            _fix_dnnl_for_cpu()
            return 'cpu'
        if cuda_devices:
            requested_indices = [int(x.strip()) for x in p.split(',')]
            valid_indices = [idx for idx in requested_indices if idx in cuda_devices]
            if valid_indices:
                return ','.join((str(i) for i in valid_indices))
            logger.info(f'[AutoDevice] 请求的设备 {requested_indices} 不可用，可用 GPU: {cuda_devices}，回退到 GPU {cuda_devices[0]}')
            return str(cuda_devices[0])
        else:
            logger.info(f'[AutoDevice] 请求 GPU {p} 但 CUDA 不可用，回退到 CPU')
            _fix_dnnl_for_cpu()
            return 'cpu'
    if cuda_devices:
        return str(cuda_devices[0])
    _fix_dnnl_for_cpu()
    return 'cpu'

def list_all_devices() -> List[Dict[str, Any]]:
    """
    列出所有可用计算设备，供 Web UI 前端展示选择。
    
    Returns:
        设备列表，每项包含 id, name, type, memory_gb, available
    """
    devices = []
    try:
        import psutil
        mem_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        mem_gb = 16.0
    try:
        import platform
        cpu_name = platform.processor() or platform.machine() or 'Unknown CPU'
    except Exception:
        cpu_name = 'CPU'
    devices.append({'id': 'cpu', 'name': cpu_name, 'type': 'cpu', 'memory_gb': mem_gb, 'available': True})
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, encoding='utf-8')
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        name = parts[1]
                        mem_mb = float(parts[2])
                        mem_gb = round(mem_mb / 1024, 1)
                        devices.append({'id': str(idx), 'name': name, 'type': 'cuda', 'memory_gb': mem_gb, 'available': True})
                    except (ValueError, IndexError):
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeDecodeError, Exception):
        pass
    if len(devices) == 1:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem_gb = round(props.total_memory / 1024 ** 3, 1)
                    devices.append({'id': str(i), 'name': props.name, 'type': 'cuda', 'memory_gb': mem_gb, 'available': True})
        except ImportError:
            pass
    return devices

def setup_device_for_trainer(train_cfg: dict) -> dict:
    """
    为 Ultralytics Trainer 准备设备配置。
    自动处理 CPU 模式下的 half/amp 禁用。
    自动验证并修正无效的设备索引。

    Args:
        train_cfg: 训练配置字典（会被修改）

    Returns:
        更新后的配置字典
    """
    device = str(train_cfg.get('device', '')).strip()
    corrected_device = detect_best_device(preferred=device if device else None)
    train_cfg['device'] = corrected_device
    if device and device != corrected_device:
        logger.info(f"[AutoDevice] 设备配置修正: '{device}' -> '{corrected_device}'")
    if corrected_device == 'cpu':
        train_cfg['half'] = False
        train_cfg['amp'] = False
        if 'precision' in train_cfg:
            train_cfg['precision'] = 32
        logger.info(f'[AutoDevice] CPU 模式: device=cpu, half=False, amp=False')
    return train_cfg