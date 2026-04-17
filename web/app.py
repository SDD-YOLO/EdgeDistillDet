"""
EdgeDistillDet Local UI
=======================
本地端点的自包含 Web UI，不依赖外部 server 模块。

使用方法: python web/app.py
"""

import copy
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# ==================== 路径配置（必须在项目内 import 之前） ====================
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
sys.path.insert(0, str(BASE_DIR))
CONFIG_DIR = BASE_DIR / 'configs'
TEMPLATE_FILE = WEB_DIR / 'templates' / 'index.html'
STATIC_DIR = WEB_DIR / 'static'

from utils import expand_env_vars


def _is_warning_like(text: str) -> bool:
    return bool(re.search(r'(\bwarn(ing)?\b|警告|告警|⚠|\[W\]|^\s*W\d*:|\bignoring\b|忽略|已忽略|\bdeprecated\b)', str(text or ''), re.IGNORECASE))

# ==================== 训练互斥锁（OS 级别真实文件锁，彻底杜绝竞态） ====================
_TRAIN_LOCK_FILE = BASE_DIR / '.training.lock'
# 【核心】线程级互斥锁：防止 Flask 多线程并发穿透文件锁检查
_train_thread_lock = threading.Lock()
# 全局文件锁句柄（进程生命周期内持有）
_train_fd = None


def _acquire_os_file_lock(lock_path: Path, timeout: float = 0) -> bool:
    """
    获取 OS 级别独占文件锁。
    
    与旧版 _TrainingLock 的本质区别：
      旧版用 write_text+rename 模拟锁 → 存在 TOCTOU 竞态窗口
      此版本使用 fcntl.flock / msvcrt.locking → 内核保证原子性
    
    Args:
        lock_path: 锁文件路径
        timeout: 超时秒数，0=非阻塞立即返回
    
    Returns:
        True=获取成功，False=被其他进程持有
    """
    global _train_fd
    try:
        import msvcrt
        fd = open(str(lock_path), 'w')
        # Windows: LOCK_EX | LOCK_NB（非阻塞排他锁）
        msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        _train_fd = fd
        fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
        fd.flush()
        return True
    except ImportError:
        pass
    except (OSError, IOError):
        return False

    # Linux/macOS: fcntl.flock
    try:
        import fcntl
        fd = open(str(lock_path), 'w')
        flags = fcntl.LOCK_EX | fcntl.LOCK_NB  # 排他 + 非阻塞
        fcntl.flock(fd.fileno(), flags)
        _train_fd = fd
        fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
        fd.flush()
        return True
    except (OSError, IOError):
        return False


def _release_os_file_lock():
    """释放 OS 文件锁"""
    global _train_fd
    if _train_fd is not None:
        try:
            import msvcrt
            msvcrt.unlocking(_train_fd.fileno(), 1)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            import fcntl
            fcntl.flock(_train_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _train_fd.close()
        except Exception:
            pass
        _train_fd = None


def _acquire_training_lock(timeout: float = 0) -> bool:
    """获取训练互斥锁的统一入口"""
    return _acquire_os_file_lock(_TRAIN_LOCK_FILE, timeout)


def _release_training_lock():
    """释放训练互斥锁的统一入口"""
    _release_os_file_lock()


def _is_process_alive(pid: int) -> bool:
    """跨平台进程存活检测"""
    if pid <= 0:
        return False
    try:
        import psutil as _psutil
        return _psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _kill_process_tree(pid: int, force: bool = False) -> int:
    """
    杀死整个进程树（主进程 + 所有子/孙进程）。
    
    【关键改进】旧版 _kill_old_training 只杀直接子进程 Popen 对象，
    实际上 ultralytics 训练运行在孙子进程中（Popen→python→ultralytics），
    导致孙子进程逃逸 → GPU 显存不释放 → 新旧进程同时占 GPU → OOM。
    
    Returns:
        成功终止的进程数量
    """
    killed_count = 0
    try:
        import psutil
    except ImportError:
        # 无 psutil 时降级为基础 kill
        try:
            os.kill(pid, signal.SIGTERM if not force else signal.SIGKILL)
            return 1
        except Exception:
            return 0

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 先杀所有子进程
        for child in children:
            try:
                child.terminate()
                killed_count += 1
            except psutil.NoSuchProcess:
                pass
        
        # 等待子进程退出（最多 3 秒）
        gone, alive = psutil.wait_procs(children, timeout=3)
        
        # 还活着的暴力杀
        for p in alive:
            try:
                p.kill()
                killed_count += 1
            except psutil.NoSuchProcess:
                pass
        
        # 最后杀主进程
        try:
            parent.terminate()
            killed_count += 1
        except psutil.NoSuchProcess:
            pass
        
        try:
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            try:
                parent.kill()
            except Exception:
                pass
        
        return killed_count
    except psutil.NoSuchProcess:
        return 0
    except Exception:
        return 0


def _scan_and_kill_stale_training_processes(exclude_pid: int = None) -> dict:
    """
    扫描并杀死残留的训练进程。
    
    场景：Web 服务重启后内存中的 training_process 引用丢失，
    但旧的 python.exe 训练进程仍在占用 GPU。
    通过扫描 .training.lock 文件和进程命令行特征来发现并清理。
    
    Returns:
        {'found': int, 'killed': int, 'details': str}
    """
    result = {'found': 0, 'killed': 0, 'details': ''}
    stale_pids = []
    
    # 1. 从锁文件读取 PID
    if _TRAIN_LOCK_FILE.exists():
        try:
            content = _TRAIN_LOCK_FILE.read_text().strip().split('\n')
            old_pid = int(content[0]) if content else -1
            if old_pid > 0 and old_pid != exclude_pid and _is_process_alive(old_pid):
                stale_pids.append(old_pid)
        except (ValueError, IndexError, OSError):
            pass
    
    # 2. 扫描包含训练特征的 python 进程（补充检测）
    try:
        import psutil
        our_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                if proc.pid == our_pid or proc.pid == exclude_pid:
                    continue
                if proc.info['name'] != 'python.exe' and proc.info['name'] != 'python':
                    continue
                cmdline = proc.info['cmdline'] or []
                cmdline_str = ' '.join(cmdline).lower()
                # 匹配训练脚本的特征命令行
                if any(kw in cmdline_str for kw in [
                    'train_with_distill', 'main.py train', 
                    'yolo train', 'ultralytics',
                    '--resume'
                ]):
                    if proc.pid not in stale_pids:
                        stale_pids.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass
    except Exception:
        pass
    
    result['found'] = len(stale_pids)
    
    # 3. 杀死所有发现的残留进程
    for spid in stale_pids:
        k = _kill_process_tree(spid, force=True)
        result['killed'] += k
        result['details'] += f"PID={spid}(killed={k}); "
    
    # 4. 删除僵尸锁文件
    if stale_pids:
        try:
            _TRAIN_LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    
    return result

try:
    from utils.edge_profiler import EdgeProfiler
except Exception:
    EdgeProfiler = None


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _resolve_project_path(project: str, allow_external: bool = False) -> Path:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = (BASE_DIR / project_path).resolve()
    else:
        project_path = project_path.resolve()
    if allow_external:
        return project_path
    if str(project_path).startswith(str(Path(BASE_DIR).resolve())) is False and str(Path(BASE_DIR).resolve()) not in str(project_path):
        raise ValueError('项目目录必须在仓库根目录下')
    return project_path


def _normalize_compute_provider(value: str | None) -> str:
    """统一算力平台标识，兼容多种写法。"""
    v = str(value or '').strip().lower()
    if v in {'google colab', 'google_colab', 'colab'}:
        return 'colab'
    if v in {'autodl', 'auto_dl', 'auto-dl'}:
        return 'autodl'
    if v in {'remote_api', 'remote-api', 'cloud_api', 'cloud-api'}:
        return 'remote_api'
    return 'local'


def _candidate_output_roots(project_path: Path):
    candidates = [project_path.resolve()]
    try:
        rel = project_path.resolve().relative_to(BASE_DIR.resolve())
    except Exception:
        rel = None
    if rel is None:
        return candidates

    for prefix in ('runs/detect', 'detect/runs'):
        alt = (BASE_DIR / prefix / rel).resolve()
        if str(alt).startswith(str(BASE_DIR.resolve())) and alt not in candidates:
            candidates.append(alt)

    if str(rel).replace('\\', '/') == 'runs':
        fallback = (BASE_DIR / 'runs' / 'detect' / 'runs').resolve()
        if str(fallback).startswith(str(BASE_DIR.resolve())) and fallback not in candidates:
            candidates.append(fallback)
    # 兼容旧训练产物目录：当 project 是 runs/distill 时，历史结果常落在 runs/detect/runs
    if str(rel).replace('\\', '/') == 'runs/distill':
        legacy = (BASE_DIR / 'runs' / 'detect' / 'runs').resolve()
        if str(legacy).startswith(str(BASE_DIR.resolve())) and legacy not in candidates:
            candidates.append(legacy)

    return candidates


def _load_yaml_file(path: Path):
    last_error = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp936', 'gbk', 'latin1'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                cfg = yaml.safe_load(f)
                return expand_env_vars(cfg) if cfg is not None else None
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except FileNotFoundError:
            return None
        except Exception:
            return None
    return None


def _save_yaml_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ==================== Resume 候选列表 ====================

def _list_resume_candidates(project_path: Path):
    candidates = []
    if not project_path.exists() or not project_path.is_dir():
        for prefix in ['runs/detect', 'detect/runs', 'runs/detect/' + project_path.name, 'detect/' + project_path.name, 'detect']:
            fallback = (BASE_DIR / prefix).resolve()
            if fallback.exists() and fallback.is_dir():
                candidates.extend(_scan_run_dirs_for_checkpoints(fallback, project_path))
            if project_path.name:
                alt_fallback = (BASE_DIR / prefix).resolve()
                if alt_fallback.exists() and alt_fallback.is_dir() and alt_fallback != fallback:
                    candidates.extend(_scan_run_dirs_for_checkpoints(alt_fallback, project_path))
        # 宽泛搜索：直接在 runs/ 下递归查找含 checkpoint 的目录
        if not candidates:
            runs_dir = BASE_DIR / 'runs'
            if runs_dir.exists() and runs_dir.is_dir():
                candidates.extend(_scan_run_dirs_for_checkpoints(runs_dir, project_path))
        return candidates

    candidates.extend(_scan_run_dirs_for_checkpoints(project_path, project_path))

    for prefix in ['runs/detect', 'detect/runs', 'detect']:
        detect_project = (BASE_DIR / prefix / project_path.relative_to(BASE_DIR)).resolve()
        if detect_project != project_path and detect_project.exists() and detect_project.is_dir():
            existing_names = {c['name'] for c in candidates}
            new_candidates = _scan_run_dirs_for_checkpoints(detect_project, project_path)
            for nc in new_candidates:
                if nc['name'] not in existing_names:
                    candidates.append(nc)
                    existing_names.add(nc['name'])

    candidates.sort(key=lambda c: c['modified_time'], reverse=True)
    return candidates


def _scan_run_dirs_for_checkpoints(search_path: Path, logical_project: Path):
    candidates = []
    if not search_path.exists() or not search_path.is_dir():
        return candidates
    # 递归到项目根目录为止，不再限制深度
    try:
        items = sorted(search_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    except PermissionError:
        return candidates
    for run_dir in items:
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith('.') or run_dir.name in {'runs', 'val', 'predict'}:
            continue
        checkpoint_files = []
        for rel_path, label in [
            ('last.pt', 'last.pt'),
            ('weights/last.pt', 'weights/last.pt'),
            ('weights/best.pt', 'weights/best.pt'),
        ]:
            candidate = run_dir / rel_path
            if candidate.exists():
                checkpoint_files.append((candidate, label))
        if checkpoint_files:
            checkpoint_path, checkpoint_label = checkpoint_files[0]
            candidates.append({
                'name': run_dir.name,
                'project': str(logical_project.relative_to(BASE_DIR)),
                'dir': str(run_dir.relative_to(BASE_DIR)),
                'display_name': f"{run_dir.name} — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_dir.stat().st_mtime))} ({checkpoint_label})",
                'checkpoint': str(checkpoint_path.resolve()).replace('\\', '/'),
                'checkpoint_name': checkpoint_label,
                'modified_time': run_dir.stat().st_mtime,
            })
        else:
            # 无深度限制地递归搜索子目录，直到遍历完所有层级
            sub_candidates = _scan_run_dirs_for_checkpoints(run_dir, logical_project)
            candidates.extend(sub_candidates)
    return candidates


# ==================== 模型分析工具 ====================

def _resolve_model_path(run_dir: Path):
    args_file = run_dir / 'args.yaml'
    if not args_file.exists():
        return None
    try:
        payload = _load_yaml_file(args_file)
    except Exception:
        return None
    model_value = payload.get('model')
    if not model_value:
        return None
    path = Path(model_value)
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    return path if path.exists() else None


def _estimate_model_params(model_path: Path):
    try:
        import importlib
        import torch

        def torch_load_checkpoint(path):
            try:
                return torch.load(str(path), map_location='cpu')
            except Exception:
                pass
            try:
                if hasattr(torch.serialization, 'safe_globals'):
                    safe_globals = []
                    ul_spec = importlib.util.find_spec('ultralytics')
                    if ul_spec:
                        ul = importlib.import_module('ultralytics')
                        if hasattr(ul, 'nn') and hasattr(ul.nn, 'tasks') and hasattr(ul.nn.tasks, 'DetectionModel'):
                            safe_globals.append(ul.nn.tasks.DetectionModel)
                    if safe_globals:
                        with torch.serialization.safe_globals(safe_globals):
                            return torch.load(str(path), map_location='cpu', weights_only=False)
                return torch.load(str(path), map_location='cpu', weights_only=False)
            except Exception:
                return None

        raw = torch_load_checkpoint(model_path)
        if raw is None:
            return None

        state_dict = None
        if hasattr(raw, 'state_dict'):
            state_dict = raw.state_dict()
        elif isinstance(raw, dict):
            if 'model' in raw and isinstance(raw['model'], dict):
                state_dict = raw['model']
            elif 'model' in raw and hasattr(raw['model'], 'state_dict'):
                state_dict = raw['model'].state_dict()
            elif 'state_dict' in raw and isinstance(raw['state_dict'], dict):
                state_dict = raw['state_dict']
            elif 'state_dict' in raw and hasattr(raw['state_dict'], 'state_dict'):
                state_dict = raw['state_dict'].state_dict()
            elif all(hasattr(v, 'numel') for v in raw.values()):
                state_dict = raw

        if not isinstance(state_dict, dict):
            return None

        total_params = 0
        for value in state_dict.values():
            if hasattr(value, 'numel'):
                total_params += int(value.numel())
        return total_params / 1e6 if total_params > 0 else None
    except Exception:
        return None


def _estimate_model_gflops(model_path: Path):
    try:
        from ultralytics import YOLO
        _model = YOLO(str(model_path))
        info = _model.info(verbose=False)
        del _model
        if isinstance(info, (list, tuple)) and len(info) >= 2:
            gflops = float(info[1])
            if gflops > 0:
                return gflops
    except Exception:
        pass
    return None


def _estimate_run_stats(run_dir: Path):
    model_path = _resolve_model_path(run_dir)
    if model_path is None:
        return {'ov-fps': '--', 'ov-params': '--'}

    try:
        if EdgeProfiler is not None:
            profiler = EdgeProfiler(str(model_path), target_device='gpu')
            result = profiler.run_full_profile()
            return {
                'ov-fps': f"{result.theoretical_fps_fp16:.0f}",
                'ov-params': f"{result.params_m:.1f}"
            }
    except Exception:
        pass

    params_m = _estimate_model_params(model_path)
    fps_str = '--'

    gflops = _estimate_model_gflops(model_path)
    if gflops is None and params_m is not None:
        gflops = params_m * 4.5
    if gflops is not None and gflops > 0:
        gpu_tops_fp16 = 13.0
        efficiency = 0.6
        fps = (gpu_tops_fp16 * 1e12 * efficiency) / (gflops * 1e9)
        fps_str = f"{fps:.0f}"

    return {
        'ov-fps': fps_str if fps_str != '--' else 'N/A',
        'ov-params': f"{params_m:.1f}" if params_m is not None else '--'
    }


# ==================== CSV / Metrics 解析 ====================

def _load_csv_summary(path: Path):
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
        return list(reader.fieldnames or []), rows
    except Exception:
        return [], []


def _summarize_series(rows, key, better='higher'):
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    values = [v for v in values if v is not None]
    if not values:
        return None
    final = values[-1]
    best = max(values) if better == 'higher' else min(values)
    diff = final - best
    sign = '+' if diff >= 0 else '-'
    improvement = f"{sign}{abs(diff) * 100:.2f}%"
    return {
        'best': best,
        'final': final,
        'improvement': improvement,
        'trend': 'down' if better == 'lower' else 'up'
    }


def _extract_class_performance(rows, columns, class_labels=None, run_dir=None):
    if not columns or not rows:
        return None

    if run_dir is not None:
        cache_path = Path(run_dir) / 'per_class_metrics.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                labels = cached.get('labels') or []
                map_values = cached.get('map') or []
                recall_values = cached.get('recall') or []
                precision_values = cached.get('precision') or []
                if labels and (map_values or recall_values or precision_values):
                    def normalize(arr):
                        arr = list(arr or [])
                        if len(arr) < len(labels):
                            arr.extend([None] * (len(labels) - len(arr)))
                        return arr[:len(labels)]
                    return {
                        'labels': labels,
                        'map': normalize(map_values),
                        'precision': normalize(precision_values),
                        'recall': normalize(recall_values)
                    }
            except Exception:
                pass

    class_metrics = {}
    for column in columns:
        normalized = column.strip()
        match = re.search(r'(?i)(?:class|cls)[\s_/\\-]*(?P<index>\d+)', normalized)
        if not match:
            continue
        idx = int(match.group('index'))
        lower = normalized.lower()
        metric_key = None
        if 'precision' in lower:
            metric_key = 'precision'
        elif 'recall' in lower:
            metric_key = 'recall'
        elif 'map50-95' in lower or 'mAP50-95' in normalized:
            metric_key = 'map50'
        elif 'map50' in lower or 'ap50' in lower:
            metric_key = 'map50'
        elif 'ap' in lower:
            metric_key = 'map50'
        if not metric_key:
            continue
        class_metrics.setdefault(idx, {})[metric_key] = normalized

    if not class_metrics:
        return None

    final_row = rows[-1]
    labels = []
    map_values = []
    recall_values = []
    precision_values = []
    for idx in sorted(class_metrics.keys()):
        labels.append(class_labels[idx] if class_labels and idx < len(class_labels) else f'class{idx}')
        map_col = class_metrics[idx].get('map50')
        precision_col = class_metrics[idx].get('precision')
        recall_col = class_metrics[idx].get('recall')
        map_values.append(_as_float(final_row.get(map_col)) if map_col else None)
        precision_values.append(_as_float(final_row.get(precision_col)) if precision_col else None)
        recall_values.append(_as_float(final_row.get(recall_col)) if recall_col else None)

    return {
        'labels': labels,
        'map': map_values,
        'precision': precision_values,
        'recall': recall_values
    }


def _load_distill_log_json(run_dir):
    if not run_dir or not isinstance(run_dir, (str, Path)):
        return []
    try:
        log_path = Path(run_dir) / 'distill_log.json'
        if not log_path.exists():
            parent_log = Path(run_dir).parent / 'distill_log.json'
            if parent_log.exists():
                log_path = parent_log
            else:
                return []
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _build_metric_series(rows, columns, run_dir):
    chart = {
        'epochs': [],
        'train_losses': {'box_loss': [], 'cls_loss': [], 'dfl_loss': []},
        'map_series': {'map50': [], 'map50_95': []},
        'lr_series': {'pg0': [], 'pg1': [], 'pg2': []},
        'precision_recall': {'precision': [], 'recall': []},
        'distill_series': {},
        'pr_curve': None,
        'class_performance': None
    }

    has_distill_columns = any(col.startswith('distill/') for col in (columns or []))
    distill_log_fallback = _load_distill_log_json(run_dir) if not has_distill_columns else []

    distill_by_epoch = {}
    for entry in distill_log_fallback:
        try:
            ep = int(entry.get('epoch', -1))
            if ep >= 0:
                distill_by_epoch[ep] = entry
        except (TypeError, ValueError):
            continue

    for row in rows:
        epoch = _as_float(row.get('epoch'))
        if epoch is None:
            continue
        epoch_int = int(epoch)
        chart['epochs'].append(epoch_int)
        chart['train_losses']['box_loss'].append(_as_float(row.get('train/box_loss')) or 0)
        chart['train_losses']['cls_loss'].append(_as_float(row.get('train/cls_loss')) or 0)
        chart['train_losses']['dfl_loss'].append(_as_float(row.get('train/dfl_loss')) or 0)
        chart['map_series']['map50'].append(_as_float(row.get('metrics/mAP50(B)')) or 0)
        chart['map_series']['map50_95'].append(_as_float(row.get('metrics/mAP50-95(B)')) or 0)
        chart['lr_series']['pg0'].append(_as_float(row.get('lr/pg0')) or 0)
        chart['lr_series']['pg1'].append(_as_float(row.get('lr/pg1')) or 0)
        chart['lr_series']['pg2'].append(_as_float(row.get('lr/pg2')) or 0)
        chart['precision_recall']['precision'].append(_as_float(row.get('metrics/precision(B)')) or 0)
        chart['precision_recall']['recall'].append(_as_float(row.get('metrics/recall(B)')) or 0)

        alpha_raw = row.get('distill/alpha')
        temp_raw = row.get('distill/temperature')
        kd_raw = row.get('distill/kd_loss')
        has_distill_values = any(v not in (None, '') for v in (alpha_raw, temp_raw, kd_raw))
        if has_distill_columns and has_distill_values:
            alpha_val = _as_float(alpha_raw)
            temp_val = _as_float(temp_raw)
            kd_val = _as_float(kd_raw)
            chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [alpha_val if alpha_val is not None else None]
            chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [temp_val if temp_val is not None else None]
            chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [kd_val if kd_val is not None else None]
        elif distill_by_epoch:
            de = distill_by_epoch.get(epoch_int) or distill_by_epoch.get(epoch_int - 1)
            if de:
                alpha_val = de.get('alpha')
                temp_val = de.get('temperature')
                kd_val = de.get('kd_loss') or de.get('avg_kd_loss')
                chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [_as_float(alpha_val)]
                chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [_as_float(temp_val)]
                chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [_as_float(kd_val)]
            else:
                for k in ('alpha', 'temperature', 'kd_loss'):
                    chart['distill_series'][k] = chart['distill_series'].get(k, []) + [None]

    if chart['precision_recall']['precision'] and chart['precision_recall']['recall']:
        chart['pr_curve'] = {
            'precision': chart['precision_recall']['precision'],
            'recall': chart['precision_recall']['recall']
        }
    else:
        chart['pr_curve'] = None
    chart['class_performance'] = _extract_class_performance(rows, columns, None, run_dir)
    return chart


# ==================== FastAPI App ====================

api = FastAPI(title='EdgeDistillDet Backend')
api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
api.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


def _error(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={'error': message})


@api.get('/favicon.ico')
def favicon():
    favicon_path = STATIC_DIR / 'favicon.ico'
    if not favicon_path.exists():
        return _error('favicon 不存在', 404)
    return FileResponse(str(favicon_path))

training_process = None
training_status = {
    'running': False,
    'pid': None,
    'config': None,
    'mode': 'distill',
    'start_time': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'logs': [],
}
remote_training_state = {
    'active': False,
    'job_id': '',
    'api_base_url': '',
    'logs_offset': 0,
}
# 保护 logs / running 等与 SSE、子进程读线程的并发访问（Flask 多线程 + 后台训练线程）
_train_state_lock = threading.RLock()
_train_log_cond = threading.Condition(_train_state_lock)
last_saved_config = None


def _strip_ansi(text: str) -> str:
    """去掉 ESC 序列（ultralytics 自带 TQDM 使用 \\r + \\033[K，原样进管道会触发误过滤）。"""
    if not text:
        return text
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)


def _extract_epoch_progress(line: str):
    """从不同训练日志格式中提取 (current_epoch, total_epochs)。"""
    if not line:
        return None

    patterns = [
        # 结构化日志: [EPOCH_START] epoch=1 total=10 / [EPOCH_PROGRESS] epoch=1 total=10
        r"\bepoch\s*=\s*(\d+)\s+total\s*=\s*(\d+)\b",
        # 常见格式: Epoch 1/10 或 Epoch: 1 / 10
        r"\bEpoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0 and 0 <= current <= total:
                return current, total

    # 兼容 YOLO 训练进度行（包含显存列）:
    # " 1/10  2.98G  1.266 1.555 ... 640: 12% ... 1/8 1.7s/it"
    # 注意：实时行通常不含 "GPU_mem/box_loss" 字面表头，因此不再依赖这些关键词。
    yolo_row = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s+\d+(?:\.\d+)?G\b", line)
    if yolo_row:
        current = int(yolo_row.group(1))
        total = int(yolo_row.group(2))
        if total > 0 and 0 <= current <= total:
            return current, total

    # 兜底：保留历史裸格式 + 关键词判定
    bare = re.search(r"^\s*(\d+)\s*/\s*(\d+)\b", line)
    if bare and re.search(r"\b(GPU_mem|box_loss|cls_loss|dfl_loss|Instances|Size|it/s|s/it)\b", line, re.IGNORECASE):
        current = int(bare.group(1))
        total = int(bare.group(2))
        if total > 0 and 0 <= current <= total:
            return current, total

    return None


def _update_status_line(line: str):
    """原样转发 Ultralytics / 训练子进程输出（仅剥 ANSI、去首尾空白）；不做语义过滤与去重。"""
    if not line:
        return
    clean_line = _strip_ansi(line).rstrip('\r\n')
    if not clean_line.strip():
        return

    with _train_state_lock:
        training_status['logs'].append(clean_line)
        _train_log_cond.notify_all()
        # 仅在训练未运行时裁剪：训练中裁剪会打乱长连接按索引追赶 logs 的语义，导致漏推尾部日志
        if len(training_status['logs']) > 8000 and not training_status['running']:
            training_status['logs'] = training_status['logs'][-4000:]

        progress = _extract_epoch_progress(clean_line)
        if progress:
            training_status['current_epoch'], training_status['total_epochs'] = progress


def _http_json_request(method: str, url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 20.0):
    req_headers = {'Content-Type': 'application/json'}
    if isinstance(headers, dict):
        req_headers.update({str(k): str(v) for k, v in headers.items() if k})
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url=url, data=data, headers=req_headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8', errors='replace')
        return json.loads(body) if body else {}


def _build_cloud_api_config(train_cfg: dict):
    cloud_api = dict(train_cfg.get('cloud_api', {}) or {})
    base_url = str(cloud_api.get('base_url', '') or '').strip().rstrip('/')
    if not base_url:
        raise ValueError('云训练 API 缺少 base_url')
    submit_path = str(cloud_api.get('submit_path', '/train/start') or '/train/start').strip()
    status_path = str(cloud_api.get('status_path', '/train/status') or '/train/status').strip()
    logs_path = str(cloud_api.get('logs_path', '/train/logs') or '/train/logs').strip()
    stop_path = str(cloud_api.get('stop_path', '/train/stop') or '/train/stop').strip()
    auth_token = str(cloud_api.get('token', '') or '').strip()
    headers = dict(cloud_api.get('headers', {}) or {})
    if auth_token and 'Authorization' not in headers:
        headers['Authorization'] = auth_token
    return {
        'base_url': base_url,
        'submit_url': f"{base_url}{submit_path}",
        'status_url': f"{base_url}{status_path}",
        'logs_url': f"{base_url}{logs_path}",
        'stop_url': f"{base_url}{stop_path}",
        'headers': headers,
        'poll_interval_sec': float(cloud_api.get('poll_interval_sec', 3)),
    }


def _resolve_dataset_via_api(train_cfg: dict, cfg: dict, mode: str, checkpoint: str | None):
    dataset_api = dict(train_cfg.get('dataset_api', {}) or {})
    source = str(dataset_api.get('source', '') or '').strip().lower()
    enabled = bool(dataset_api.get('enabled', False) or source == 'api')
    resolve_url = str(dataset_api.get('resolve_url', '') or '').strip()
    if not enabled or not resolve_url:
        return None

    headers = dict(dataset_api.get('headers', {}) or {})
    token = str(dataset_api.get('token', '') or '').strip()
    if not token:
        token = str((train_cfg.get('cloud_api') or {}).get('token', '') or '').strip()
    if token and 'Authorization' not in headers:
        headers['Authorization'] = token

    request_payload = dataset_api.get('request_body')
    if not isinstance(request_payload, dict):
        request_payload = {
            'dataset_name': dataset_api.get('dataset_name', ''),
            'config': cfg,
            'mode': mode,
            'checkpoint': checkpoint,
        }
    timeout_sec = float(dataset_api.get('timeout_sec', 30))
    result = _http_json_request('POST', resolve_url, payload=request_payload, headers=headers, timeout=timeout_sec)
    if not isinstance(result, dict):
        raise ValueError('数据集 API 返回格式非法（需为 JSON 对象）')

    data_yaml = str(
        result.get('data_yaml')
        or result.get('dataset_yaml')
        or result.get('dataset_path')
        or ''
    ).strip()
    dataset_id = str(result.get('dataset_id') or result.get('id') or '').strip()
    if not data_yaml and not dataset_id:
        raise ValueError('数据集 API 未返回 data_yaml 或 dataset_id')

    out = dict(result)
    out['data_yaml'] = data_yaml
    out['dataset_id'] = dataset_id
    return out


def _remote_polling_loop(api_cfg: dict, job_id: str):
    global remote_training_state
    while True:
        with _train_state_lock:
            if not remote_training_state.get('active'):
                break
            logs_offset = int(remote_training_state.get('logs_offset', 0) or 0)

        try:
            status_qs = urllib.parse.urlencode({'job_id': job_id})
            status_data = _http_json_request('GET', f"{api_cfg['status_url']}?{status_qs}", headers=api_cfg['headers'])
            state = str(status_data.get('state', '') or '').lower()
            current_epoch = int(status_data.get('current_epoch', 0) or 0)
            total_epochs = int(status_data.get('total_epochs', 0) or 0)
            with _train_state_lock:
                training_status['current_epoch'] = current_epoch
                training_status['total_epochs'] = total_epochs

            logs_qs = urllib.parse.urlencode({'job_id': job_id, 'offset': logs_offset, 'limit': 200})
            logs_data = _http_json_request('GET', f"{api_cfg['logs_url']}?{logs_qs}", headers=api_cfg['headers'])
            lines = list(logs_data.get('logs') or [])
            if lines:
                with _train_state_lock:
                    for line in lines:
                        _update_status_line(str(line))
                    remote_training_state['logs_offset'] = logs_offset + len(lines)

            if state in {'completed', 'failed', 'stopped', 'cancelled', 'done', 'success'}:
                with _train_state_lock:
                    training_status['running'] = False
                    training_status['pid'] = None
                    remote_training_state['active'] = False
                _update_status_line(f"[REMOTE] 云训练结束，状态: {state or 'unknown'}")
                break
        except Exception as e:
            _update_status_line(f"[REMOTE] 轮询异常: {e}")

        time.sleep(max(1.0, float(api_cfg.get('poll_interval_sec', 3.0))))


def _iter_pipe_lines(stdout_pipe, chunk_size: int = 4096):
    """按块读取 stdout，并将 \\r/\\n 统一视为行结束，保证 tqdm 刷新也能实时输出。"""
    if stdout_pipe is None:
        return
    pending = ''
    while True:
        raw = stdout_pipe.read(chunk_size)
        if not raw:
            break
        text = raw.decode('utf-8', errors='replace') if isinstance(raw, (bytes, bytearray)) else str(raw)
        if not text:
            continue
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        pending += normalized
        while '\n' in pending:
            line, pending = pending.split('\n', 1)
            yield line
    if pending:
        yield pending


def _kill_old_training():
    """
    强制终止旧训练进程树并等待完全退出 — 彻底版
    
    【关键改进】使用 _kill_process_tree 杀整个进程树，不再只杀 Popen 直接子进程。
    解决：Popen 启动的 python.exe → ultralytics 子进程逃逸 → GPU 显存不释放
    """
    global training_process
    if training_process is None:
        return
    old_pid = getattr(training_process, 'pid', None)

    killed = 0
    try:
        if old_pid and old_pid > 0:
            # 先尝试优雅终止
            try:
                if os.name == 'nt':
                    training_process.send_signal(signal.CTRL_C_EVENT)
                else:
                    training_process.send_signal(signal.SIGINT)
            except Exception:
                pass

            # 等待正常退出
            try:
                training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            # 还活着就暴力杀进程树
            if training_process.poll() is None:
                killed = _kill_process_tree(old_pid, force=False)

                # 等一下看是否都死了
                time.sleep(1)
                if _is_process_alive(old_pid):
                    _kill_process_tree(old_pid, force=True)
                    time.sleep(0.5)

            # 关闭 stdout pipe
            if training_process.stdout and not training_process.stdout.closed:
                try:
                    training_process.stdout.close()
                except Exception:
                    pass

        msg = f"旧训练进程已终止 (PID={old_pid}, 树中杀死={killed}个)"
        _update_status_line(msg)
    finally:
        with _train_state_lock:
            training_status['running'] = False
            training_status['pid'] = None
        training_process = None


def _run_training_process_safe(cmd):
    """
    安全版训练进程启动 — 在锁已持有的前提下执行
    
    此函数的调用者 (start_training) 必须已经：
      1. 持有 _train_thread_lock
      2. 已获取 _TRAIN_LOCK_FILE 的 OS 文件锁
      3. 已杀掉所有旧训练进程
      4. 已清理 GPU 资源
      
    本函数只负责：启动新子进程 → 实时内存监控 → 读取日志 → 进程结束 → 清理状态
    【新增】内置内存超限检测：一旦子进程树总内存超过阈值立即杀掉整棵进程树
    """
    global training_process

    # 内存监控配置（单位：字节）
    # 默认限制 12GB — 超过此值视为异常（正常蒸馏训练约 4-8GB）
    MAX_MEMORY_BYTES = int(os.environ.get('EDGE_TRAIN_MAX_MEM_GB', '12')) * 1024 * 1024 * 1024
    # 监控间隔：每隔多少行日志检查一次内存（平衡性能与响应速度）
    MEMORY_CHECK_INTERVAL = 50
    # 连续超限次数阈值（防止瞬时峰值误杀）
    MEMORY_OVERLIMIT_THRESHOLD = 3

    proc = None
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        # 子进程无 TTY 时强制行缓冲，避免 ultralytics / logging 长时间不刷到管道
        env['PYTHONUNBUFFERED'] = '1'
        # 训练子进程内将 TQDM 的 \\r 刷新改为按行输出，否则管道 readline 只能收到表头收不到每 batch 数值行
        env['EDGE_WEB_LOG'] = '1'

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        preexec_fn = None if os.name == 'nt' else os.setsid

        # Windows 上 text=True + bufsize=1 对管道往往仍块缓冲，改为二进制 bufsize=0 + readline，尽快收到每一行
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        training_process = proc
        with _train_state_lock:
            training_status['pid'] = proc.pid

        _update_status_line(f'[MEM_GUARD] 内存监控已启用 | 上限={MAX_MEMORY_BYTES // (1024**3)}GB | '
                           f'检查间隔={MEMORY_CHECK_INTERVAL}行 | 连续超限={MEMORY_OVERLIMIT_THRESHOLD}次')

        line_count = 0

        # 全程使用局部 proc，避免 stop_training 把全局 training_process 置 None 后出现 .wait() 竞态
        for raw_line in _iter_pipe_lines(proc.stdout):
            line_count += 1
            _update_status_line(raw_line)

            # ════════════════════════════════════════════
            # 定期内存安全检查
            # ════════════════════════════════════════════
            if line_count % MEMORY_CHECK_INTERVAL != 0:
                continue

            if not _check_and_enforce_memory_limit(
                pid=proc.pid,
                max_bytes=MAX_MEMORY_BYTES,
                threshold=MEMORY_OVERLIMIT_THRESHOLD,
            ):
                # _check_and_enforce_memory_limit 返回 False 表示已触发杀戮
                # 此时进程已被杀死或正在被杀，退出循环
                break

        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass

        # 等待进程完全退出（如果还没退的话）
        if proc.poll() is None:
            proc.wait()
            
    except Exception as e:
        _update_status_line(f"训练异常: {e}")
    finally:
        # 清理状态 + 释放文件锁
        with _train_state_lock:
            training_status['running'] = False
            training_status['pid'] = None
        training_process = None
        _release_training_lock()


# 内存监控模块内部状态
_mem_overlimit_count = 0



def _check_and_enforce_memory_limit(pid: int, max_bytes: int, threshold: int) -> bool:
    """
    检查指定 PID 及其子进程的总内存占用。
    
    Returns:
        True  = 内存正常，继续运行
        False = 已触发强制杀戮，调用者应停止读取日志循环
    """
    global _mem_overlimit_count
    
    total_rss = 0
    process_list = []
    
    try:
        import psutil
        
        try:
            parent = psutil.Process(pid)
            process_list.append(parent)
            process_list.extend(parent.children(recursive=True))
        except psutil.NoSuchProcess:
            return True  # 进程已死，不算异常
        
        # 累加所有进程的 RSS（常驻物理内存）
        for proc in process_list:
            try:
                mem_info = proc.memory_info()
                total_rss += mem_info.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        return True  # 无 psutil 时跳过监控
    
    except Exception:
        return True  # 其他异常跳过本次检查
    
    # 判断是否超限
    mb_used = total_rss / (1024 * 1024)
    mb_limit = max_bytes / (1024 * 1024)
    
    if total_rss > max_bytes:
        _mem_overlimit_count += 1
        
        if _mem_overlimit_count >= threshold:
            # 连续超限达到阈值 → 强制杀掉整棵进程树
            msg = (f"[MEM_ALERT] ⚠️ 内存严重超标！当前 {mb_used:.1f}MB > 限制 {mb_limit:.0f}MB "
                   f"(连续{_mem_overlimit_count}次)，强制终止训练进程树...")
            _update_status_line(msg)
            killed = _kill_process_tree(pid, force=True)
            time.sleep(1)  # 等待系统回收
            _update_status_line(f"[MEM_ALERT] 已强制终止 {killed} 个进程，释放 {mb_used:.0f}MB 内存")
            return False
        else:
            # 未达阈值但已超限 → 警告
            _update_status_line(
                f"[MEM_WARN] 内存偏高: {mb_used:.0f}MB / {mb_limit:.0f}MB "
                f"({_mem_overlimit_count}/{threshold})，继续观察..."
            )
            return True
    else:
        # 内存正常 → 重置计数器
        if _mem_overlimit_count > 0:
            _mem_overlimit_count = 0
        return True


# ==================== 路由定义 ====================

class SaveConfigRequest(BaseModel):
    name: str = 'distill_config.yaml'
    config: dict = Field(default_factory=dict)


class UploadConfigRequest(BaseModel):
    content: str


class DialogFilterItem(BaseModel):
    name: str = "All Files"
    patterns: list[str] = Field(default_factory=lambda: ["*.*"])


class DialogPickRequest(BaseModel):
    kind: str = "file"
    title: str = "选择路径"
    initial_path: str | None = None
    filters: list[DialogFilterItem] = Field(default_factory=list)


class AgentPatchPreviewRequest(BaseModel):
    patch: dict = Field(default_factory=dict)


class AgentPatchApplyRequest(BaseModel):
    approval_token: str | None = None
    token: str | None = None


class TrainStartRequest(BaseModel):
    config: str = 'distill_config.yaml'
    mode: str = 'distill'
    checkpoint: str | None = None
    allow_overwrite: bool = False


@api.get('/', response_class=HTMLResponse)
def index():
    """读取 HTML 并手动替换 Jinja2 url_for（避免模板查找问题）"""
    with open(str(TEMPLATE_FILE), 'r', encoding='utf-8') as f:
        html = f.read()

    def replace_static_url(m):
        fname = m.group(1)
        return '/static/' + fname

    html = re.sub(
        r"\{\{\s*url_for\(\s*'static'\s*,\s*filename\s*=\s*'([^']+)'\s*\)\s*\}\}",
        replace_static_url,
        html
    )
    html = re.sub(
        r'\{\{\s*url_for\(\s*"static"\s*,\s*filename\s*=\s*"([^"]+)"\s*\)\s*\}\}',
        replace_static_url,
        html
    )

    return HTMLResponse(content=html)


# ---- Config API ----

@api.get('/api/configs')
def get_configs():
    configs = []
    if CONFIG_DIR.exists():
        for path in sorted(CONFIG_DIR.iterdir()):
            if path.is_file() and path.suffix in {'.yaml', '.yml'}:
                configs.append(path.name)
    return {'status': 'ok', 'configs': configs}


@api.get('/api/config/{config_name}')
def get_config(config_name):
    config_path = CONFIG_DIR / config_name
    config = _load_yaml_file(config_path)
    if config is None:
        return _error(f'配置文件不存在: {config_name}', 404)
    return {'status': 'ok', 'config': config}


@api.get('/api/config/recent')
def get_recent_config():
    global last_saved_config
    if last_saved_config is not None:
        return {'status': 'ok', 'name': last_saved_config['name'], 'config': last_saved_config['config']}

    default_path = CONFIG_DIR / 'distill_config.yaml'
    config = _load_yaml_file(default_path) or {}
    return {'status': 'ok', 'name': 'distill_config.yaml', 'config': config}


@api.post('/api/config/save')
def save_config(payload: SaveConfigRequest):
    global last_saved_config
    name = payload.name
    config = payload.config
    if not isinstance(name, str) or not isinstance(config, dict):
        return _error('请求格式错误', 400)
    if not name.endswith(('.yaml', '.yml')):
        name = f'{name}.yaml'
    config_path = CONFIG_DIR / name
    _save_yaml_file(config_path, config)
    last_saved_config = {'name': name, 'config': config}
    return {'status': 'ok', 'message': f'配置已保存: {name}'}


@api.post('/api/config/upload')
def upload_config(payload: UploadConfigRequest):
    content = payload.content
    if not isinstance(content, str):
        return _error('请求格式错误', 400)

    try:
        config = expand_env_vars(yaml.safe_load(content) or {})
        if not isinstance(config, dict):
            return _error('配置文件必须包含顶层映射对象', 400)
        return {'status': 'ok', 'config': config}
    except yaml.YAMLError as exc:
        return _error(f'YAML 解析失败: {exc}', 400)


@api.post('/api/dialog/pick')
def pick_path_dialog(payload: DialogPickRequest):
    """打开本机原生文件/目录选择窗口，返回用户选择路径。"""
    kind = (payload.kind or "file").strip().lower()
    if kind not in {"file", "directory"}:
        return _error("kind 仅支持 file 或 directory", 400)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        return _error(f'当前环境不支持本机文件选择窗口: {exc}', 500)

    initial_dir = None
    initial_file = None
    if payload.initial_path:
        try:
            initial = Path(payload.initial_path).expanduser()
            if initial.exists():
                if initial.is_dir():
                    initial_dir = str(initial)
                else:
                    initial_dir = str(initial.parent)
                    initial_file = initial.name
            else:
                parent = initial.parent
                if parent and str(parent) not in {"", "."}:
                    initial_dir = str(parent)
                if initial.suffix:
                    initial_file = initial.name
        except Exception:
            initial_dir = None
            initial_file = None

    selected = ""
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
            root.update()
        except Exception:
            pass

        if kind == "directory":
            selected = filedialog.askdirectory(
                title=payload.title or "选择目录",
                initialdir=initial_dir
            ) or ""
        else:
            filetypes = []
            for item in payload.filters or []:
                pats = [str(p).strip() for p in (item.patterns or []) if str(p).strip()]
                if not pats:
                    continue
                filetypes.append((item.name or "文件", tuple(pats)))
            if not filetypes:
                filetypes = [("All Files", "*.*")]
            selected = filedialog.askopenfilename(
                title=payload.title or "选择文件",
                initialdir=initial_dir,
                initialfile=initial_file,
                filetypes=filetypes
            ) or ""
    except Exception as exc:
        return _error(f'打开文件选择窗口失败: {exc}', 500)
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    return {'status': 'ok', 'path': selected}


# ---- Agent: 用户审批后写入训练配置（与前端表单 / distill_config.yaml 对齐） ----

_AGENT_PATCH_TTL = 600.0
_agent_patch_store = {}


def _prune_agent_patch_store():
    now = time.time()
    for k, rec in list(_agent_patch_store.items()):
        if rec.get('expires', 0) < now:
            _agent_patch_store.pop(k, None)


def _deep_merge_shallow(dst: dict, src: dict) -> dict:
    out = copy.deepcopy(dst) if isinstance(dst, dict) else {}
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_shallow(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_distill_patch(base: dict, patch: dict) -> dict:
    allowed = frozenset({'distillation', 'training', 'output'})
    if not isinstance(patch, dict) or not patch:
        raise ValueError('patch 必须为非空对象')
    extra = set(patch.keys()) - allowed
    if extra:
        raise ValueError('不允许的顶层键: ' + ', '.join(sorted(extra)))
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for top in patch:
        sub = patch[top]
        if not isinstance(sub, dict):
            merged[top] = copy.deepcopy(sub)
            continue
        cur = merged.get(top)
        merged[top] = _deep_merge_shallow(cur if isinstance(cur, dict) else {}, sub)
    return merged


@api.get('/api/agent/config-schema')
def agent_config_schema():
    """供外部 LLM / Agent 与前端对齐：当前 distill 配置与允许改动的顶层分区。"""
    path = CONFIG_DIR / 'distill_config.yaml'
    cfg = _load_yaml_file(path) or {}
    return {
        'status': 'ok',
        'config_file': 'distill_config.yaml',
        'allowed_top_level': ['distillation', 'training', 'output'],
        'current': cfg,
        'hint': '外部 Agent 请在 JSON 中返回 patch 字段；用户在前端确认后调用 /api/agent/patch/preview 再 /api/agent/patch/apply。',
    }


@api.post('/api/agent/patch/preview')
def agent_patch_preview(payload: AgentPatchPreviewRequest):
    """合并 patch 到 distill_config.yaml 的内存预览，并签发短时审批令牌。"""
    global last_saved_config
    _prune_agent_patch_store()
    patch = payload.patch
    if not isinstance(patch, dict) or not patch:
        return _error('patch 必须为非空对象', 400)
    path = CONFIG_DIR / 'distill_config.yaml'
    base = _load_yaml_file(path) or {}
    try:
        merged = _merge_distill_patch(base, patch)
    except ValueError as e:
        return _error(str(e), 400)
    except Exception as e:
        return _error(str(e), 500)
    tok = str(uuid.uuid4())
    _agent_patch_store[tok] = {'merged': merged, 'expires': time.time() + _AGENT_PATCH_TTL}
    patch_yaml = yaml.dump(patch, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return {
        'status': 'ok',
        'approval_token': tok,
        'expires_in_sec': int(_AGENT_PATCH_TTL),
        'patch_yaml': patch_yaml,
        'merged_preview': merged,
    }


@api.post('/api/agent/patch/apply')
def agent_patch_apply(payload: AgentPatchApplyRequest):
    """凭审批令牌将预览中的合并结果写入 configs/distill_config.yaml。"""
    global last_saved_config
    _prune_agent_patch_store()
    tok = payload.approval_token or payload.token
    if not isinstance(tok, str) or not tok:
        return _error('缺少 approval_token', 400)
    rec = _agent_patch_store.pop(tok, None)
    if not rec or rec.get('expires', 0) < time.time():
        return _error('审批令牌无效或已过期，请重新预览', 400)
    merged = rec.get('merged')
    if not isinstance(merged, dict):
        return _error('内部数据损坏', 500)
    out_path = CONFIG_DIR / 'distill_config.yaml'
    try:
        _save_yaml_file(out_path, merged)
        last_saved_config = {'name': 'distill_config.yaml', 'config': merged}
        return {'status': 'ok', 'message': '已写入 configs/distill_config.yaml', 'config': merged}
    except Exception as e:
        return _error(str(e), 500)


# ---- Output Check ----

@api.get('/api/output/check')
def output_check(project: str = Query('runs/distill')):
    project = project or 'runs/distill'
    try:
        project_path = _resolve_project_path(project, allow_external=Path(project).is_absolute())
    except ValueError as e:
        return _error(str(e), 400)

    existing_names = []
    next_exp = 'exp1'
    candidate_roots = _candidate_output_roots(project_path)
    merged_names = set()
    for root in candidate_roots:
        if root.exists() and root.is_dir():
            for item in sorted(root.iterdir()):
                if item.is_dir():
                    merged_names.add(item.name)
    existing_names = sorted(merged_names)

    exp_numbers = []
    for name in existing_names:
        if name.startswith('exp'):
            try:
                exp_numbers.append(int(name[3:] or 0))
            except ValueError:
                pass
    next_exp = f'exp{max(exp_numbers) + 1}' if exp_numbers else 'exp1'
    return {
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'existing_names': existing_names,
        'next_exp_name': next_exp,
    }


# ---- Training API ----

def _cleanup_gpu_resources():
    """清理残留 GPU 显存资源"""
    try:
        import gc as _gc2
        import torch as _torch2
        _gc2.collect()
        if _torch2.cuda.is_available():
            _torch2.cuda.empty_cache()
            _torch2.cuda.reset_peak_memory_stats()
            if hasattr(_torch2.cuda, 'synchronize'):
                try:
                    _torch2.cuda.synchronize()
                except Exception:
                    pass
    except ImportError:
        pass
    except Exception:
        pass


def _wait_for_gpu_free(timeout_sec: float = 15.0) -> bool:
    """
    等待 GPU 显存释放到安全水平。
    
    通过 nvidia-smi 或 torch 检测 GPU 使用情况，
    在启动新训练前确保旧进程的显存已被回收。
    
    Returns:
        True=GPU 已就绪可使用，False=超时
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            import torch as _t
            if _t.cuda.is_available():
                try:
                    test_tensor = _t.zeros(1, device='cuda')
                    del test_tensor
                    _t.cuda.empty_cache()
                    return True
                except RuntimeError as _e:
                    if 'out of memory' in str(_e).lower():
                        _update_status_line(f'[GPU] 等待显存释放中... ({int(deadline - time.time())}s)')
                        time.sleep(2)
                        continue
            return True
        except ImportError:
            return True
        except Exception:
            time.sleep(2)
    return False


@api.post('/api/train/start')
def start_training(payload: TrainStartRequest):
    """
    启动训练 — 五层防护，坚决杜绝双进程同时运行
    
    ═══════ 防护层次（由外到内）══════
      Layer-0: 线程级互斥锁     → 防 Flask 多线程并发穿透
      Layer-1: 残留进程扫描     → 防重启后丢失引用的僵尸进程
      Layer-2: 内存状态检查     → 防 running=True 的已知进程逃逸  
      Layer-3: OS 级文件锁       → 防多实例/多 Web 进程并发（内核原子保证）
      Layer-4: GPU 安全等待     → 确保显存真正释放后才启动
    """
    global training_status, training_process

    config_name = payload.config
    mode = payload.mode
    checkpoint = payload.checkpoint
    allow_overwrite = bool(payload.allow_overwrite)

    if mode not in {'distill', 'resume'}:
        return _error(f'不支持的训练模式: {mode}', 400)

    # ════════════════════════════════════════════════════════
    # Layer-0：线程级互斥锁 — 防止 Flask 多线程并发穿透
    # ════════════════════════════════════════════════════════
    acquired = _train_thread_lock.acquire(blocking=False)
    if not acquired:
        return _error('系统繁忙：另一个请求正在处理中，请稍后再试', 503)

    try:
        # ═══════════════════════════════════════════════════
        # Layer-1：扫描残留/僵尸训练进程（Web 重启后丢失引用的场景）
        # ═══════════════════════════════════════════════════
        our_pid = os.getpid()
        stale = _scan_and_kill_stale_training_processes(exclude_pid=our_pid)
        if stale['found'] > 0:
            _update_status_line(f"[GUARD] 发现 {stale['found']} 个残留训练进程，已清理: {stale['details']}")
            time.sleep(3)

        # ═══════════════════════════════════════════════════
        # Layer-2：杀掉内存中已知的旧训练进程（resume 模式允许覆盖）
        # ═══════════════════════════════════════════════════
        with _train_state_lock:
            busy = training_status['running']
        if busy or training_process is not None:
            if mode == 'resume':
                _update_status_line('[RESUME] 检测到旧训练进程，正在终止...')
                _kill_old_training()
                time.sleep(3)
                _cleanup_gpu_resources()
                time.sleep(1)
            else:
                return _error('已有训练任务在运行中，请先停止或等待完成', 400)

        # ═══════════════════════════════════════════════════
        # Layer-3：获取 OS 级别文件排他锁（内核原子保证，非模拟！）
        # ═══════════════════════════════════════════════════
        lock_ok = _acquire_training_lock()
        if not lock_ok:
            err_msg = '训练互斥锁被占用'
            try:
                if _TRAIN_LOCK_FILE.exists():
                    content = _TRAIN_LOCK_FILE.read_text().strip().split('\n')
                    holder_pid = int(content[0]) if content else -1
                    if holder_pid > 0:
                        alive = _is_process_alive(holder_pid)
                        if not alive:
                            _release_os_file_lock()
                            _TRAIN_LOCK_FILE.unlink(missing_ok=True)
                            lock_ok = _acquire_training_lock()
                            if lock_ok:
                                _update_status_line(f'[GUARD] 已清除僵尸锁 (原PID={holder_pid})，重新获取成功')
                        else:
                            err_msg = f'训练进程 (PID={holder_pid}) 仍在运行中'
                    else:
                        err_msg = '训练锁文件损坏'
            except Exception:
                pass
            
            if not lock_ok:
                return _error(err_msg, 400)

        # ═══════════════════════════════════════════════════
        # Layer-4：二次验证 + GPU 安全等待
        # ═══════════════════════════════════════════════════
        
        # 二次验证：再次扫描残留（防御性编程）
        recheck_stale = _scan_and_kill_stale_training_processes(exclude_pid=our_pid)
        if recheck_stale['found'] > 0:
            _update_status_line(f'[GUARD] 二次扫描发现 {recheck_stale["found"]} 个漏网进程，已清理')
            time.sleep(2)

        # 验证配置文件
        config_path = CONFIG_DIR / config_name
        if not config_path.exists():
            _release_training_lock()
            return _error(f'配置文件不存在: {config_name}', 404)

        cfg = _load_yaml_file(config_path) or {}
        train_cfg = dict(cfg.get('training', {}) or {})
        compute_provider = _normalize_compute_provider(train_cfg.get('compute_provider'))
        if compute_provider == 'remote_api':
            try:
                api_cfg = _build_cloud_api_config(train_cfg)
            except ValueError as e:
                return _error(str(e), 400)
            request_payload = {
                'config': cfg,
                'mode': mode,
                'checkpoint': checkpoint,
                'allow_overwrite': allow_overwrite,
            }
            try:
                dataset_result = _resolve_dataset_via_api(train_cfg, cfg, mode, checkpoint)
                if isinstance(dataset_result, dict):
                    dataset_yaml = str(dataset_result.get('data_yaml', '') or '').strip()
                    dataset_id = str(dataset_result.get('dataset_id', '') or '').strip()
                    if dataset_yaml:
                        request_payload['config'] = copy.deepcopy(cfg)
                        request_payload['config'].setdefault('training', {})
                        request_payload['config']['training']['data_yaml'] = dataset_yaml
                    request_payload['dataset'] = dataset_result
                    _update_status_line(
                        f"[REMOTE] 数据集API已解析: "
                        f"{dataset_yaml or dataset_id or 'unknown'}"
                    )
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', errors='replace')
                except Exception:
                    body = ''
                return _error(f'数据集 API 调用失败: HTTP {e.code} {body}', 502)
            except Exception as e:
                return _error(f'数据集 API 调用失败: {e}', 502)
            try:
                submit_result = _http_json_request('POST', api_cfg['submit_url'], payload=request_payload, headers=api_cfg['headers'])
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode('utf-8', errors='replace')
                except Exception:
                    body = ''
                return _error(f'云训练提交失败: HTTP {e.code} {body}', 502)
            except Exception as e:
                return _error(f'云训练提交失败: {e}', 502)

            job_id = str(submit_result.get('job_id', '') or submit_result.get('id', '') or '').strip()
            if not job_id:
                return _error('云训练接口未返回 job_id', 502)

            with _train_state_lock:
                training_status.update({
                    'running': True,
                    'pid': None,
                    'config': config_name,
                    'mode': mode,
                    'start_time': time.time(),
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'logs': [f"[REMOTE] 已提交云训练任务: job_id={job_id}"],
                })
                remote_training_state.update({
                    'active': True,
                    'job_id': job_id,
                    'api_base_url': api_cfg['base_url'],
                    'logs_offset': 0,
                })

            threading.Thread(target=_remote_polling_loop, args=(api_cfg, job_id), daemon=True).start()
            return {'status': 'ok', 'message': '云训练任务已提交', 'remote': True, 'job_id': job_id}
        allow_external_project = compute_provider in {'autodl', 'colab'}
        output_cfg = dict(cfg.get('output', {}) or {})
        target_project = str(output_cfg.get('project', 'runs/distill') or 'runs/distill')
        target_name = str(output_cfg.get('name', 'exp') or 'exp').strip()
        try:
            project_path = _resolve_project_path(target_project, allow_external=allow_external_project)
        except ValueError:
            _release_training_lock()
            return _error(f'输出目录非法: {target_project}', 400)
        candidate_roots = _candidate_output_roots(project_path)
        target_run_paths = [(root / target_name).resolve() for root in candidate_roots]
        existing_target_path = next((p for p in target_run_paths if p.exists()), None)

        if mode != 'resume' and target_name and existing_target_path is not None and not allow_overwrite:
            _release_training_lock()
            conflict_project = target_project
            try:
                conflict_project = str(existing_target_path.parent.relative_to(BASE_DIR))
            except Exception:
                pass
            return JSONResponse(status_code=409, content={
                'error': f'输出目录已存在：{conflict_project}/{target_name}',
                'requires_confirmation': True,
                'project': conflict_project,
                'name': target_name,
            })

        # 构建命令：蒸馏 / 断点续训 共用同一子进程入口，避免双栈逻辑与重复加载
        cmd = [sys.executable, '-u', '-m', 'scripts.train_with_distill', '--config', str(config_path)]
        if mode == 'resume':
            if checkpoint:
                checkpoint_path = Path(checkpoint)
                if not checkpoint_path.is_absolute():
                    checkpoint_path = (BASE_DIR / checkpoint_path).resolve()
                cmd.extend(['--resume', str(checkpoint_path)])
            else:
                cmd.append('--resume')
                cmd.append('auto')
        elif allow_overwrite:
            cmd.append('--allow-overwrite')

        # GPU 安全等待：确保显存真正释放后再启动新训练
        _cleanup_gpu_resources()
        gpu_ready = _wait_for_gpu_free(timeout_sec=20.0)
        if not gpu_ready:
            _release_training_lock()
            return JSONResponse(status_code=503, content={
                'error': 'GPU 显存未能在超时时间内释放，可能仍有残留训练进程',
                'hint': '请手动结束占用 GPU 的进程后重试',
            })

        # 最终步骤：更新状态并启动新训练线程
        with _train_state_lock:
            training_status.update({
                'running': True,
                'pid': None,
                'config': config_name,
                'mode': mode,
                'start_time': time.time(),
                'current_epoch': 0,
                'total_epochs': 0,
                'logs': [f"{'[RESUME] 断点续训' if mode == 'resume' else '[TRAIN] 训练'} 已启动..."],
            })

        thread = threading.Thread(target=_run_training_process_safe, args=(cmd,), daemon=True)
        thread.start()
        return {'status': 'ok', 'message': f"{'断点续训' if mode == 'resume' else '训练'}已启动"}

    finally:
        # 【关键】线程锁在 finally 中释放。
        # 文件锁仍由子线程 (_run_training_process_safe) 持有直到训练结束。
        _train_thread_lock.release()


@api.post('/api/train/stop')
def stop_training():
    """
    停止训练进程 — 彻底版
    
    修复：解决多线程竞态条件导致进程残留的问题
    1. 先关闭 stdout pipe → 让后台日志循环立即退出（不再阻塞）
    2. 本地引用 proc → 防止竞态条件下 training_process 被置为 None
    3. 分阶段杀戮：SIGINT → 优雅 terminate → 暴力 kill
    4. 多次验证进程存活状态
    5. 清理 GPU 显存资源
    """
    global training_process, training_status
    with _train_state_lock:
        running = training_status['running']
    if not running:
        return {'warning': '没有运行中的训练任务'}

    # 【关键】立即保存本地引用，防止与后台线程产生竞态
    proc = training_process
    old_pid = getattr(proc, 'pid', None) if proc else None

    if remote_training_state.get('active'):
        try:
            cfg = _load_yaml_file(CONFIG_DIR / 'distill_config.yaml') or {}
            train_cfg = dict(cfg.get('training', {}) or {})
            api_cfg = _build_cloud_api_config(train_cfg)
            job_id = str(remote_training_state.get('job_id') or '')
            if job_id:
                _http_json_request('POST', api_cfg['stop_url'], payload={'job_id': job_id}, headers=api_cfg['headers'])
                _update_status_line(f'[REMOTE] 已请求停止云任务: {job_id}')
        except Exception as e:
            _update_status_line(f'[REMOTE] 停止云任务失败: {e}')
        with _train_state_lock:
            remote_training_state['active'] = False
            remote_training_state['job_id'] = ''

    if proc and old_pid and old_pid > 0:
        # ═══ Step 1: 先关闭 stdout pipe → 让 _run_training_process_safe 的 for 循环立即退出 ═══
        if proc.stdout and not proc.stdout.closed:
            try:
                proc.stdout.close()
            except Exception:
                pass

        # ═══ Step 2: 发送中断信号（请求优雅退出）═══
        try:
            if os.name == 'nt':
                proc.send_signal(signal.CTRL_C_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass

        # ═══ Step 3: 等待进程自行退出（最多 8 秒）═══
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pass

        # ═══ Step 4: 还活着 → 优雅杀进程树 ═══
        if _is_process_alive(old_pid):
            _kill_process_tree(old_pid, force=False)
            time.sleep(1.5)

        # ═══ Step 5: 仍活着 → 暴力杀进程树 ═══
        if _is_process_alive(old_pid):
            _kill_process_tree(old_pid, force=True)
            time.sleep(1)

        # ═══ Step 6: 最终验证 + 告警 ═══
        if _is_process_alive(old_pid):
            warn_msg = f"[STOP_WARN] 进程 PID={old_pid} 停止失败！请手动结束该进程以释放 GPU 显存"
            _update_status_line(warn_msg)

    # ═══ Step 7: 先写入停止日志再清 running，避免 SSE 在 running=False 瞬间关流漏掉本行 ═══
    _update_status_line('训练已被用户停止')

    # ═══ Step 8: 统一更新全局状态（无论是否需要杀进程都执行）═══
    with _train_state_lock:
        training_status['running'] = False
        training_status['pid'] = None
    training_process = None

    # ═══ Step 9: 释放锁 + 清理 GPU 显存 ═══
    _release_training_lock()
    _cleanup_gpu_resources()

    return {'status': 'ok', 'message': '训练已停止'}


@api.get('/api/train/status')
def get_training_status():
    with _train_state_lock:
        snap = {k: (list(v) if k == 'logs' and isinstance(v, list) else v) for k, v in training_status.items()}
    snap['log_count'] = len(snap.get('logs') or [])
    return {'status': 'ok', **snap}


@api.get('/api/train/resume_candidates')
def get_resume_candidates(project: str = Query('runs/distill')):
    project = project or 'runs/distill'
    try:
        project_path = _resolve_project_path(project, allow_external=Path(project).is_absolute())
    except ValueError as e:
        return _error(str(e), 400)
    candidates = _list_resume_candidates(project_path)
    return {
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'candidates': candidates
    }


@api.get('/api/train/logs')
def get_training_logs(offset: int = Query(0), limit: int = Query(100)):
    offset = max(0, int(offset))
    limit = min(5000, max(1, int(limit)))
    with _train_state_lock:
        logs = list(training_status['logs'])
    total = len(logs)
    if offset > total:
        offset = total
    return {'status': 'ok', 'logs': logs[offset:offset + limit], 'total': total, 'offset': offset, 'limit': limit}


@api.get('/api/train/logs/download')
def download_training_logs():
    """导出当前内存中的训练日志为纯文本（与 /api/train/logs 同源）。"""
    with _train_state_lock:
        text = '\n'.join(training_status.get('logs') or [])
    return PlainTextResponse(
        text + ('\n' if text and not text.endswith('\n') else ''),
        headers={'Content-Disposition': 'attachment; filename=training_log.txt'},
    )


@api.get('/api/train/logs/stream')
def stream_training_logs(offset: int = Query(0)):
    offset = max(0, int(offset))

    def generate():
        # 基于 offset 线性追赶日志，保证重连后不会漏行也不会重复刷整屏。
        next_idx = offset
        while True:
            batch = []
            batch_start = next_idx
            with _train_state_lock:
                buf = training_status['logs']
                n = len(buf)
                running = training_status['running']
                if next_idx < n:
                    batch = list(buf[next_idx:n])
                    batch_start = next_idx
                    next_idx = n
                elif running:
                    _train_log_cond.wait(timeout=2.0)
                else:
                    break

            if batch:
                for idx, line in enumerate(batch, start=batch_start + 1):
                    payload = {'line': line, 'idx': idx}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                continue

            with _train_state_lock:
                is_done = (not training_status['running']) and next_idx >= len(training_status['logs'])
            if is_done:
                break

            if not batch:
                yield ': keepalive\n\n'
        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'}
    )


# ---- Metrics API (完整自包含实现) ----

@api.get('/api/metrics')
def get_metrics(source: str = Query('')):
    runs_dir = BASE_DIR / 'runs'
    base_resolved = Path(BASE_DIR).resolve()
    metrics_data = []
    if runs_dir.exists():
        for root, dirs, files in os.walk(str(runs_dir)):
            for file_name in files:
                if file_name != 'results.csv':
                    continue
                result_path = Path(root) / file_name
                run_dir = result_path.parent
                try:
                    display_name = f"{run_dir.name} @ {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
                    rel_dir = str(run_dir.resolve().relative_to(base_resolved))
                    rel_path = str(result_path.resolve().relative_to(base_resolved))
                except (ValueError, OSError):
                    continue
                entry = {
                    "name": run_dir.name,
                    "display_name": display_name,
                    "dir": rel_dir,
                    "has_results": True,
                    "path": rel_path
                }
                try:
                    columns, rows = _load_csv_summary(result_path)
                    entry["columns"] = columns
                    entry["rows"] = len(rows)
                except Exception:
                    entry["columns"] = []
                    entry["rows"] = 0
                metrics_data.append(entry)
        try:
            metrics_data.sort(key=lambda item: os.path.getmtime(str(base_resolved / item['dir'])), reverse=True)
        except Exception:
            pass

    selected_path = source.strip()
    selected_data = None
    if selected_path:
        try:
            target = (BASE_DIR / selected_path).resolve()
            if target.exists() and target.is_file() and target.suffix == '.csv' and str(target).startswith(str(base_resolved)):
                columns, rows = _load_csv_summary(target)
                if rows:
                    chart_series = _build_metric_series(rows, columns, target.parent)
                    summary_metrics = {}
                    for key, col, better in [
                        ('box_loss', 'train/box_loss', 'lower'),
                        ('cls_loss', 'train/cls_loss', 'lower'),
                        ('dfl_loss', 'train/dfl_loss', 'lower'),
                        ('map50', 'metrics/mAP50(B)', 'higher'),
                        ('map50_95', 'metrics/mAP50-95(B)', 'higher'),
                        ('precision', 'metrics/precision(B)', 'higher'),
                        ('recall', 'metrics/recall(B)', 'higher'),
                    ]:
                        try:
                            s = _summarize_series(rows, col, better=better)
                            if s is not None:
                                summary_metrics[key] = s
                        except Exception:
                            pass
                    total_time = None
                    for row in reversed(rows):
                        total_time = _as_float(row.get('time'))
                        if total_time is not None:
                            break

                    run_stats = _estimate_run_stats(target.parent)
                    ov_map50 = '--'
                    if summary_metrics.get('map50'):
                        try:
                            ov_map50 = f"{(summary_metrics['map50']['best'] * 100):.2f}%"
                        except Exception:
                            pass
                    ov_time = '--'
                    if total_time is not None:
                        ov_time = f"{int(total_time // 60)}m {int(total_time % 60)}s"
                    selected_data = {
                        'source': selected_path,
                        'columns': columns,
                        'rows': len(rows),
                        'chart_series': chart_series,
                        'summary_metrics': summary_metrics,
                        'overview_stats': {
                            'ov-map50': ov_map50,
                            **run_stats,
                            'ov-time': ov_time
                        }
                    }
        except Exception as e:
            import traceback
            traceback.print_exc()
            selected_data = {'error': str(e)}

    response = {
        "status": "ok",
        "csv_metrics": metrics_data
    }
    if selected_data:
        response.update(selected_data)
    return response


if __name__ == '__main__':
    _port = int(os.environ.get('EDGE_BACKEND_PORT', os.environ.get('EDGE_FLASK_PORT', '5000')))
    print('=' * 60)
    print('  EdgeDistillDet Local UI')
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  Template : {TEMPLATE_FILE} (exists: {TEMPLATE_FILE.exists()})")
    print(f"  http://localhost:{_port}")
    print('=' * 60)

    if not TEMPLATE_FILE.exists():
        print(f"[FATAL] Template file not found: {TEMPLATE_FILE}")
        sys.exit(1)

    uvicorn.run(api, host='0.0.0.0', port=_port, log_level='info')
