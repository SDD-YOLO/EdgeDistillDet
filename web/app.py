"""
EdgeDistillDet Local UI
=======================
本地端点的自包含 Web UI，不依赖外部 server 模块。

使用方法: python web/app.py
"""

import csv
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request

# ==================== 路径配置（必须在项目内 import 之前） ====================
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
sys.path.insert(0, str(BASE_DIR))
CONFIG_DIR = BASE_DIR / 'configs'
TEMPLATE_FILE = WEB_DIR / 'templates' / 'index.html'
STATIC_DIR = WEB_DIR / 'static'

from utils import expand_env_vars
from flask_cors import CORS

# ==================== 训练互斥锁（跨平台） ====================
_TRAIN_LOCK_FILE = BASE_DIR / '.training.lock'


class _TrainingLock:
    """基于文件 + PID 检查的跨平台训练互斥锁"""

    def __init__(self, lock_path: Path):
        self._path = lock_path
        self._held = False

    def acquire(self) -> bool:
        """
        尝试获取锁。返回 True 表示成功。
        原理：原子性地创建/更新锁文件，写入当前 PID。
        如果文件已存在且对应进程存活，返回 False。
        """
        try:
            pid = os.getpid()
            ts = datetime.now().isoformat()
            content = f"{pid}\n{ts}\n"

            # 原子性写入：先写临时文件，再 rename（跨平台原子操作）
            tmp_path = self._path.with_suffix('.tmp')
            for _ in range(3):  # 重试 3 次
                try:
                    # 检查是否有活跃进程持有锁
                    if self._path.exists():
                        existing = self._path.read_text().strip().split('\n')
                        old_pid = int(existing[0]) if existing else -1
                        if self._is_process_alive(old_pid):
                            return False  # 进程仍存活，获取失败
                    # 写入新锁
                    tmp_path.write_text(content)
                    tmp_path.replace(self._path)  # 原子替换
                    self._held = True
                    return True
                except FileExistsError:
                    time.sleep(0.1)
                except Exception:
                    time.sleep(0.1)
            return False
        except Exception:
            return False

    def release(self):
        """释放锁"""
        if self._held or self._path.exists():
            self._held = False
            try:
                self._path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                tmp = self._path.with_suffix('.tmp')
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """检查进程是否存活"""
        if pid <= 0:
            return False
        try:
            import psutil as _psutil
            return _psutil.pid_exists(pid)
        except ImportError:
            # 无 psutil 时用简单方法
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False


_train_lock = _TrainingLock(_TRAIN_LOCK_FILE)


def _acquire_training_lock() -> bool:
    return _train_lock.acquire()


def _release_training_lock():
    _train_lock.release()

try:
    from utils.edge_profiler import EdgeProfiler
except Exception:
    EdgeProfiler = None


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _resolve_project_path(project: str) -> Path:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = (BASE_DIR / project_path).resolve()
    else:
        project_path = project_path.resolve()
    if str(project_path).startswith(str(Path(BASE_DIR).resolve())) is False and str(Path(BASE_DIR).resolve()) not in str(project_path):
        raise ValueError('项目目录必须在仓库根目录下')
    return project_path


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


# ==================== Flask App ====================

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / 'templates'),
    static_folder=str(STATIC_DIR),
    static_url_path='/static'
)
CORS(app)

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

training_process = None
log_queue = queue.Queue(maxsize=500)
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
last_saved_config = None


def _update_status_line(line: str):
    if not line:
        return
    clean_line = line.rstrip('\r\n')

    # ═══ 日志过滤：去掉 ultralytics 噪音和重复内容 ═══
    # 1. 空行/纯空白
    if not clean_line.strip():
        return
    # 2. ultralytics 进度条（ANSI 转义序列 + 进度百分比行）
    #    格式: "     3/10      3.98G ..." 或含 \x1b[ 的行
    if '\x1b[' in clean_line and ('Epoch' not in clean_line):
        return
    # 3. ultralytics 版本提示
    if 'New https://pypi.org/project/ultralytics/' in clean_line:
        return
    # 4. AMP 检查通过（无意义）
    if 'AMP: checks passed' in clean_line or 'AMP: running Automatic Mixed Precision' in clean_line:
        return
    # 5. optimizer 自动选择信息（冗长）
    if clean_line.startswith("optimizer:"):
        return
    # 6. 数据集扫描缓存信息（太详细）
    if 'Scanning' in clean_line and '.cache...' in clean_line:
        return
    # 7. 模型摘要（太长，保留精简版）
    if 'Model summary' in clean_line and 'GFLOPs' in clean_line:
        return
    # 8. Transferred 权重信息
    if clean_line.startswith('Transferred'):
        return
    # 9. Freezing layer 信息
    if clean_line.startswith('Freezing'):
        return
    # 10. 验证进度行（"all      128        929..."）
    stripped = clean_line.strip()
    if re.match(r'^\s*all\s+\d+\s+\d+', stripped) and 'Class' not in clean_line:
        return
    # 11. Speed 行（每张图片速度，噪音太多）
    if stripped.startswith('Speed:'):
        return
    # 12. Results saved 行
    if stripped.startswith('Results saved to'):
        return
    # 13. Optimizer stripped 行
    if stripped.startswith('Optimizer stripped from'):
        return
    # 14. Validating 行
    if stripped.startswith('Validating ') and 'Model summary' not in clean_line:
        return

    # ═══ 日志去重：连续相同内容只保留一条 ═══
    logs = training_status['logs']
    if logs and logs[-1] == clean_line:
        return  # 连续重复，跳过

    training_status['logs'].append(clean_line)
    try:
        log_queue.put_nowait(clean_line)
    except queue.Full:
        pass
    if len(training_status['logs']) > 500:
        training_status['logs'] = training_status['logs'][-200:]

    match = re.search(r"Epoch\s*(\d+)\s*/\s*(\d+)", clean_line, re.IGNORECASE)
    if match:
        training_status['current_epoch'] = int(match.group(1))
        training_status['total_epochs'] = int(match.group(2))


def _kill_old_training():
    """强制终止旧训练进程并等待其完全退出，防止 GPU 显存重叠导致 OOM"""
    global training_process
    if training_process is None:
        return
    old_pid = getattr(training_process, 'pid', None)
    try:
        # 1. 发送终止信号
        if os.name == 'nt':
            training_process.send_signal(signal.CTRL_C_EVENT)
        else:
            training_process.send_signal(signal.SIGINT)

        # 2. 给进程正常退出的时间窗口（5秒）
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

        # 3. 如果还在跑，暴力杀掉（kill -9 等效）
        if training_process.poll() is None:
            training_process.terminate()
            try:
                training_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                training_process.kill()
                try:
                    training_process.wait(timeout=2)
                except Exception:
                    pass

        # 4. 关闭 stdout pipe 防止资源泄漏
        if training_process.stdout and not training_process.stdout.closed:
            training_process.stdout.close()

        logger_msg = f"旧训练进程已终止 (PID={old_pid})"
        try:
            log_queue.put_nowait(logger_msg)
        except queue.Full:
            pass
    except Exception as e:
        pass
    finally:
        training_status['running'] = False
        training_status['pid'] = None
        training_process = None


def _run_training_process(cmd):
    global training_process
    try:
        # 【OOM修复】启动新训练前，必须先杀死旧进程并等待 GPU 资源释放！
        _kill_old_training()

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        preexec_fn = None if os.name == 'nt' else os.setsid

        training_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        training_status['pid'] = training_process.pid

        for raw_line in training_process.stdout:
            _update_status_line(raw_line)

        training_process.wait()
    except Exception as e:
        _update_status_line(f"训练异常: {e}")
    finally:
        # 清理状态 + 释放锁
        training_status['running'] = False
        training_status['pid'] = None
        training_process = None
        _release_training_lock()


# ==================== 路由定义 ====================

@app.route('/')
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

    return html


# ---- Config API ----

@app.route('/api/configs', methods=['GET'])
def get_configs():
    configs = []
    if CONFIG_DIR.exists():
        for path in sorted(CONFIG_DIR.iterdir()):
            if path.is_file() and path.suffix in {'.yaml', '.yml'}:
                configs.append(path.name)
    return jsonify({'status': 'ok', 'configs': configs})


@app.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    config_path = CONFIG_DIR / config_name
    config = _load_yaml_file(config_path)
    if config is None:
        return jsonify({'error': f'配置文件不存在: {config_name}'}), 404
    return jsonify({'status': 'ok', 'config': config})


@app.route('/api/config/recent', methods=['GET'])
def get_recent_config():
    global last_saved_config
    if last_saved_config is not None:
        return jsonify({'status': 'ok', 'name': last_saved_config['name'], 'config': last_saved_config['config']})

    default_path = CONFIG_DIR / 'distill_config.yaml'
    config = _load_yaml_file(default_path) or {}
    return jsonify({'status': 'ok', 'name': 'distill_config.yaml', 'config': config})


@app.route('/api/config/save', methods=['POST'])
def save_config():
    global last_saved_config
    payload = request.json or {}
    name = payload.get('name', 'distill_config.yaml')
    config = payload.get('config', {})
    if not isinstance(name, str) or not isinstance(config, dict):
        return jsonify({'error': '请求格式错误'}), 400
    if not name.endswith(('.yaml', '.yml')):
        name = f'{name}.yaml'
    config_path = CONFIG_DIR / name
    _save_yaml_file(config_path, config)
    last_saved_config = {'name': name, 'config': config}
    return jsonify({'status': 'ok', 'message': f'配置已保存: {name}'})


@app.route('/api/config/upload', methods=['POST'])
def upload_config():
    payload = request.json or {}
    content = payload.get('content')
    if not isinstance(content, str):
        return jsonify({'error': '请求格式错误'}), 400

    try:
        config = expand_env_vars(yaml.safe_load(content) or {})
        if not isinstance(config, dict):
            return jsonify({'error': '配置文件必须包含顶层映射对象'}), 400
        return jsonify({'status': 'ok', 'config': config})
    except yaml.YAMLError as exc:
        return jsonify({'error': f'YAML 解析失败: {exc}'}), 400


# ---- Output Check ----

@app.route('/api/output/check', methods=['GET'])
def output_check():
    project = request.args.get('project', 'runs/distill') or 'runs/distill'
    try:
        project_path = _resolve_project_path(project)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    existing_names = []
    next_exp = 'exp1'
    if project_path.exists() and project_path.is_dir():
        for item in sorted(project_path.iterdir()):
            if item.is_dir():
                existing_names.append(item.name)
        exp_numbers = [-1]
        for name in existing_names:
            if name.startswith('exp'):
                try:
                    exp_numbers.append(int(name[3:] or 0))
                except ValueError:
                    pass
        next_exp = f'exp{max(exp_numbers) + 1}' if exp_numbers else 'exp1'

    return jsonify({
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'existing_names': existing_names,
        'next_exp_name': next_exp,
    })


# ---- Training API ----

@app.route('/api/train/start', methods=['POST'])
def start_training():
    global training_status

    payload = request.json or {}
    config_name = payload.get('config', 'distill_config.yaml')
    mode = payload.get('mode', 'distill')
    checkpoint = payload.get('checkpoint')

    if mode not in {'distill', 'resume'}:
        return jsonify({'error': f'不支持的训练模式: {mode}'}), 400

    # 【OOM修复】resume 模式下，即使旧进程显示为 running 也允许启动（会先杀掉旧进程）
    if mode == 'resume' and training_status['running']:
        _update_status_line('[RESUME] 检测到旧训练进程正在运行，将先终止再启动续训...')
        _kill_old_training()
        # 等待 GPU 资源释放
        time.sleep(2)
    elif training_status['running']:
        return jsonify({'error': '已有训练任务在运行中，请先停止或等待完成'}), 400

    # 【关键】文件锁双重保护，防止竞态条件导致重复启动
    if not _acquire_training_lock():
        # 检查是否是僵尸锁（原进程已死但锁文件残留）
        try:
            if _TRAIN_LOCK_FILE.exists():
                content = _TRAIN_LOCK_FILE.read_text().strip().split('\n')
                old_pid = int(content[0]) if content else -1
                # 检查旧进程是否存在
                import psutil as _psutil
                if not _psutil.pid_exists(old_pid):
                    _release_training_lock()  # 清除僵尸锁
                else:
                    return jsonify({'error': f'训练进程 (PID={old_pid}) 仍在运行中'}), 400
            else:
                return jsonify({'error': '无法获取训练锁，可能有其他实例在运行'}), 400
        except Exception:
            return jsonify({'error': '训练任务正在运行中（文件锁被占用）'}), 400

    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        _release_training_lock()
        return jsonify({'error': f'配置文件不存在: {config_name}'}), 404

    cmd = [sys.executable]
    if mode == 'distill':
        cmd.extend(['-u', '-m', 'scripts.train_with_distill', '--config', str(config_path)])
    else:
        main_py = BASE_DIR / 'main.py'
        cmd.extend([str(main_py), 'train', '--config', str(config_path)])
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = (BASE_DIR / checkpoint_path).resolve()
            cmd.extend(['--resume', str(checkpoint_path)])
        else:
            cmd.append('--resume')

    training_status.update({
        'running': True,
        'pid': None,
        'config': config_name,
        'mode': mode,
        'start_time': time.time(),
        'current_epoch': 0,
        'total_epochs': 0,
        'logs': ['正在启动训练...'],
    })

    thread = threading.Thread(target=_run_training_process, args=(cmd,), daemon=True)
    thread.start()
    return jsonify({'status': 'ok', 'message': '训练已启动'})


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_process, training_status
    if not training_status['running']:
        return jsonify({'warning': '没有运行中的训练任务'})
    if training_process:
        try:
            if os.name == 'nt':
                training_process.send_signal(signal.CTRL_C_EVENT)
            else:
                training_process.send_signal(signal.SIGINT)
        except Exception:
            training_process.terminate()
        training_status['running'] = False
        training_process = None
        _release_training_lock()
        stop_msg = "训练已被用户停止"
        training_status['logs'].append(stop_msg)
        try:
            log_queue.put_nowait(stop_msg)
        except queue.Full:
            pass
    return jsonify({'status': 'ok', 'message': '训练已停止'})


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    return jsonify({'status': 'ok', **training_status, 'log_count': len(training_status['logs'])})


@app.route('/api/train/resume_candidates', methods=['GET'])
def get_resume_candidates():
    project = request.args.get('project', 'runs/distill') or 'runs/distill'
    try:
        project_path = _resolve_project_path(project)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    candidates = _list_resume_candidates(project_path)
    return jsonify({
        'status': 'ok',
        'project': str(project_path.relative_to(BASE_DIR)),
        'candidates': candidates
    })


@app.route('/api/train/logs', methods=['GET'])
def get_training_logs():
    offset = max(0, int(request.args.get('offset', 0)))
    limit = min(500, max(1, int(request.args.get('limit', 100))))
    logs = training_status['logs']
    total = len(logs)
    if offset > total:
        offset = total
    return jsonify({'status': 'ok', 'logs': logs[offset:offset + limit], 'total': total, 'offset': offset, 'limit': limit})


@app.route('/api/train/logs/stream')
def stream_training_logs():
    def generate():
        # 新连接时先从 logs 列表补全历史消息（仅一次），之后只从 queue 读取新增消息
        _sent_count = 0
        logs = training_status['logs']
        # 首次连接：补全已有日志
        while _sent_count < len(logs):
            line = logs[_sent_count]
            _sent_count += 1
            yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"

        # 之后只从 queue 消费新消息，避免与 logs 列表双重推送
        while True:
            if not training_status['running']:
                # 训练结束后再补一次可能遗漏的末尾日志
                remaining_logs = training_status['logs']
                while _sent_count < len(remaining_logs):
                    line = remaining_logs[_sent_count]
                    _sent_count += 1
                    yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
                break
            try:
                line = log_queue.get(timeout=2)
                _sent_count += 1
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ': keepalive\n\n'
        yield 'event: done\ndata: {}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'}
    )


# ---- Metrics API (完整自包含实现) ----

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
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

    selected_path = request.args.get('source', '').strip()
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
    return jsonify(response)


if __name__ == '__main__':
    print('=' * 60)
    print('  EdgeDistillDet Local UI')
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  Template : {TEMPLATE_FILE} (exists: {TEMPLATE_FILE.exists()})")
    print("  http://localhost:5000")
    print('=' * 60)

    if not TEMPLATE_FILE.exists():
        print(f"[FATAL] Template file not found: {TEMPLATE_FILE}")
        sys.exit(1)

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
