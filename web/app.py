"""
EdgeDistillDet Web UI - Flask Backend
=====================================
Material Design风格的本地训练管理与监控界面
"""

import os
import sys
import json
import re
import time
import yaml
import locale
import subprocess
import threading
import queue
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path


try:
    from utils.edge_profiler import EdgeProfiler
except Exception:
    EdgeProfiler = None

_model_overview_cache = {}

def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _resolve_model_path(run_dir: Path):
    args_file = run_dir / 'args.yaml'
    if not args_file.exists():
        return None
    try:
        payload = yaml.safe_load(args_file.read_text(encoding='utf-8'))
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
    """通过 ultralytics YOLO.info() 返回值获取 GFLOPs，用于 FPS 回退估算。"""
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

    cache_key = str(model_path)
    if cache_key in _model_overview_cache:
        return _model_overview_cache[cache_key]

    try:
        if EdgeProfiler is not None:
            profiler = EdgeProfiler(str(model_path), target_device='gpu')
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = profiler.run_full_profile()
            stats = {
                'ov-fps': f"{result.theoretical_fps_fp16:.0f}",
                'ov-params': f"{result.params_m:.1f}"
            }
        else:
            params_m = _estimate_model_params(model_path)
            gflops = _estimate_model_gflops(model_path)
            if gflops is None and params_m is not None:
                gflops = params_m * 2.0
            fps_str = '--'
            if gflops is not None and gflops > 0:
                gpu_tops_fp16 = 13.0
                efficiency = 0.6
                fps = (gpu_tops_fp16 * 1e12 * efficiency) / (gflops * 1e9)
                fps_str = f"{fps:.0f}"
            stats = {
                'ov-fps': fps_str if fps_str != '--' else 'N/A',
                'ov-params': f"{params_m:.1f}" if params_m is not None else '--'
            }
    except Exception:
        params_m = _estimate_model_params(model_path)
        stats = {
            'ov-fps': 'N/A',
            'ov-params': f"{params_m:.1f}" if params_m is not None else '--'
        }

    _model_overview_cache[cache_key] = stats
    return stats


def _parse_dataset_labels(run_dir: Path) -> list[str]:
    labels = []
    args_file = run_dir / 'args.yaml'
    if not args_file.exists():
        return labels

    try:
        payload = yaml.safe_load(args_file.read_text(encoding='utf-8'))
        data_value = payload.get('data')
        if not data_value:
            return labels

        data_path = Path(data_value)
        if not data_path.is_absolute():
            data_path = (run_dir / data_path).resolve()
        if not data_path.exists():
            data_path = (ROOT / data_value).resolve()
        if not data_path.exists():
            return labels

        data_payload = yaml.safe_load(data_path.read_text(encoding='utf-8'))
        names = data_payload.get('names') or data_payload.get('class_names') or data_payload.get('classes')
        if isinstance(names, dict):
            labels = [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])]
        elif isinstance(names, list):
            labels = [str(v) for v in names]
    except Exception:
        pass

    return labels


def _extract_class_performance(rows: list[dict], columns: list[str], class_labels: list[str] | None = None):
    if not columns or not rows:
        return None

    class_metrics: dict[int, dict[str, str]] = {}
    for column in columns:
        normalized = column.strip()
        match = re.search(r'(?i)(?:class|cls)[\s_/\\-]*(?P<index>\d+)', normalized)
        if not match:
            continue
        idx = int(match.group('index'))
        metric_key = None
        lower = normalized.lower()

        if 'precision' in lower:
            metric_key = 'precision'
        elif 'recall' in lower:
            metric_key = 'recall'
        elif 'map50-95' in lower or 'mAP50-95' in normalized:
            metric_key = 'map50_95'
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
    for idx in sorted(class_metrics.keys()):
        labels.append(class_labels[idx] if class_labels and idx < len(class_labels) else f'class{idx}')
        map_column = class_metrics[idx].get('map50') or class_metrics[idx].get('map50_95')
        map_value = _as_float(final_row.get(map_column)) or 0
        recall_value = _as_float(final_row.get(class_metrics[idx].get('recall'))) or 0
        map_values.append(map_value)
        recall_values.append(recall_value)

    return {
        'labels': labels,
        'map': map_values,
        'recall': recall_values
    }


def _resolve_project_path(project: str) -> Path:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = (ROOT / project_path).resolve()
    else:
        project_path = project_path.resolve()
    if ROOT not in project_path.parents and project_path != ROOT:
        raise ValueError('项目目录必须在仓库根目录下')
    return project_path


def _list_resume_candidates(project_path: Path):
    candidates = []
    if project_path.exists() and project_path.is_dir():
        for run_dir in sorted(project_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            checkpoint_files = []
            for rel_path, label in [
                ('last.pt', 'last.pt'),
                ('weights/last.pt', 'weights/last.pt'),
                ('weights/best.pt', 'weights/best.pt')
            ]:
                candidate = run_dir / rel_path
                if candidate.exists():
                    checkpoint_files.append((candidate, label))
            if not checkpoint_files:
                continue

            checkpoint_path, checkpoint_label = checkpoint_files[0]
            candidates.append({
                'name': run_dir.name,
                'project': str(project_path.relative_to(ROOT)) if ROOT in project_path.parents or project_path == ROOT else str(project_path),
                'dir': str(run_dir.relative_to(ROOT)) if ROOT in run_dir.parents else str(run_dir),
                'display_name': f"{run_dir.name} — {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} ({checkpoint_label})",
                'checkpoint': str(checkpoint_path.relative_to(ROOT)) if ROOT in checkpoint_path.parents else str(checkpoint_path),
                'checkpoint_name': checkpoint_label,
                'modified_time': run_dir.stat().st_mtime
            })

    if not candidates:
        rel_project = str(project_path.relative_to(ROOT)).replace('\\', '/') if ROOT in project_path.parents or project_path == ROOT else str(project_path).replace('\\', '/')
        for candidate in sorted(ROOT.rglob('**/last.pt'), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate_rel = str(candidate.relative_to(ROOT)).replace('\\', '/')
            if rel_project not in candidate_rel:
                continue
            run_dir = candidate.parent
            if run_dir.name == 'weights':
                run_dir = run_dir.parent
            if not run_dir.is_dir():
                continue
            candidates.append({
                'name': run_dir.name,
                'project': str(run_dir.parent.relative_to(ROOT)).replace('\\', '/') if ROOT in run_dir.parent.parents or run_dir.parent == ROOT else str(run_dir.parent),
                'dir': str(run_dir.relative_to(ROOT)).replace('\\', '/'),
                'display_name': f"{run_dir.name} — {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} (weights/last.pt)",
                'checkpoint': str(candidate.relative_to(ROOT)).replace('\\', '/'),
                'checkpoint_name': 'weights/last.pt',
                'modified_time': run_dir.stat().st_mtime
            })

    return candidates


def _unescape_json_unicode(text: str) -> str:
    if text is None:
        return ''
    if '\\u' not in text and '\\U' not in text:
        return text
    def _replace(match):
        code = int(match.group(1), 16)
        try:
            return chr(code)
        except ValueError:
            return match.group(0)
    text = re.sub(r'\\u([0-9a-fA-F]{4})', _replace, text)
    text = re.sub(r'\\U([0-9a-fA-F]{8})', _replace, text)
    return text


def _load_yaml_file(path: Path):
    last_exception = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp936', 'gbk', 'latin1'):
        try:
            text = path.read_text(encoding=encoding)
            return yaml.safe_load(text)
        except UnicodeDecodeError as e:
            last_exception = e
            continue
        except Exception:
            raise
    raise UnicodeDecodeError('utf-8', b'', 0, 1, f'Failed to decode {path} using fallback encodings')

def normalize_log_line(line: str) -> str:
    if line is None:
        return ''
    # Strip ANSI escape sequences like ESC[31m, ESC[K, ESC[0m
    cleaned = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
    # Remove leftover escape characters, carriage returns, and C1 control chars
    cleaned = cleaned.replace('\x1b', '').replace('\u001b', '').replace('\r', '')
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
    cleaned = _unescape_json_unicode(cleaned)
    return cleaned
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS

# ========== 直接使用项目根目录（修复Windows路径别名问题）==========
WEB_DIR = Path(r'D:\Personal_Files\Projects\EdgeDistillDet\web')
ROOT = WEB_DIR.parent
_template_path = WEB_DIR / 'templates'
_static_path = WEB_DIR / 'static'

print(f"\n{'='*60}")
print(f"  EdgeDistillDet BOOT DIAGNOSTIC")
print(f"  __file__     : {__file__}")
print(f"  WEB_DIR      : {WEB_DIR}")
print(f"  ROOT         : {ROOT}")
print(f"  CWD          : {Path.cwd()}")
print(f"  templates/   : {_template_path.exists()}")
print(f"  index.html   : {(_template_path / 'index.html').exists()}")
print(f"{'='*60}\n")

# 确保 ROOT 在 sys.path 中
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

app = Flask(__name__,
            template_folder=str(_template_path),
            static_folder=str(_static_path),
            static_url_path='/static')
CORS(app)

training_process = None
log_queue = queue.Queue(maxsize=2000)
training_status = {
    "running": False,
    "pid": None,
    "config": None,
    "mode": "full",
    "start_time": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": [],
    "metrics_history": []
}


@app.route('/')
def index():
    """直接读取HTML文件返回，绕过Jinja2模板查找（修复Windows路径别名问题）"""
    _html_file = Path(r'D:\Personal_Files\Projects\EdgeDistillDet\web\templates\index.html')
    if not _html_file.exists():
        return f"<h1>Error: Template not found at {_html_file}</h1>", 500
    with open(_html_file, 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/configs', methods=['GET'])
def get_configs():
    config_dir = ROOT / 'configs'
    configs = {}
    for f in config_dir.glob('*.yaml'):
        try:
            configs[f.name] = _load_yaml_file(f)
        except Exception as e:
            configs[f.name] = {"_error": str(e)}
    return jsonify({"status": "ok", "configs": configs})


@app.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    config_path = ROOT / 'configs' / config_name
    if not config_path.exists():
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404
    cfg = _load_yaml_file(config_path)
    return jsonify({"status": "ok", "config": cfg})


@app.route('/api/config/recent', methods=['GET'])
def get_recent_config():
    config_dir = ROOT / 'configs'
    recent_meta_path = config_dir / '.recent_config.json'
    if recent_meta_path.exists():
        try:
            with open(recent_meta_path, 'r', encoding='utf-8') as meta_file:
                meta = json.load(meta_file)
            config_name = meta.get('name')
            if config_name:
                config_path = config_dir / config_name
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                    return jsonify({"status": "ok", "config": cfg, "name": config_name})
        except Exception:
            pass

    recent_file = None
    recent_mtime = 0
    for f in config_dir.glob('*.yaml'):
        mtime = f.stat().st_mtime
        if recent_file is None or mtime > recent_mtime:
            recent_file = f
            recent_mtime = mtime
    if recent_file is None:
        return jsonify({"error": "未找到任何配置文件"}), 404
    with open(recent_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return jsonify({"status": "ok", "config": cfg, "name": recent_file.name})


@app.route('/api/config/save', methods=['POST'])
def save_config():
    data = request.json
    config_name = data.get('name', 'distill_config.yaml')
    config_data = data.get('config')  
    config_path = ROOT / 'configs' / config_name
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    recent_meta_path = ROOT / 'configs' / '.recent_config.json'
    with open(recent_meta_path, 'w', encoding='utf-8') as meta_file:
        json.dump({"name": config_name, "saved_at": datetime.utcnow().isoformat() + 'Z'}, meta_file, ensure_ascii=False, indent=2)
    
    return jsonify({"status": "ok", "message": f"配置已保存: {config_name}"})


@app.route('/api/train/start', methods=['POST'])
def start_training():
    global training_process, training_status
    
    if training_status["running"]:
        return jsonify({"error": "已有训练任务在运行中"}), 400
    
    data = request.json or {}
    config_name = data.get('config', 'distill_config.yaml')
    mode = data.get('mode', 'full')
    checkpoint = data.get('checkpoint')
    if mode not in {'full', 'resume'}:
        return jsonify({"error": f"不支持的训练模式: {mode}"}), 400

    config_path = ROOT / 'configs' / config_name
    if not config_path.exists():
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404

    total_epochs = 0
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        total_epochs = int(config_data.get('training', {}).get('epochs', 0) or 0)
    except Exception:
        total_epochs = 0

    def run_train():
        global training_process, training_status
        try:
            mode_msg = f"训练模式: {mode}"
            training_status["logs"].append(mode_msg)
            try:
                log_queue.put_nowait(mode_msg)
            except queue.Full:
                pass

            if mode == 'resume':
                resume_msg = "断点续训模式已启用，正在尝试从上次 checkpoint 自动恢复训练。"
                training_status["logs"].append(resume_msg)
                try:
                    log_queue.put_nowait(resume_msg)
                except queue.Full:
                    pass

            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            cmd = [sys.executable, str(ROOT / 'main.py'), 'train', '--config', str(config_path)]
            if mode == 'resume':
                if checkpoint:
                    checkpoint_path = Path(checkpoint)
                    if not checkpoint_path.is_absolute():
                        checkpoint_path = (ROOT / checkpoint_path).resolve()
                    cmd.extend(['--resume', str(checkpoint_path)])
                else:
                    cmd.append('--resume')

            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            start_new_session = False if os.name == 'nt' else True
            training_process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env,
                creationflags=creation_flags,
                start_new_session=start_new_session
            )

            training_status["pid"] = training_process.pid
            
            buffer = ''
            while True:
                char = training_process.stdout.read(1)
                if not char:
                    break

                buffer += char

                if char in ('\r', '\n'):
                    cleaned = normalize_log_line(buffer).rstrip('\r\n')
                    buffer = ''

                    if cleaned:
                        training_status["logs"].append(cleaned)
                        try:
                            log_queue.put_nowait(cleaned)
                        except queue.Full:
                            pass

                        epoch_match = re.search(r"\bEpoch\s*(\d+)\s*/\s*(\d+)\b", cleaned, re.IGNORECASE)
                        if not epoch_match:
                            raw_match = re.search(r"^\s*(\d+)\s*/\s*(\d+)\b", cleaned)
                            if raw_match and re.search(r"\bGPU_mem\b|\bbox_loss\b|\bcls_loss\b|\bdfl_loss\b|\bSize\b|\bInstances\b|\bit/s\b|/s\b", cleaned, re.IGNORECASE):
                                epoch_match = raw_match

                        if epoch_match:
                            current_epoch = int(epoch_match.group(1))
                            total_epochs = int(epoch_match.group(2))
                            training_status["current_epoch"] = current_epoch
                            training_status["total_epochs"] = total_epochs
            training_process.wait()
        except Exception as e:
            err_msg = f"训练异常: {str(e)}"
            training_status["logs"].append(err_msg)
            try:
                log_queue.put_nowait(err_msg)
            except queue.Full:
                pass
        finally:
            training_status["running"] = False
            training_status["pid"] = None
            training_process = None
    
    training_status.update({
        "running": True,
        "pid": None,
        "config": config_name,
        "mode": mode,
        "start_time": time.time(),
        "current_epoch": 1 if total_epochs else 0,
        "total_epochs": total_epochs,
        "logs": ["正在启动训练..."]
    })
    try:
        log_queue.put_nowait("正在启动训练...")
    except queue.Full:
        pass
    
    thread = threading.Thread(target=run_train, daemon=True)
    thread.start()
    
    return jsonify({"status": "ok", "message": "训练已启动"})


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_process, training_status
    
    if not training_status["running"]:
        return jsonify({"warning": "没有运行中的训练任务"})
    
    if training_process:
        stop_msg = "训练停止请求已发送，正在终止训练进程..."
        training_status["logs"].append(stop_msg)
        try:
            log_queue.put_nowait(stop_msg)
        except queue.Full:
            pass

        try:
            training_process.terminate()
        except Exception:
            pass

        for _ in range(5):
            if training_process.poll() is not None:
                break
            time.sleep(1)

        if training_process.poll() is None:
            try:
                training_process.kill()
            except Exception:
                pass

        training_status["running"] = False
        training_process = None
        stop_msg = "训练已停止，最近完成的 checkpoint 已保留，可通过断点续训继续。"
        training_status["logs"].append(stop_msg)
        try:
            log_queue.put_nowait(stop_msg)
        except queue.Full:
            pass
        
    return jsonify({"status": "ok", "message": "训练已停止"})


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    return jsonify({
        "status": "ok",
        **training_status,
        "log_count": len(training_status["logs"])
    })


@app.route('/api/train/resume_candidates', methods=['GET'])
def get_resume_candidates():
    project = request.args.get('project', 'runs/distill') or 'runs/distill'
    try:
        project_path = _resolve_project_path(project)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    candidates = _list_resume_candidates(project_path)
    return jsonify({
        "status": "ok",
        "project": str(project_path.relative_to(ROOT)) if ROOT in project_path.parents or project_path == ROOT else str(project_path),
        "candidates": candidates
    })


@app.route('/api/train/logs', methods=['GET'])
def get_training_logs():
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 100, type=int)
    offset = max(0, offset)
    limit = max(1, min(limit, 500))
    
    logs = training_status["logs"]
    total = len(logs)
    if offset > total:
        offset = total
    paginated_logs = logs[offset:offset + limit]
    
    return jsonify({
        "status": "ok",
        "logs": paginated_logs,
        "total": total,
        "offset": offset,
        "limit": limit
    })


@app.route('/api/train/logs/download')
def download_training_logs():
    logs = [normalize_log_line(line) for line in training_status.get('logs', [])]
    content = '\n'.join(logs)
    response = Response(content, mimetype='text/plain; charset=utf-8')
    response.headers['Content-Disposition'] = 'attachment; filename="training_logs.txt"'
    return response


@app.route('/api/train/logs/stream')
def stream_training_logs():
    def generate():
        while training_status["running"] or not log_queue.empty():
            try:
                line = log_queue.get(timeout=2)
                line = normalize_log_line(line)
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            except queue.Empty:
                if not training_status["running"] and log_queue.empty():
                    break
                yield ": keepalive\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    import csv

    as_float = _as_float

    def format_seconds(seconds):
        try:
            seconds = float(seconds)
        except Exception:
            return '--'
        if seconds < 0:
            return '--'
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    def load_csv_summary(path):
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader]
            return list(reader.fieldnames or []), rows
        except Exception:
            return [], []

    def summarize_series(rows, key, better='higher'):
        values = [as_float(row.get(key)) for row in rows if row.get(key) is not None]
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

    def build_metric_series(rows, columns, run_dir):
        chart = {
            'epochs': [],
            'train_losses': {'box_loss': [], 'cls_loss': [], 'dfl_loss': []},
            'map_series': {'map50': [], 'map50_95': []},
            'lr_series': {'pg0': [], 'pg1': [], 'pg2': []},
            'distill_series': {},
            'pr_curve': None,
            'class_performance': None
        }
        for row in rows:
            epoch = as_float(row.get('epoch'))
            if epoch is None:
                continue
            chart['epochs'].append(int(epoch))
            chart['train_losses']['box_loss'].append(as_float(row.get('train/box_loss')) or 0)
            chart['train_losses']['cls_loss'].append(as_float(row.get('train/cls_loss')) or 0)
            chart['train_losses']['dfl_loss'].append(as_float(row.get('train/dfl_loss')) or 0)
            chart['map_series']['map50'].append(as_float(row.get('metrics/mAP50(B)')) or 0)
            chart['map_series']['map50_95'].append(as_float(row.get('metrics/mAP50-95(B)')) or 0)
            chart['lr_series']['pg0'].append(as_float(row.get('lr/pg0')) or 0)
            chart['lr_series']['pg1'].append(as_float(row.get('lr/pg1')) or 0)
            chart['lr_series']['pg2'].append(as_float(row.get('lr/pg2')) or 0)
            if 'distill/alpha' in row or 'distill/temperature' in row or 'distill/kd_loss' in row:
                chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [as_float(row.get('distill/alpha')) or 0]
                chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [as_float(row.get('distill/temperature')) or 0]
                chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [as_float(row.get('distill/kd_loss')) or 0]

        labels = _parse_dataset_labels(run_dir)
        chart['class_performance'] = _extract_class_performance(rows, columns, labels)
        return chart

    runs_dir = ROOT / 'runs'
    metrics_data = []
    if runs_dir.exists():
        run_dirs = [p for p in runs_dir.rglob('*') if p.is_dir() and p.name.startswith('exp')]
        run_dirs = sorted({p.resolve(): p for p in run_dirs}.values(), key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            result_file = run_dir / 'results.csv'
            has_results = result_file.exists()
            display_name = f"{run_dir.name} @ {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            if not has_results:
                display_name += ' (无结果)'

            entry = {
                'name': run_dir.name,
                'display_name': display_name,
                'dir': str(run_dir.relative_to(ROOT)),
                'has_results': has_results,
                'path': str(result_file.relative_to(ROOT)) if has_results else ''
            }
            if has_results:
                try:
                    with open(result_file, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.DictReader(f)
                        columns = list(reader.fieldnames or [])
                    with open(result_file, 'r', encoding='utf-8', newline='') as counter_file:
                        row_count = sum(1 for _ in counter_file) - 1
                    entry['columns'] = columns
                    entry['rows'] = max(row_count, 0)
                except Exception:
                    entry['columns'] = []
                    entry['rows'] = 0
            else:
                entry['columns'] = []
                entry['rows'] = 0
            metrics_data.append(entry)

    selected_path = request.args.get('source', '').strip()
    selected_data = None
    if selected_path:
        target = (ROOT / selected_path).resolve()
        if target.exists() and target.is_file() and target.suffix == '.csv' and ROOT in target.parents:
            columns, rows = load_csv_summary(target)
            if rows:
                chart_series = build_metric_series(rows, columns, target.parent)
                summary_metrics = {
                    'box_loss': summarize_series(rows, 'train/box_loss', better='lower'),
                    'cls_loss': summarize_series(rows, 'train/cls_loss', better='lower'),
                    'dfl_loss': summarize_series(rows, 'train/dfl_loss', better='lower'),
                    'map50': summarize_series(rows, 'metrics/mAP50(B)', better='higher'),
                    'map50_95': summarize_series(rows, 'metrics/mAP50-95(B)', better='higher'),
                    'precision': summarize_series(rows, 'metrics/precision(B)', better='higher'),
                    'recall': summarize_series(rows, 'metrics/recall(B)', better='higher')
                }
                total_time = None
                for row in reversed(rows):
                    total_time = as_float(row.get('time'))
                    if total_time is not None:
                        break

                run_stats = _estimate_run_stats(target.parent)
                selected_data = {
                    'source': selected_path,
                    'columns': columns,
                    'rows': len(rows),
                    'chart_series': chart_series,
                    'summary_metrics': {k: v for k, v in summary_metrics.items() if v is not None},
                    'overview_stats': {
                        'ov-map50': f"{(summary_metrics['map50']['best'] * 100):.2f}%" if summary_metrics.get('map50') else '--',
                        **run_stats,
                        'ov-time': format_seconds(total_time)
                    }
                }

    distill_logs = list(runs_dir.rglob('distill_log.json')) if runs_dir.exists() else []
    response = {
        'status': 'ok',
        'csv_metrics': metrics_data,
        'distill_logs': [str(p.relative_to(ROOT)) for p in distill_logs]
    }
    if selected_data:
        response.update(selected_data)
    return jsonify(response)


@app.route('/api/agent', methods=['GET', 'POST'])
def agent_interface():
    if request.method == 'GET':
        return jsonify({
            "status": "ok",
            "message": "Agent 接口就绪",
            "available_actions": [
                {"name": "auto_tune", "description": "自动超参数调优"},
                {"name": "analyze_result", "description": "分析训练结果"},
                {"name": "suggest_config", "description": "基于数据集建议配置"},
                {"name": "compare_models", "description": "模型对比分析"}
            ]
        })
    else:
        data = request.json or {}
        action = data.get('action')
        params = data.get('params', {})
        
        agent_responses = {
            "auto_tune": {"status": "pending", "message": "自动调优功能开发中"},
            "analyze_result": {"status": "pending", "message": "结果分析功能开发中"},
            "suggest_config": {"status": "pending", "message": "配置建议功能开发中"},
            "compare_models": {"status": "pending", "message": "模型对比分析功能开发中"}
        }
        
        response = agent_responses.get(action, {"error": f"未知操作: {action}"})
        response["action"] = action
        response["params"] = params
        
        return jsonify(response)


@app.route('/api/files/browse', methods=['GET'])
def browse_files():
    path = request.args.get('path', str(ROOT))
    
    try:
        target = Path(path)
        if not target.exists():
            return jsonify({"error": "路径不存在"}), 404
            
        items = []
        for item in target.iterdir():
            try:
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else None
                })
            except:
                pass
                
        return jsonify({
            "status": "ok",
            "current_path": path,
            "parent": str(target.parent) if target != target.parent else None,
            "items": sorted(items, key=lambda x: (not x["is_dir"], x["name"].lower()))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/output/check', methods=['GET'])
def check_output_name():
    project = request.args.get('project', 'runs/distill') or 'runs/distill'
    try:
        project_path = Path(project)
        if not project_path.is_absolute():
            project_path = (ROOT / project_path).resolve()
        else:
            project_path = project_path.resolve()

        if ROOT not in project_path.parents and project_path != ROOT:
            return jsonify({"error": "项目目录必须在仓库根目录下"}), 400

        existing_names = []
        if project_path.exists() and project_path.is_dir():
            for item in project_path.iterdir():
                if item.is_dir():
                    existing_names.append(item.name)

        exp_numbers = [-1]
        for name in existing_names:
            match = re.fullmatch(r'exp(\d*)', name)
            if match:
                exp_numbers.append(int(match.group(1) or 0))

        next_exp = 'exp1'
        if exp_numbers:
            next_index = max(exp_numbers) + 1
            next_exp = f'exp{next_index}'

        return jsonify({
            "status": "ok",
            "project": str(project_path.relative_to(ROOT)) if ROOT in project_path.parents or project_path == ROOT else str(project_path),
            "existing_names": existing_names,
            "next_exp_name": next_exp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    print("=" * 55)
    print("  EdgeDistillDet Web UI")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=True)
