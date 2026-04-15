"""
EdgeDistillDet Web Server - 自包含启动脚本
==========================================
不依赖 import app.py，彻底避免 Windows 路径缓存问题

使用方法: python web/server.py
"""

# ============ 路径配置（自动检测，不依赖硬编码）============
import os as _os
import sys as _sys
from pathlib import Path as _Path

BASE_DIR = str(_Path(__file__).resolve().parent.parent)
WEB_DIR = BASE_DIR + r'\web'
TEMPLATE_FILE = WEB_DIR + r'\templates\index.html'
STATIC_DIR = WEB_DIR + r'\static'

import os
import sys
import json
import re
import time
import csv
import yaml
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, Response, send_from_directory, render_template
from flask_cors import CORS

print("=" * 60)
print("  EdgeDistillDet Web Server (Standalone)")
print(f"  BASE_DIR      : {BASE_DIR}")
print(f"  TEMPLATE      : {TEMPLATE_FILE}")
print(f"  Template exists: {os.path.exists(TEMPLATE_FILE)}")
print(f"  CWD           : {os.getcwd()}")
print(f"  sys.executable: {sys.executable}")
print("=" * 60)

if not os.path.exists(TEMPLATE_FILE):
    print(f"[FATAL] Template file not found: {TEMPLATE_FILE}")
    sys.exit(1)

sys.path.insert(0, BASE_DIR)

app = Flask(__name__,
            template_folder=WEB_DIR + r'\templates',
            static_folder=STATIC_DIR,
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


def _resolve_project_path(project: str) -> Path:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = (Path(BASE_DIR) / project_path).resolve()
    else:
        project_path = project_path.resolve()
    if not str(project_path).startswith(str(Path(BASE_DIR).resolve())):
        raise ValueError('项目目录必须在仓库根目录下')
    return project_path


def _list_resume_candidates(project_path: Path):
    candidates = []
    if not project_path.exists() or not project_path.is_dir():
        return candidates
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
            'project': str(Path(project_path).relative_to(Path(BASE_DIR))) if Path(BASE_DIR) in project_path.parents or project_path == Path(BASE_DIR) else str(project_path),
            'dir': str(Path(run_dir).relative_to(Path(BASE_DIR))) if Path(BASE_DIR) in run_dir.parents else str(run_dir),
            'display_name': f"{run_dir.name} — {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} ({checkpoint_label})",
            'checkpoint': str(Path(checkpoint_path).relative_to(Path(BASE_DIR))) if Path(BASE_DIR) in checkpoint_path.parents else str(checkpoint_path),
            'checkpoint_name': checkpoint_label,
            'modified_time': run_dir.stat().st_mtime
        })
    return candidates


def _load_yaml_file(path: Path):
    last_exception = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp936', 'gbk', 'latin1'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError as e:
            last_exception = e
            continue
        except Exception:
            raise
    raise UnicodeDecodeError('utf-8', b'', 0, 1, f'Failed to decode {path} using fallback encodings')

try:
    from utils.edge_profiler import EdgeProfiler
except Exception:
    EdgeProfiler = None


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

    # EdgeProfiler 失败时的回退：单独计算参数量和 FPS
    params_m = _estimate_model_params(model_path)
    fps_str = '--'

    # 尝试通过 ultralytics 独立获取 GFLOPs 并计算理论 FPS（GPU FP16 基准）
    gflops = _estimate_model_gflops(model_path)
    if gflops is None and params_m is not None:
        # 粗略估算：YOLO 系列模型 GFLOPs ≈ params_m * 2（640px 输入典型值）
        gflops = params_m * 2.0
    if gflops is not None and gflops > 0:
        # 参考 RTX 3060 FP16 规格：13 TOPS，利用率 60%
        gpu_tops_fp16 = 13.0
        efficiency = 0.6
        fps = (gpu_tops_fp16 * 1e12 * efficiency) / (gflops * 1e9)
        fps_str = f"{fps:.0f}"

    return {
        'ov-fps': fps_str if fps_str != '--' else 'N/A',
        'ov-params': f"{params_m:.1f}" if params_m is not None else '--'
    }


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


def _load_distill_log_json(run_dir):
    """尝试从 run_dir 中加载 distill_log.json 作为蒸馏数据的回退源。"""
    if not run_dir or not isinstance(run_dir, (str, Path)):
        return []
    try:
        log_path = Path(run_dir) / 'distill_log.json'
        if not log_path.exists():
            # 尝试在子目录中查找（如 weights/ 或上级目录）
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
        'distill_series': {},
        'pr_curve': None,
        'class_performance': None
    }

    # 检查 CSV 是否包含蒸馏列
    has_distill_columns = any(
        col.startswith('distill/') for col in (columns or [])
    )

    # 预加载 distill_log.json（回退数据源）
    distill_log_fallback = _load_distill_log_json(run_dir) if not has_distill_columns else []

    # 将 distill_log.json 按 epoch 建立索引
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

        # 蒸馏数据：优先从 CSV 列读取，回退到 distill_log.json
        if has_distill_columns and ('distill/alpha' in row or 'distill/temperature' in row or 'distill/kd_loss' in row):
            alpha_val = _as_float(row.get('distill/alpha'))
            temp_val = _as_float(row.get('distill/temperature'))
            kd_val = _as_float(row.get('distill/kd_loss'))
            chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [alpha_val if alpha_val is not None else None]
            chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [temp_val if temp_val is not None else None]
            chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [kd_val if kd_val is not None else None]
        elif distill_by_epoch:
            # 从 distill_log.json 回退：精确匹配 epoch，若没有则尝试 epoch-1（因为记录时机可能差一个）
            de = distill_by_epoch.get(epoch_int) or distill_by_epoch.get(epoch_int - 1)
            if de:
                alpha_val = de.get('alpha')
                temp_val = de.get('temperature')
                # 兼容新旧字段名：新版用 avg_kd_loss，旧版可能用 kd_loss
                kd_val = de.get('kd_loss') or de.get('avg_kd_loss')
                chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [_as_float(alpha_val)]
                chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [_as_float(temp_val)]
                chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [_as_float(kd_val)]

    return chart


@app.route('/')
def index():
    """读取HTML并手动替换Jinja2模板变量（完全绕过Flask模板查找机制）"""
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        html = f.read()

    # 替换 {{ url_for('static', filename='xxx') }}  => /static/xxx
    # 替换 {{ url_for("static", filename="xxx") }}  => /static/xxx
    def replace_static_url(m):
        fname = m.group(1)
        return '/static/' + fname

    # 单引号形式: url_for('static', filename='...')
    html = re.sub(
        r"\{\{\s*url_for\(\s*'static'\s*,\s*filename\s*=\s*'([^']+)'\s*\)\s*\}\}",
        replace_static_url,
        html
    )
    # 双引号形式: url_for("static", filename="...")
    html = re.sub(
        r'\{\{\s*url_for\(\s*"static"\s*,\s*filename\s*=\s*"([^"]+)"\s*\)\s*\}\}',
        replace_static_url,
        html
    )

    return html


@app.route('/api/configs', methods=['GET'])
def get_configs():
    config_dir = os.path.join(BASE_DIR, 'configs')
    configs = {}
    for fname in os.listdir(config_dir):
        if fname.endswith('.yaml'):
            try:
                configs[fname] = _load_yaml_file(os.path.join(config_dir, fname))
            except Exception as e:
                configs[fname] = {"_error": str(e)}
    return jsonify({"status": "ok", "configs": configs})


@app.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    config_path = os.path.join(BASE_DIR, 'configs', config_name)
    if not os.path.exists(config_path):
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404
    cfg = _load_yaml_file(config_path)
    return jsonify({"status": "ok", "config": cfg})


@app.route('/api/config/save', methods=['POST'])
def save_config():
    data = request.json
    config_name = data.get('name', 'distill_config.yaml')
    config_data = data.get('config')
    config_path = os.path.join(BASE_DIR, 'configs', config_name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return jsonify({"status": "ok", "message": f"配置已保存: {config_name}"})


@app.route('/api/train/start', methods=['POST'])
def start_training():
    global training_process, training_status

    if training_status["running"]:
        return jsonify({"error": "已有训练任务在运行中"}), 400

    data = request.json or {}
    config_name = data.get('config', 'distill_config.yaml')
    mode = data.get('mode', 'distill')
    checkpoint = data.get('checkpoint')
    if mode not in {'distill', 'resume'}:
        return jsonify({"error": f"不支持的训练模式: {mode}"}), 400

    config_path = os.path.join(BASE_DIR, 'configs', config_name)
    if not os.path.exists(config_path):
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404

    def run_train():
        global training_process, training_status
        try:
            training_status["logs"].append(f"训练模式: {mode}")
            for msg in list(training_status["logs"][-5:]):
                try:
                    log_queue.put_nowait(msg)
                except queue.Full:
                    pass

            if mode == 'distill':
                cmd = [sys.executable, '-u', os.path.join(BASE_DIR, 'scripts', 'train_with_distill.py'), '--config', config_path]
            else:
                main_py = os.path.join(BASE_DIR, 'main.py')
                cmd = [sys.executable, main_py, 'train', '--config', config_path]
                if mode == 'resume':
                    if checkpoint:
                        checkpoint_path = Path(checkpoint)
                        if not checkpoint_path.is_absolute():
                            checkpoint_path = (Path(BASE_DIR) / checkpoint_path).resolve()
                        cmd.extend(['--resume', str(checkpoint_path)])
                    else:
                        cmd.append('--resume')
            training_process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )

            training_status["pid"] = training_process.pid

            buffer = ''
            while True:
                char = training_process.stdout.read(1)
                if not char:
                    break
                buffer += char
                if char in ('\r', '\n'):
                    cleaned = buffer.strip()
                    buffer = ''
                    if cleaned:
                        training_status["logs"].append(cleaned)
                        try:
                            log_queue.put_nowait(cleaned)
                        except queue.Full:
                            pass
                        epoch_match = re.search(r"Epoch\s*(\d+)\s*/\s*(\d+)", cleaned, re.IGNORECASE)
                        if epoch_match:
                            training_status["current_epoch"] = int(epoch_match.group(1))
                            training_status["total_epochs"] = int(epoch_match.group(2))
                        if len(training_status["logs"]) > 1000:
                            training_status["logs"] = training_status["logs"][-500:]

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
        "current_epoch": 0,
        "total_epochs": 0,
        "logs": ["正在启动训练..."]
    })

    thread = threading.Thread(target=run_train, daemon=True)
    thread.start()
    return jsonify({"status": "ok", "message": "训练已启动"})


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_process, training_status
    if not training_status["running"]:
        return jsonify({"warning": "没有运行中的训练任务"})
    if training_process:
        training_process.terminate()
        training_status["running"] = False
        training_process = None
        stop_msg = "训练已被用户停止"
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
        "project": str(Path(project_path).relative_to(Path(BASE_DIR))) if Path(BASE_DIR) in project_path.parents or project_path == Path(BASE_DIR) else str(project_path),
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


@app.route('/api/train/logs/stream')
def stream_training_logs():
    def generate():
        for line in training_status["logs"]:
            yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
        while training_status["running"]:
            try:
                line = log_queue.get(timeout=2)
                yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
        drained = []
        while True:
            try:
                drained.append(log_queue.get_nowait())
            except queue.Empty:
                break
        for line in drained:
            yield f"data: {json.dumps({'line': line}, ensure_ascii=False)}\n\n"
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
    runs_dir = os.path.join(BASE_DIR, 'runs')
    metrics_data = []
    if os.path.exists(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            for file_name in files:
                if file_name == 'results.csv':
                    result_path = os.path.join(root, file_name)
                    run_dir = os.path.dirname(result_path)
                    display_name = f"{os.path.basename(run_dir)} @ {datetime.fromtimestamp(os.path.getmtime(run_dir)).strftime('%Y-%m-%d %H:%M:%S')}"
                    entry = {
                        "name": os.path.basename(run_dir),
                        "display_name": display_name,
                        "dir": os.path.relpath(run_dir, BASE_DIR),
                        "has_results": True,
                        "path": os.path.relpath(result_path, BASE_DIR)
                    }
                    try:
                        columns, rows = _load_csv_summary(Path(result_path))
                        entry["columns"] = columns
                        entry["rows"] = len(rows)
                    except Exception:
                        entry["columns"] = []
                        entry["rows"] = 0
                    metrics_data.append(entry)
        metrics_data.sort(key=lambda item: os.path.getmtime(os.path.join(BASE_DIR, item['dir'])), reverse=True)

    selected_path = request.args.get('source', '').strip()
    selected_data = None
    if selected_path:
        target = (Path(BASE_DIR) / selected_path).resolve()
        if target.exists() and target.is_file() and target.suffix == '.csv' and str(target).startswith(str(Path(BASE_DIR).resolve())):
            columns, rows = _load_csv_summary(target)
            if rows:
                chart_series = _build_metric_series(rows, columns, target.parent)
                summary_metrics = {
                    'box_loss': _summarize_series(rows, 'train/box_loss', better='lower'),
                    'cls_loss': _summarize_series(rows, 'train/cls_loss', better='lower'),
                    'dfl_loss': _summarize_series(rows, 'train/dfl_loss', better='lower'),
                    'map50': _summarize_series(rows, 'metrics/mAP50(B)', better='higher'),
                    'map50_95': _summarize_series(rows, 'metrics/mAP50-95(B)', better='higher'),
                    'precision': _summarize_series(rows, 'metrics/precision(B)', better='higher'),
                    'recall': _summarize_series(rows, 'metrics/recall(B)', better='higher')
                }
                total_time = None
                for row in reversed(rows):
                    total_time = _as_float(row.get('time'))
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
                        'ov-time': (lambda seconds: f"{int(seconds // 60)}m {int(seconds % 60)}s" if seconds is not None else '--')(_as_float(total_time))
                    }
                }

    response = {
        "status": "ok",
        "csv_metrics": metrics_data
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
        action = data.get('action', '')
        response = {
            "auto_tune": {"status": "pending", "message": "自动调优功能开发中"},
            "analyze_result": {"status": "pending", "message": "结果分析功能开发中"},
            "suggest_config": {"status": "pending", "message": "配置建议功能开发中"},
            "compare_models": {"status": "pending", "message": "模型对比分析功能开发中"}
        }.get(action, {"error": f"未知操作: {action}"})
        response["action"] = action
        response["params"] = data.get('params', {})
        return jsonify(response)


@app.route('/api/files/browse', methods=['GET'])
def browse_files():
    path = request.args.get('path', BASE_DIR)
    try:
        items = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            try:
                items.append({
                    "name": item,
                    "path": full_path,
                    "is_dir": os.path.isdir(full_path),
                    "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None
                })
            except OSError:
                pass
        parent = str(os.path.dirname(path)) if path != BASE_DIR else None
        return jsonify({
            "status": "ok",
            "current_path": path,
            "parent": parent,
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
            project_path = (Path(BASE_DIR) / project_path).resolve()
        else:
            project_path = project_path.resolve()

        base_path = Path(BASE_DIR).resolve()
        if base_path not in project_path.parents and project_path != base_path:
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
            "project": str(project_path.relative_to(base_path)) if base_path in project_path.parents or project_path == base_path else str(project_path),
            "existing_names": existing_names,
            "next_exp_name": next_exp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("\n  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
