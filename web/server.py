"""
EdgeDistillDet Web Server - 自包含启动脚本
==========================================
不依赖 import app.py，彻底避免 Windows 路径缓存问题

使用方法: python D:\Personal_Files\Projects\EdgeDistillDet\web\server.py
"""

# ============ 路径配置（硬编码，不依赖 __file__）============
BASE_DIR = r'D:\Personal_Files\Projects\EdgeDistillDet'
WEB_DIR = BASE_DIR + r'\web'
TEMPLATE_FILE = WEB_DIR + r'\templates\index.html'
STATIC_DIR = WEB_DIR + r'\static'

import os
import sys
import json
import re
import time
import yaml
import subprocess
import threading
import queue
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


@app.route('/')
def index():
    """渲染模板（Flask已正确配置template_folder和static_folder）"""
    try:
        return render_template('index.html')
    except Exception as e:
        # 降级：如果模板查找失败，尝试手动替换 url_for
        import re
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            html = f.read()
        # 手动将 {{ url_for('static', filename=...) }} 替换为 /static/...
        def _url_for_replace(match):
            fname = match.group(1).strip().strip("'\"")
            return f'/static/{fname}'
        html = re.sub(r"url_for\('static',\s*filename\s*=\s*'([^']+)'\)", _url_for_replace, html)
        html = re.sub(r'url_for\("static",\s*filename\s*=\s*"([^"]+)"\)', _url_for_replace, html)
        return html


@app.route('/api/configs', methods=['GET'])
def get_configs():
    config_dir = os.path.join(BASE_DIR, 'configs')
    configs = {}
    for fname in os.listdir(config_dir):
        if fname.endswith('.yaml'):
            with open(os.path.join(config_dir, fname), 'r', encoding='utf-8') as fp:
                configs[fname] = yaml.safe_load(fp)
    return jsonify({"status": "ok", "configs": configs})


@app.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    config_path = os.path.join(BASE_DIR, 'configs', config_name)
    if not os.path.exists(config_path):
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
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
    mode = data.get('mode', 'full')
    if mode not in {'full', 'resume', 'fast'}:
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

            main_py = os.path.join(BASE_DIR, 'main.py')
            training_process = subprocess.Popen(
                [sys.executable, main_py, 'train', '--config', config_path],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                newline='\n'
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
        import pandas as pd
        for root, dirs, files in os.walk(runs_dir):
            csv_files = [f for f in files if f.endswith('.csv')]
            csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True)
            for fname in csv_files[:10]:
                try:
                    fpath = os.path.join(root, fname)
                    df = pd.read_csv(fpath)
                    metrics_data.append({
                        "name": fname.replace('.csv', ''),
                        "path": os.path.relpath(fpath, BASE_DIR),
                        "columns": list(df.columns),
                        "rows": len(df),
                        "preview": df.tail(5).to_dict(orient='records')
                    })
                except Exception:
                    pass
    distill_logs = []
    if os.path.exists(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            distill_logs.extend(
                os.path.relpath(os.path.join(root, f), BASE_DIR)
                for f in files if f == 'distill_log.json'
            )
    return jsonify({
        "status": "ok",
        "csv_metrics": metrics_data,
        "distill_logs": distill_logs
    })


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


# ============ 静态文件路由 ============
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == '__main__':
    print("\n  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
