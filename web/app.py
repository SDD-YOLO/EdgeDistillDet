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
import subprocess
import threading
import queue
from pathlib import Path
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
        with open(f, 'r', encoding='utf-8') as fp:
            configs[f.name] = yaml.safe_load(fp)
    return jsonify({"status": "ok", "configs": configs})


@app.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    config_path = ROOT / 'configs' / config_name
    if not config_path.exists():
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return jsonify({"status": "ok", "config": cfg})


@app.route('/api/config/save', methods=['POST'])
def save_config():
    data = request.json
    config_name = data.get('name', 'distill_config.yaml')
    config_data = data.get('config')  
    config_path = ROOT / 'configs' / config_name
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
    
    config_path = ROOT / 'configs' / config_name
    if not config_path.exists():
        return jsonify({"error": f"配置文件不存在: {config_name}"}), 404
    
    def run_train():
        global training_process, training_status
        try:
            training_status["logs"].append(f"训练模式: {mode}")
            if mode == 'resume':
                training_status["logs"].append("提示: 当前 CLI 未提供 resume 参数，已按常规训练方式启动。")
            elif mode == 'fast':
                training_status["logs"].append("提示: 快速模式已应用简化训练配置。")

            for msg in list(training_status["logs"][-5:]):
                try:
                    log_queue.put_nowait(msg)
                except queue.Full:
                    pass

            training_process = subprocess.Popen(
                [sys.executable, str(ROOT / 'main.py'), 'train', '--config', str(config_path)],
                cwd=str(ROOT),
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
    runs_dir = ROOT / 'runs'
    metrics_data = []
    
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.rglob('*'), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            if run_dir.is_file() and run_dir.suffix == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(run_dir)
                    metrics_data.append({
                        "name": run_dir.stem,
                        "path": str(run_dir.relative_to(ROOT)),
                        "columns": list(df.columns),
                        "rows": len(df),
                        "preview": df.tail(5).to_dict(orient='records')
                    })
                except:
                    pass
    
    distill_logs = list(runs_dir.rglob('distill_log.json')) if runs_dir.exists() else []
    
    return jsonify({
        "status": "ok",
        "csv_metrics": metrics_data,
        "distill_logs": [str(p.relative_to(ROOT)) for p in distill_logs]
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


if __name__ == '__main__':
    print("=" * 55)
    print("  EdgeDistillDet Web UI")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=True)
