from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi.responses import JSONResponse

from web.core.paths import BASE_DIR
from web.services.backend_common import _error
from web.services.backend_train_runtime import _kill_process_tree

SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript", "tflite", "saved_model", "coreml"}

_export_process = None
_export_lock = threading.RLock()
_export_status = {
    'running': False,
    'pid': None,
    'start_time': None,
    'finish_time': None,
    'exit_code': None,
    'output_path': None,
    'logs': [],
}


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text or '')


def _validate_export_payload(payload: dict) -> str | None:
    export_path = payload.get('export_path')
    if not export_path or not str(export_path).strip():
        return '导出路径 export_path 不能为空。'

    export_target = _resolve_path(str(export_path))
    if export_target is None:
        return f'导出路径无效: {export_path}'

    parent_dir = export_target.parent if export_target.suffix else export_target
    if not parent_dir.exists() or not parent_dir.is_dir():
        return f'导出路径不存在: {parent_dir}'

    fmt = str(payload.get('format') or '').lower()
    if fmt and fmt not in SUPPORTED_EXPORT_FORMATS:
        return f'导出格式无效: {fmt}'

    opset = payload.get('opset')
    if opset is not None and isinstance(opset, int) and opset < 0:
        return 'ONNX opset 必须为非负整数。'

    workspace = payload.get('workspace')
    if workspace is not None and isinstance(workspace, int) and workspace < 0:
        return 'workspace 必须为非负整数。'

    weight = payload.get('weight')
    if weight:
        weight_path = _resolve_path(str(weight))
        if not weight_path or not weight_path.exists():
            return f'指定的权重文件不存在: {weight}'

    return None


def _append_log(line: str) -> None:
    with _export_lock:
        clean_line = _strip_ansi(line).rstrip('\r\n')
        if not clean_line:
            return
        _export_status['logs'].append(clean_line)
        if len(_export_status['logs']) > 4000:
            _export_status['logs'] = _export_status['logs'][-4000:]


def _read_process_output(proc: subprocess.Popen) -> None:
    try:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            if raw_line is None:
                break
            _append_log(raw_line)
    finally:
        with _export_lock:
            _export_status['running'] = False
            _export_status['finish_time'] = time.time()
            _export_status['exit_code'] = proc.returncode
            _export_status['pid'] = None
            global _export_process
            _export_process = None


def _build_command(payload: dict) -> list[str]:
    script = BASE_DIR / 'scripts' / 'export_model.py'
    return [sys.executable, '-u', str(script)]


def _serialize_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def start_export(payload: dict):
    global _export_process
    validation_error = _validate_export_payload(payload)
    if validation_error:
        return _error(validation_error, 400)

    with _export_lock:
        if _export_status['running']:
            return _error('模型导出任务已在运行中', 400)
        _export_status['logs'].clear()
        _export_status['exit_code'] = None
        _export_status['finish_time'] = None
        _export_status['start_time'] = time.time()
        _export_status['output_path'] = payload.get('export_path')

        command = _build_command(payload)
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(BASE_DIR),
            )
        except Exception as exc:
            return _error(f'启动模型导出失败: {exc}', 500)

        if proc.stdin is not None:
            try:
                proc.stdin.write(_serialize_payload(payload))
            except Exception:
                pass
            finally:
                proc.stdin.close()

        _export_status['running'] = True
        _export_status['pid'] = proc.pid
        _export_process = proc
        thread = threading.Thread(target=_read_process_output, args=(proc,), daemon=True)
        thread.start()

    return {'status': 'ok', 'message': '模型导出已启动', 'pid': proc.pid}


def stop_export():
    global _export_process
    with _export_lock:
        proc = _export_process
        if proc is None or proc.poll() is not None:
            return _error('当前没有正在运行的模型导出任务', 400)
        if proc.pid is not None:
            _kill_process_tree(proc.pid, force=True)
        _export_status['running'] = False
        _export_status['finish_time'] = time.time()
        _export_status['exit_code'] = proc.returncode
        _export_status['pid'] = None
        _export_process = None
    return {'status': 'ok', 'message': '模型导出已停止'}


def get_export_status():
    with _export_lock:
        return {
            'running': _export_status['running'],
            'pid': _export_status['pid'],
            'start_time': _export_status['start_time'],
            'finish_time': _export_status['finish_time'],
            'exit_code': _export_status['exit_code'],
            'output_path': _export_status['output_path'],
        }


def get_export_logs(offset: int = 0, limit: int = 100):
    with _export_lock:
        logs = list(_export_status['logs'])
    return {
        'offset': min(len(logs), max(0, offset)),
        'logs': logs[offset:offset + limit],
    }
