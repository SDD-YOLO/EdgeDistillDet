from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi.responses import JSONResponse

from web.core.paths import BASE_DIR
from web.services.backend_common import _error
from web.services.backend_train_runtime import _kill_process_tree

_display_process = None
_display_lock = threading.RLock()
_display_status = {
    'running': False,
    'pid': None,
    'start_time': None,
    'finish_time': None,
    'exit_code': None,
    'output_dir': None,
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


def _append_log(line: str) -> None:
    with _display_lock:
        clean_line = _strip_ansi(line).rstrip('\r\n')
        if not clean_line:
            return
        _display_status['logs'].append(clean_line)
        if len(_display_status['logs']) > 4000:
            _display_status['logs'] = _display_status['logs'][-4000:]


def _read_process_output(proc: subprocess.Popen) -> None:
    try:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            if raw_line is None:
                break
            _append_log(raw_line)
    finally:
        with _display_lock:
            _display_status['running'] = False
            _display_status['finish_time'] = time.time()
            _display_status['exit_code'] = proc.returncode
            _display_status['pid'] = None
            global _display_process
            _display_process = None


def _build_command(payload: dict) -> list[str]:
    script = BASE_DIR / 'scripts' / 'display_inference.py'
    return [sys.executable, '-u', str(script)]


def _serialize_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def start_display(payload: dict):
    global _display_process
    with _display_lock:
        if _display_status['running']:
            return _error('可视化推理已在运行中', 400)
        _display_status['logs'].clear()
        _display_status['exit_code'] = None
        _display_status['finish_time'] = None
        _display_status['start_time'] = time.time()
        _display_status['output_dir'] = payload.get('output_dir')

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
            return _error(f'启动可视化推理失败: {exc}', 500)

        if proc.stdin is not None:
            try:
                proc.stdin.write(_serialize_payload(payload))
            except Exception:
                pass
            finally:
                proc.stdin.close()

        _display_status['running'] = True
        _display_status['pid'] = proc.pid
        _display_process = proc
        if payload.get('output_dir'):
            _display_status['output_dir'] = payload.get('output_dir')
        thread = threading.Thread(target=_read_process_output, args=(proc,), daemon=True)
        thread.start()

    return {'status': 'ok', 'message': '可视化推理已启动', 'pid': proc.pid}


def stop_display():
    global _display_process
    with _display_lock:
        proc = _display_process
        if proc is None or proc.poll() is not None:
            return _error('当前没有正在运行的可视化推理任务', 400)
        if proc.pid is not None:
            _kill_process_tree(proc.pid, force=True)
        _display_status['running'] = False
        _display_status['finish_time'] = time.time()
        _display_status['exit_code'] = proc.returncode
        _display_status['pid'] = None
        _display_process = None
    return {'status': 'ok', 'message': '可视化推理已停止'}


def get_display_status():
    with _display_lock:
        return {
            'running': _display_status['running'],
            'pid': _display_status['pid'],
            'start_time': _display_status['start_time'],
            'finish_time': _display_status['finish_time'],
            'exit_code': _display_status['exit_code'],
            'output_dir': _display_status['output_dir'],
        }


def get_display_logs(offset: int = 0, limit: int = 100):
    with _display_lock:
        logs = list(_display_status['logs'])
    return {
        'offset': min(len(logs), max(0, offset)),
        'logs': logs[offset:offset + limit],
    }
