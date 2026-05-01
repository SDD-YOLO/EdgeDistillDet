from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi.responses import JSONResponse

from core.logging import get_logger
from web.core.paths import BASE_DIR
from web.services.backend_common import _error
from web.services.backend_train_runtime import _kill_process_tree

logger = get_logger(__name__)

SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript"}

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
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f'导出路径不可创建: {parent_dir} ({exc})'
    if not parent_dir.is_dir():
        return f'导出路径不是目录: {parent_dir}'

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
    buffer = ""
    try:
        if proc.stdout is None:
            _append_log('WARNING: 未能捕获导出子进程输出')
            return
        while True:
            char = proc.stdout.read(1)
            if char == "":
                break
            buffer += char
            if char in {"\n", "\r"}:
                line = buffer.rstrip("\r\n")
                buffer = ""
                if line:
                    logger.debug(f'export stdout line: {line!r}')
                    _append_log(line)
                    clean_line = _strip_ansi(line).strip()
                    if clean_line.startswith('INFO: 导出完成:'):
                        try:
                            completed_path = clean_line[len('INFO: 导出完成:'):].strip()
                            if completed_path:
                                with _export_lock:
                                    _export_status['output_path'] = completed_path
                        except Exception:
                            pass
                    elif clean_line.startswith('INFO: 已移动导出文件到'):
                        try:
                            moved_path = clean_line[len('INFO: 已移动导出文件到'):].strip()
                            if moved_path:
                                with _export_lock:
                                    _export_status['output_path'] = moved_path
                        except Exception:
                            pass
        if buffer:
            line = buffer
            logger.debug(f'export stdout final partial: {line!r}')
            _append_log(line)
    except Exception as exc:
        _append_log(f'WARNING: 读取导出子进程输出时出错: {exc}')
        logger.exception('读取导出子进程输出时异常')
    finally:
        with _export_lock:
            _export_status['running'] = False
            _export_status['finish_time'] = time.time()
            _export_status['exit_code'] = proc.returncode
            _export_status['pid'] = None
            global _export_process
            _export_process = None
            _append_log(f'INFO: 导出子进程已退出，exit_code={proc.returncode}')
            logger.info(f'导出任务结束 | returncode={proc.returncode}')
            
            # 清理临时文件
            try:
                for arg in proc.args:
                    if isinstance(arg, str) and arg.endswith('.json') and 'tmp' in arg.lower():
                        os.unlink(arg)
            except Exception:
                logger.exception('清理临时文件时异常')

def _build_command(payload: dict) -> list[str]:
    script = BASE_DIR / 'scripts' / 'export_model.py'
    # 写入临时 JSON 文件，用 --config-file 传递
    tmp = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.json', 
        delete=False, 
        encoding='utf-8',
        dir=str(BASE_DIR)  # 放在项目目录，避免权限问题
    )
    json.dump(payload, tmp, ensure_ascii=False)
    tmp.close()
    return [sys.executable, '-u', str(script), '--config-file', tmp.name]


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
        resolved_path = None
        if payload.get('export_path'):
            try:
                resolved_target = _resolve_path(str(payload.get('export_path')))
                resolved_path = str(resolved_target) if resolved_target is not None else str(payload.get('export_path'))
            except Exception:
                resolved_path = str(payload.get('export_path'))
        _export_status['output_path'] = resolved_path

        command = _build_command(payload)
        try:
            proc = subprocess.Popen(
                command,
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

        # 删除 stdin 写入代码，改用 --config-file
        
        _export_status['running'] = True
        _export_status['pid'] = proc.pid
        _export_process = proc
        _append_log(f'INFO: 导出子进程已启动，PID={proc.pid}')
        logger.info(f'导出任务已启动 | pid={proc.pid} format={payload.get("format")} export_path={resolved_path}')
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
