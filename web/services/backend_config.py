from __future__ import annotations

from pathlib import Path

import yaml

from web.core.paths import CONFIG_DIR
from web.schemas import DialogPickRequest, SaveConfigRequest, UploadConfigRequest
from web.services import backend_state, config_service
from web.services.backend_common import _error

def get_configs():
    return {'status': 'ok', 'configs': config_service.list_config_names(CONFIG_DIR)}

def get_config(config_name):
    config_path = CONFIG_DIR / config_name
    config = config_service.load_config(config_path)
    if config is None:
        return _error(f'配置文件不存在: {config_name}', 404)
    try:
        file_mtime_ns = int(config_path.stat().st_mtime_ns)
    except OSError:
        file_mtime_ns = 0
    return {'status': 'ok', 'config': config, 'file_mtime_ns': file_mtime_ns}

def get_recent_config():
    
    if backend_state.last_saved_config is not None:
        return {'status': 'ok', 'name': backend_state.last_saved_config['name'], 'config': backend_state.last_saved_config['config']}
    payload = config_service.get_recent_or_default(CONFIG_DIR)
    return {'status': 'ok', 'name': payload['name'], 'config': payload['config']}

def save_config(payload: SaveConfigRequest):
    
    name = payload.name
    config = payload.config
    if not isinstance(name, str) or not isinstance(config, dict):
        return _error('请求格式错误', 400)
    name, file_mtime_ns = config_service.save_config(CONFIG_DIR, name, config)
    backend_state.last_saved_config = {'name': name, 'config': config}
    return {'status': 'ok', 'message': f'配置已保存: {name}', 'file_mtime_ns': file_mtime_ns}

def upload_config(payload: UploadConfigRequest):
    content = payload.content
    if not isinstance(content, str):
        return _error('请求格式错误', 400)

    try:
        config = config_service.parse_uploaded_yaml(content)
        return {'status': 'ok', 'config': config}
    except yaml.YAMLError as exc:
        return _error(f'YAML 解析失败: {exc}', 400)
    except ValueError as exc:
        return _error(str(exc), 400)

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
