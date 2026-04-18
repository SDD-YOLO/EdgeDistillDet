"""配置与本地文件选择相关 API 服务。"""

from __future__ import annotations

from web.schemas import DialogPickRequest, SaveConfigRequest, UploadConfigRequest
from web.services import backend_logic


def get_configs():
    return backend_logic.get_configs()


def get_config(config_name: str):
    return backend_logic.get_config(config_name)


def get_recent_config():
    return backend_logic.get_recent_config()


def save_config(payload: SaveConfigRequest):
    return backend_logic.save_config(payload)


def upload_config(payload: UploadConfigRequest):
    return backend_logic.upload_config(payload)


def pick_path_dialog(payload: DialogPickRequest):
    return backend_logic.pick_path_dialog(payload)
