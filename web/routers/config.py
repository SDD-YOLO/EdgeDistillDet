from __future__ import annotations

from fastapi import APIRouter, Depends

from web.deps.saas_deps import workbench_context
from web.schemas import DialogPickRequest, SaveConfigRequest, UploadConfigRequest
from web.services import config_api_service

router = APIRouter(dependencies=[Depends(workbench_context)])


@router.get("/api/configs")
def get_configs():
    return config_api_service.get_configs()


@router.get("/api/config/{config_name}")
def get_config(config_name: str):
    return config_api_service.get_config(config_name)


@router.get("/api/config/recent")
def get_recent_config():
    return config_api_service.get_recent_config()


@router.post("/api/config/save")
def save_config(payload: SaveConfigRequest):
    return config_api_service.save_config(payload)


@router.post("/api/config/upload")
def upload_config(payload: UploadConfigRequest):
    return config_api_service.upload_config(payload)


@router.post("/api/dialog/pick")
def pick_path_dialog(payload: DialogPickRequest):
    return config_api_service.pick_path_dialog(payload)
