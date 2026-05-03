from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from web.schemas import DisplayStartRequest, ExportStartRequest, TrainStartRequest
from web.services import display_service, export_service, train_service

router = APIRouter()


@router.get("/api/output/check")
def output_check(project: str = Query("runs")):
    return train_service.output_check(project=project)


@router.post("/api/train/start")
def start_training(payload: TrainStartRequest):
    return train_service.start_training(payload)


@router.post("/api/train/stop")
def stop_training():
    return train_service.stop_training()


@router.get("/api/train/status")
def get_training_status():
    return train_service.get_training_status()


@router.get("/api/train/resume_candidates")
def get_resume_candidates(project: str = Query("runs")):
    return train_service.get_resume_candidates(project=project)


@router.get("/api/train/export_weight_candidates")
def get_export_weight_candidates(project: str = Query("runs")):
    return train_service.fetch_export_weight_candidates(project=project)


@router.get("/api/train/logs")
def get_training_logs(offset: int = Query(0), limit: int = Query(100)):
    return train_service.get_training_logs(offset=offset, limit=limit)


@router.get("/api/train/logs/download")
def download_training_logs():
    return train_service.download_training_logs()


@router.post("/api/display/start")
def start_display(payload: DisplayStartRequest):
    return display_service.start_display(payload)


@router.post("/api/display/stop")
def stop_display():
    return display_service.stop_display()


@router.get("/api/display/status")
def get_display_status():
    return display_service.get_display_status()


@router.get("/api/display/logs")
def get_display_logs(offset: int = Query(0), limit: int = Query(100)):
    return display_service.get_display_logs(offset=offset, limit=limit)


@router.post("/api/export/start")
def start_export(payload: ExportStartRequest):
    return export_service.start_export(payload)


@router.get("/api/export/start")
def start_export_help():
    return JSONResponse(
        status_code=405,
        content={"detail": "Use POST to /api/export/start with JSON body. Example: {'format':'onnx','export_path':'./test.onnx','weight':'models/yolov8n.pt'}"},
    )


@router.post("/api/export/stop")
def stop_export():
    return export_service.stop_export()


@router.get("/api/export/stop")
def stop_export_help():
    return JSONResponse(
        status_code=405,
        content={"detail": "Use POST to /api/export/stop to stop the running export task."},
    )


@router.get("/api/export/status")
def get_export_status():
    return export_service.get_export_status()


@router.get("/api/export/logs")
def get_export_logs(offset: int = Query(0), limit: int = Query(100)):
    return export_service.get_export_logs(offset=offset, limit=limit)


@router.get("/api/train/logs/stream")
def stream_training_logs(offset: int = Query(0)):
    return train_service.stream_training_logs(offset=offset)
