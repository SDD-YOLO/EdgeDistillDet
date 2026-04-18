from __future__ import annotations

from fastapi import APIRouter, Query

from web.schemas import TrainStartRequest
from web.services import train_service

router = APIRouter()


@router.get("/api/output/check")
def output_check(project: str = Query("runs/distill")):
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
def get_resume_candidates(project: str = Query("runs/distill")):
    return train_service.get_resume_candidates(project=project)


@router.get("/api/train/logs")
def get_training_logs(offset: int = Query(0), limit: int = Query(100)):
    return train_service.get_training_logs(offset=offset, limit=limit)


@router.get("/api/train/logs/download")
def download_training_logs():
    return train_service.download_training_logs()


@router.get("/api/train/logs/stream")
def stream_training_logs(offset: int = Query(0)):
    return train_service.stream_training_logs(offset=offset)
