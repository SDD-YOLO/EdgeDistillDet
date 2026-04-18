"""训练与输出检查 API 服务。"""

from __future__ import annotations

from web.schemas import TrainStartRequest
from web.services import backend_logic


def output_check(project: str):
    return backend_logic.output_check(project=project)


def start_training(payload: TrainStartRequest):
    return backend_logic.start_training(payload)


def stop_training():
    return backend_logic.stop_training()


def get_training_status():
    return backend_logic.get_training_status()


def get_resume_candidates(project: str):
    return backend_logic.get_resume_candidates(project=project)


def get_training_logs(offset: int, limit: int):
    return backend_logic.get_training_logs(offset=offset, limit=limit)


def download_training_logs():
    return backend_logic.download_training_logs()


def stream_training_logs(offset: int):
    return backend_logic.stream_training_logs(offset=offset)
