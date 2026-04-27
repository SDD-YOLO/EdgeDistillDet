"""Model export service wrappers."""

from __future__ import annotations

from web.schemas import ExportStartRequest
from web.services import backend_export


def start_export(payload: ExportStartRequest):
    return backend_export.start_export(payload.dict())


def stop_export():
    return backend_export.stop_export()


def get_export_status():
    return backend_export.get_export_status()


def get_export_logs(offset: int, limit: int):
    return backend_export.get_export_logs(offset, limit)
