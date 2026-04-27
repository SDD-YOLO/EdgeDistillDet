"""Display inference service wrappers."""

from __future__ import annotations

from web.schemas import DisplayStartRequest
from web.services import backend_display


def start_display(payload: DisplayStartRequest):
    return backend_display.start_display(payload.dict())


def stop_display():
    return backend_display.stop_display()


def get_display_status():
    return backend_display.get_display_status()


def get_display_logs(offset: int, limit: int):
    return backend_display.get_display_logs(offset, limit)
