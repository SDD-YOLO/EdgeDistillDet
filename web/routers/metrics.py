from __future__ import annotations

from fastapi import APIRouter, Query

from web.services import metrics_service

router = APIRouter()


@router.get("/api/metrics")
def get_metrics(source: str = Query("")):
    return metrics_service.get_metrics(source=source)
