from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from web.deps.saas_deps import workbench_context
from web.services import metrics_service

router = APIRouter(dependencies=[Depends(workbench_context)])


@router.get("/api/metrics")
def get_metrics(source: str = Query("")):
    return metrics_service.get_metrics(source=source)
