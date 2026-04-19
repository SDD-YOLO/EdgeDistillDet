"""Run 列表、API Key、审计日志。"""

from __future__ import annotations

import hashlib
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from web.core.saas_settings import saas_enabled
from web.core.workspace import get_current_team_id
from web.db.models import ApiKey, AuditLog, TrainingJob
from web.db.session import get_db
from web.deps.saas_deps import require_user_id, workbench_context

router = APIRouter(prefix="/api/hub", tags=["hub"], dependencies=[Depends(workbench_context)])


def _require_saas():
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="SaaS 未启用")


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@router.get("/runs")
def list_runs(
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    tid = get_current_team_id()
    if not tid:
        raise HTTPException(status_code=400, detail="无团队上下文")
    rows = db.scalars(
        select(TrainingJob).where(TrainingJob.team_id == str(tid)).order_by(TrainingJob.created_at.desc()).limit(100)
    ).all()
    return {
        "status": "ok",
        "runs": [
            {
                "id": j.id,
                "status": j.status,
                "config": j.config_name,
                "mode": j.mode,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
                "error": j.error_message,
            }
            for j in rows
        ],
    }


@router.post("/api-keys")
def create_api_key(
    body: ApiKeyCreate,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    tid = get_current_team_id()
    if not tid:
        raise HTTPException(status_code=400, detail="无团队上下文")
    raw = "edd_" + secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode()).hexdigest()
    prefix = raw[:12]
    row = ApiKey(team_id=str(tid), user_id=user_id, name=body.name, key_prefix=prefix, key_hash=h)
    db.add(row)
    db.add(AuditLog(team_id=str(tid), user_id=user_id, action="api_key.create", detail={"name": body.name}))
    db.commit()
    db.refresh(row)
    return {"status": "ok", "id": row.id, "key": raw, "warning": "请立即保存，此密钥仅显示一次"}


@router.get("/audit")
def list_audit(
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    tid = get_current_team_id()
    if not tid:
        raise HTTPException(status_code=400, detail="无团队上下文")
    rows = db.scalars(
        select(AuditLog).where(AuditLog.team_id == str(tid)).order_by(AuditLog.created_at.desc()).limit(200)
    ).all()
    return {
        "status": "ok",
        "entries": [
            {"id": a.id, "action": a.action, "detail": a.detail, "created_at": a.created_at.isoformat() if a.created_at else None}
            for a in rows
        ],
    }

