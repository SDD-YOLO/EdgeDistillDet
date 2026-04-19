from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from web.core.saas_settings import saas_enabled
from web.db.models import DatasetRecord
from web.db.session import get_db
from web.deps.saas_deps import require_user_id, workbench_context
from web.core.workspace import get_current_team_id

router = APIRouter(prefix="/api/datasets", tags=["datasets"], dependencies=[Depends(workbench_context)])


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    storage_subpath: str = ""


def _require_saas():
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="SaaS 未启用")


@router.get("")
def list_datasets(
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    tid = get_current_team_id()
    if not tid:
        raise HTTPException(status_code=400, detail="无团队上下文")
    rows = db.scalars(select(DatasetRecord).where(DatasetRecord.team_id == str(tid))).all()
    return {
        "status": "ok",
        "datasets": [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "storage_subpath": r.storage_subpath,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.post("")
def create_dataset(
    body: DatasetCreate,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    tid = get_current_team_id()
    if not tid:
        raise HTTPException(status_code=400, detail="无团队上下文")
    sub = body.storage_subpath.strip() or f"ds-{uuid.uuid4().hex[:8]}"
    r = DatasetRecord(team_id=str(tid), name=body.name, description=body.description, storage_subpath=sub)
    db.add(r)
    db.commit()
    db.refresh(r)
    from web.saas.team_fs import team_root_path

    root = team_root_path(str(tid))
    (root / "datasets" / sub).mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "dataset": {"id": r.id, "name": r.name, "storage_subpath": r.storage_subpath}}
