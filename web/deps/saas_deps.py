from __future__ import annotations

from collections.abc import Generator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select

from web.core.request_context import current_user_id as uid_ctx
from web.core.saas_settings import saas_enabled
from web.core.workspace import reset_workspace, set_workspace
from web.db.models import TeamMembership
from web.db.session import SessionLocal, get_engine
from web.security.jwt_tokens import decode_token_safe


def optional_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    if not saas_enabled() or not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token_safe(token)
    if not payload or payload.get("typ") != "access":
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


def require_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """非 SaaS 模式返回空字符串（供内部占位）；SaaS 下必须有效 access token。"""
    if not saas_enabled():
        return ""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token_safe(token)
    if not payload or payload.get("typ") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效")
    return str(sub)


def get_current_user_id(
    uid: Annotated[str, Depends(require_user_id)],
) -> str:
    return uid


def workbench_context(
    authorization: Annotated[str | None, Header()] = None,
    x_team_id: Annotated[str | None, Header(alias="X-Team-Id")] = None,
) -> Generator[None, None, None]:
    """为训练台 API 绑定团队工作区（configs/runs/agent）。"""
    if not saas_enabled():
        yield
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token_safe(token)
    if not payload or payload.get("typ") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌无效")
    user_id = str(payload.get("sub") or "")
    if not x_team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少请求头 X-Team-Id")
    try:
        UUID(x_team_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-Team-Id 无效") from e
    get_engine()
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        m = db.scalars(
            select(TeamMembership).where(
                TeamMembership.user_id == user_id,
                TeamMembership.team_id == x_team_id,
            )
        ).first()
        if not m:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该团队")
        from web.saas.team_fs import ensure_team_workspace, team_root_path

        ensure_team_workspace(x_team_id)
        root = team_root_path(x_team_id)
        t1, t2 = set_workspace(UUID(x_team_id), root)
        ut = uid_ctx.set(user_id)
        try:
            yield
        finally:
            uid_ctx.reset(ut)
            reset_workspace(t1, t2)
    finally:
        db.close()
