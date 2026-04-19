from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from web.core.saas_settings import saas_enabled
from web.db.models import Team, TeamInvitation, TeamMembership, TeamRole, User
from web.db.session import get_db
from web.deps.saas_deps import require_user_id
from web.saas.team_fs import ensure_team_workspace

router = APIRouter(prefix="/api/teams", tags=["teams"])


def _require_saas():
    if not saas_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="SaaS 未启用")


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", name.strip().lower())
    s = s.strip("-")[:80]
    return s or "team"


class CreateTeamBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class InviteBody(BaseModel):
    email: EmailStr
    role: str = TeamRole.member.value


class AcceptInviteBody(BaseModel):
    token: str = Field(min_length=10)


@router.post("")
def create_team(
    body: CreateTeamBody,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    base = _slugify(body.name)
    slug = base
    n = 0
    while db.scalars(select(Team).where(Team.slug == slug)).first():
        n += 1
        slug = f"{base}-{n}"
    team = Team(name=body.name.strip(), slug=slug, owner_id=user_id)
    db.add(team)
    db.flush()
    db.add(TeamMembership(user_id=user_id, team_id=team.id, role=TeamRole.owner.value))
    db.commit()
    ensure_team_workspace(team.id)
    return {"status": "ok", "team": {"id": team.id, "name": team.name, "slug": team.slug}}


@router.get("")
def list_teams(
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    memberships = db.scalars(select(TeamMembership).where(TeamMembership.user_id == user_id)).all()
    out = []
    for mm in memberships:
        t = db.get(Team, mm.team_id)
        if t:
            out.append({"id": t.id, "name": t.name, "slug": t.slug, "role": mm.role})
    return {"status": "ok", "teams": out}


@router.get("/{team_id}/members")
def list_members(
    team_id: str,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    m = db.scalars(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    ).first()
    if not m:
        raise HTTPException(status_code=403, detail="无权访问")
    ms = db.scalars(select(TeamMembership).where(TeamMembership.team_id == team_id)).all()
    users_out = []
    for mm in ms:
        u = db.get(User, mm.user_id)
        users_out.append(
            {"user_id": mm.user_id, "email": u.email if u else "", "role": mm.role}
        )
    return {"status": "ok", "members": users_out}


@router.post("/{team_id}/invitations")
def create_invitation(
    team_id: str,
    body: InviteBody,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    m = db.scalars(
        select(TeamMembership).where(
            TeamMembership.team_id == team_id,
            TeamMembership.user_id == user_id,
        )
    ).first()
    if not m or m.role not in (TeamRole.owner.value, TeamRole.admin.value):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    raw = secrets.token_urlsafe(32)
    th = hashlib.sha256(raw.encode()).hexdigest()
    inv = TeamInvitation(
        team_id=team_id,
        email=body.email.lower(),
        token_hash=th,
        role=body.role if body.role in {r.value for r in TeamRole} else TeamRole.member.value,
        expires_at=datetime.utcnow() + timedelta(days=7),
    )
    db.add(inv)
    db.commit()
    return {"status": "ok", "token": raw, "expires_in_days": 7}


@router.post("/invitations/accept")
def accept_invitation(
    body: AcceptInviteBody,
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    th = hashlib.sha256(body.token.encode()).hexdigest()
    inv = db.scalars(select(TeamInvitation).where(TeamInvitation.token_hash == th)).first()
    if not inv or inv.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="邀请无效或已过期")
    u = db.get(User, user_id)
    if not u or u.email.lower() != inv.email.lower():
        raise HTTPException(status_code=403, detail="请使用被邀请的邮箱登录")
    exists = db.scalars(
        select(TeamMembership).where(
            TeamMembership.team_id == inv.team_id,
            TeamMembership.user_id == user_id,
        )
    ).first()
    if not exists:
        db.add(TeamMembership(user_id=user_id, team_id=inv.team_id, role=inv.role))
    db.delete(inv)
    db.commit()
    return {"status": "ok", "team_id": inv.team_id}
