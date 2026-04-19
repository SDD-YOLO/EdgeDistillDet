from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from web.core.saas_settings import saas_enabled
from web.db.models import User
from web.db.session import get_db
from web.deps.saas_deps import require_user_id
from web.security.jwt_tokens import create_tokens, decode_token_safe
from web.security.password import hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class RefreshBody(BaseModel):
    refresh_token: str


def _require_saas():
    if not saas_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="SaaS 未启用")


@router.post("/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    _require_saas()
    existing = db.scalars(select(User).where(User.email == body.email.lower())).first()
    if existing:
        raise HTTPException(status_code=400, detail="邮箱已注册")
    u = User(email=body.email.lower(), hashed_password=hash_password(body.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    tokens = create_tokens(u.id)
    return {"status": "ok", "user": {"id": u.id, "email": u.email}, **tokens}


@router.post("/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    _require_saas()
    u = db.scalars(select(User).where(User.email == body.email.lower())).first()
    if not u or not u.hashed_password or not verify_password(body.password, u.hashed_password):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    if not u.is_active:
        raise HTTPException(status_code=403, detail="账户已禁用")
    tokens = create_tokens(u.id)
    return {"status": "ok", "user": {"id": u.id, "email": u.email}, **tokens}


@router.post("/refresh")
def refresh_token(body: RefreshBody):
    _require_saas()
    payload = decode_token_safe(body.refresh_token)
    if not payload or payload.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="刷新令牌无效")
    uid = str(payload.get("sub") or "")
    if not uid:
        raise HTTPException(status_code=401, detail="刷新令牌无效")
    tokens = create_tokens(uid)
    return {"status": "ok", **tokens}


@router.get("/me")
def me(
    db: Session = Depends(get_db),
    user_id: Annotated[str, Depends(require_user_id)] = "",
):
    _require_saas()
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"status": "ok", "user": {"id": u.id, "email": u.email}}
