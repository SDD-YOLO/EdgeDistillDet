"""GitHub OAuth（可选配置）。"""

from __future__ import annotations

import urllib.parse

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from web.core.saas_settings import get_oauth_github_redirect_uri, get_oauth_github_settings, get_public_app_url, saas_enabled
from web.db.models import OAuthAccount, User
from web.db.session import get_db
from web.security.jwt_tokens import create_tokens

router = APIRouter(prefix="/api/oauth", tags=["oauth"])


def _require_saas():
    if not saas_enabled():
        raise HTTPException(status_code=503, detail="SaaS 未启用")


@router.get("/github/login")
def github_login():
    _require_saas()
    cid, _ = get_oauth_github_settings()
    if not cid:
        raise HTTPException(status_code=503, detail="未配置 EDGE_OAUTH_GITHUB_CLIENT_ID")
    redirect = get_oauth_github_redirect_uri() or f"{get_public_app_url()}/api/oauth/github/callback"
    q = urllib.parse.urlencode(
        {"client_id": cid, "redirect_uri": redirect, "scope": "read:user user:email"}
    )
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{q}")


@router.get("/github/callback")
def github_callback(code: str, db: Session = Depends(get_db)):
    _require_saas()
    cid, secret = get_oauth_github_settings()
    redirect = get_oauth_github_redirect_uri() or f"{get_public_app_url()}/api/oauth/github/callback"
    if not cid or not secret:
        raise HTTPException(status_code=503, detail="OAuth 未完整配置")
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": cid,
                "client_secret": secret,
                "code": code,
                "redirect_uri": redirect,
            },
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        tok = r.json()
        access = str(tok.get("access_token") or "")
        if not access:
            raise HTTPException(status_code=400, detail="GitHub 未返回 access_token")
        u = client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access}", "Accept": "application/vnd.github+json"},
        )
        u.raise_for_status()
        profile = u.json()
        gh_id = str(profile.get("id") or "")
        email = str(profile.get("email") or "") or None
        if not email:
            er = client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access}", "Accept": "application/vnd.github+json"},
            )
            if er.is_success:
                for item in er.json():
                    if item.get("primary") and item.get("email"):
                        email = item["email"]
                        break
        if not email:
            raise HTTPException(status_code=400, detail="无法获取 GitHub 邮箱，请在 GitHub 公开邮箱或授权 email 权限")

    oa = db.scalars(
        select(OAuthAccount).where(
            OAuthAccount.provider == "github",
            OAuthAccount.provider_user_id == gh_id,
        )
    ).first()
    if oa:
        user = db.get(User, oa.user_id)
        if not user:
            raise HTTPException(status_code=500, detail="账户数据不一致")
    else:
        user = db.scalars(select(User).where(User.email == email.lower())).first()
        if not user:
            user = User(email=email.lower(), hashed_password=None)
            db.add(user)
            db.flush()
        oa = OAuthAccount(user_id=user.id, provider="github", provider_user_id=gh_id, email=email.lower())
        db.add(oa)
    db.commit()
    db.refresh(user)
    tokens = create_tokens(user.id)
    return {"status": "ok", "user": {"id": user.id, "email": user.email}, **tokens}
