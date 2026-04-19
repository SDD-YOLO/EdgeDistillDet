from __future__ import annotations

from fastapi import APIRouter

from web.core.saas_settings import get_redis_url, saas_enabled

router = APIRouter()


@router.get("/api/health")
def health():
    out: dict = {"status": "ok", "saas": saas_enabled()}
    if saas_enabled():
        try:
            import redis

            r = redis.from_url(get_redis_url(), socket_connect_timeout=1.0)
            r.ping()
            out["redis"] = "ok"
        except Exception as e:
            out["redis"] = f"error: {e}"
        try:
            from web.db.session import get_engine

            get_engine().connect().close()
            out["database"] = "ok"
        except Exception as e:
            out["database"] = f"error: {e}"
    return out
