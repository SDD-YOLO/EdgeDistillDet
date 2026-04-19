"""SaaS / 多租户相关环境变量。"""

from __future__ import annotations

import os
from functools import lru_cache


def saas_enabled() -> bool:
    return os.environ.get("EDGE_SAAS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def get_database_url() -> str:
    url = os.environ.get("EDGE_DATABASE_URL", "").strip()
    if url:
        return url
    # 本地开发默认 SQLite
    return "sqlite:///./.saas_data/saas.db"


def get_redis_url() -> str:
    return os.environ.get("EDGE_REDIS_URL", "redis://127.0.0.1:6379/0").strip()


def get_jwt_secret() -> str:
    s = os.environ.get("EDGE_JWT_SECRET", "").strip()
    if not s:
        # 开发回退；生产必须设置
        return "dev-only-change-in-production-edge-jwt-secret"
    return s


def get_team_data_root() -> str:
    root = os.environ.get("EDGE_TEAM_DATA_ROOT", "").strip()
    if root:
        return root
    from web.core.paths import BASE_DIR

    return str(BASE_DIR / ".team_data")


def get_jwt_algorithm() -> str:
    return os.environ.get("EDGE_JWT_ALGORITHM", "HS256").strip() or "HS256"


def get_access_token_expire_minutes() -> int:
    return int(os.environ.get("EDGE_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def get_refresh_token_expire_days() -> int:
    return int(os.environ.get("EDGE_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


@lru_cache
def get_oauth_github_settings() -> tuple[str | None, str | None]:
    cid = os.environ.get("EDGE_OAUTH_GITHUB_CLIENT_ID", "").strip() or None
    sec = os.environ.get("EDGE_OAUTH_GITHUB_CLIENT_SECRET", "").strip() or None
    return cid, sec


def get_oauth_github_redirect_uri() -> str | None:
    u = os.environ.get("EDGE_OAUTH_GITHUB_REDIRECT_URI", "").strip()
    return u or None


def get_public_app_url() -> str:
    """前端/API 对外基础 URL（OAuth redirect 等）。"""
    return os.environ.get("EDGE_PUBLIC_APP_URL", "http://127.0.0.1:5000").rstrip("/")


def use_training_queue() -> bool:
    """SaaS 下是否将训练提交到 Redis 队列由 Worker 执行。"""
    if not saas_enabled():
        return False
    return os.environ.get("EDGE_USE_TRAINING_QUEUE", "1").strip().lower() not in {"0", "false", "no", "off"}
