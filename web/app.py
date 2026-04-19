"""
EdgeDistillDet Local UI
=======================
仅负责应用装配（FastAPI 初始化、挂载静态资源、注册路由、启动入口）。

使用方法: python web/app.py
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 兼容 `python web/app.py` 直接启动
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from web.core.paths import STATIC_DIR, TEMPLATE_FILE
from web.core.saas_settings import saas_enabled
from web.core.settings import get_bind_host, get_bind_port, get_cors_middleware_kwargs
from web.routers.agent import router as agent_router
from web.routers.auth_api import router as auth_router
from web.routers.config import router as config_router
from web.routers.datasets_api import router as datasets_router
from web.routers.health import router as health_router
from web.routers.hub_api import router as hub_router
from web.routers.metrics import router as metrics_router
from web.routers.oauth_api import router as oauth_router
from web.routers.teams_api import router as teams_router
from web.routers.train import router as train_router
from web.routers.ui import router as ui_router

from main import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    if saas_enabled():
        from pathlib import Path

        from web.core.saas_settings import get_database_url

        url = get_database_url()
        if url.startswith("sqlite"):
            # 确保 SQLite 父目录存在
            rest = url.replace("sqlite:///", "", 1)
            if rest and not rest.startswith(":"):
                Path(rest).parent.mkdir(parents=True, exist_ok=True)
        from web.db.session import init_db

        init_db()
        from web.core.saas_settings import get_team_data_root

        Path(get_team_data_root()).mkdir(parents=True, exist_ok=True)
    yield


api = FastAPI(title="EdgeDistillDet Backend", version=__version__, lifespan=lifespan)
api.add_middleware(CORSMiddleware, **get_cors_middleware_kwargs())

api.include_router(health_router)
api.include_router(ui_router)
api.include_router(config_router)
api.include_router(train_router)
api.include_router(metrics_router)
api.include_router(agent_router)
api.include_router(auth_router)
api.include_router(teams_router)
api.include_router(datasets_router)
api.include_router(hub_router)
api.include_router(oauth_router)


@api.get("/api/version")
def api_version():
    return {"name": "edgedistilldet", "version": __version__, "saas": saas_enabled()}


api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    host = get_bind_host()
    port = get_bind_port()
    print("=" * 60)
    print("  EdgeDistillDet Local UI")
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  SaaS     : {saas_enabled()}")
    print(f"  Template : {TEMPLATE_FILE} (exists: {TEMPLATE_FILE.exists()})")
    print(f"  Listen   : {host}:{port}")
    if host == "0.0.0.0":
        print("  [安全] 已监听所有网卡；请仅在可信局域网使用，或改回 EDGE_BACKEND_HOST=127.0.0.1")
    print(f"  Open     : http://127.0.0.1:{port}")
    print("=" * 60)

    if not TEMPLATE_FILE.exists():
        print(f"[FATAL] Template file not found: {TEMPLATE_FILE}")
        sys.exit(1)

    uvicorn.run(api, host=host, port=port, log_level="info")
