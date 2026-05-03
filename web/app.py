"""
EdgeDistillDet Local UI
=======================
仅负责应用装配（FastAPI 初始化、挂载静态资源、注册路由、启动入口）。

使用方法: python web/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.logging import get_logger, init_logging
from main import __version__
from web.core.paths import STATIC_DIR, TEMPLATE_FILE
from web.core.settings import get_bind_host, get_bind_port, get_cors_middleware_kwargs
from web.routers.agent import router as agent_router
from web.routers.config import router as config_router
from web.routers.metrics import router as metrics_router
from web.routers.train import router as train_router
from web.routers.ui import router as ui_router
from web.routers.ws import router as ws_router
from web.schemas import ResponseModel
from web.services.ws_manager import manager as ws_manager

init_logging()
# 兼容 `python web/app.py` 直接启动
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

app_logger = get_logger("edgedistilldet.app")

api = FastAPI(title="EdgeDistillDet Backend", version=__version__)
api.add_middleware(CORSMiddleware, **get_cors_middleware_kwargs())


# Simple request logging middleware
@api.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception:
        app_logger.exception("Unhandled exception while processing request")
        raise
    finally:
        elapsed = time.time() - start
        app_logger.info(f"{request.method} {request.url.path} completed_in={elapsed:.3f}s")


# Global exception handlers
@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    content = ResponseModel(ok=False, error="validation_error", meta={"errors": exc.errors()})
    return JSONResponse(status_code=422, content=content.model_dump())


@api.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    app_logger.exception("Unhandled exception")
    content = ResponseModel(ok=False, error=str(exc))
    return JSONResponse(status_code=500, content=content.model_dump())


@api.get("/api/version")
def api_version():
    return {"name": "edgedistilldet", "version": __version__}


api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

api.include_router(ui_router)
api.include_router(config_router)
api.include_router(train_router)
api.include_router(metrics_router)
api.include_router(agent_router)
api.include_router(ws_router)


# Start and stop the websocket manager with the app lifecycle
api.add_event_handler("startup", ws_manager.start)
api.add_event_handler("shutdown", ws_manager.stop)


if __name__ == "__main__":
    host = get_bind_host()
    port = get_bind_port()
    app_logger.info(f"EdgeDistillDet web UI starting | base_dir={BASE_DIR} template={TEMPLATE_FILE} template_exists={TEMPLATE_FILE.exists()} bind={host}:{port}")
    if host == "0.0.0.0":
        app_logger.warning("Web backend is listening on all interfaces; use only on a trusted LAN or set EDGE_BACKEND_HOST=127.0.0.1")

    if not TEMPLATE_FILE.exists():
        app_logger.error(f"Template file not found: {TEMPLATE_FILE}")
        sys.exit(1)

    uvicorn.run(api, host=host, port=port, log_level="info", log_config=None)
