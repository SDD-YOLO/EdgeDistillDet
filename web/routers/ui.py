from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from web.core.paths import STATIC_DIR
from web.services import backend_logic

router = APIRouter()


def _error(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={"error": message})


@router.get("/static/dist/app.js")
def serve_dist_app_js():
    path = STATIC_DIR / "dist" / "app.js"
    if not path.is_file():
        return _error("app.js 不存在", 404)
    return FileResponse(
        str(path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@router.get("/static/dist/app.css")
def serve_dist_app_css():
    path = STATIC_DIR / "dist" / "app.css"
    if not path.is_file():
        return _error("app.css 不存在", 404)
    return FileResponse(
        str(path),
        media_type="text/css",
        headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
    )


@router.get("/favicon.ico")
def favicon():
    return backend_logic.favicon()


@router.get("/")
def index():
    return backend_logic.index()
