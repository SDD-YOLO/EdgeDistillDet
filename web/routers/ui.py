from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from web.core.paths import STATIC_DIR
from web.services import backend_logic

router = APIRouter()
_DEBUG_LOG_PATH = STATIC_DIR.parent.parent / "debug-87ccac.log"


def _debug_log(*, hypothesis_id: str, location: str, message: str, data: dict):
    # region agent log
    try:
        payload = {
            "sessionId": "87ccac",
            "runId": "repro-1",
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "message": str(message),
            "data": data if isinstance(data, dict) else {},
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


def _error(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={"error": message})


@router.get("/static/dist/app.js")
def serve_dist_app_js():
    path = STATIC_DIR / "dist" / "app.js"
    # region agent log
    _debug_log(
        hypothesis_id="H2",
        location="web/routers/ui.py:serve_dist_app_js",
        message="dist app.js requested",
        data={"exists": path.is_file(), "path": str(path)},
    )
    # endregion
    if not path.is_file():
        return _error("app.js 不存在", 404)
    # region agent log
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        _debug_log(
            hypothesis_id="H1",
            location="web/routers/ui.py:serve_dist_app_js",
            message="label tokens in bundled app.js",
            data={
                "has_old_label": "预热轮数" in text,
                "has_target_label": "训练预热轮数" in text,
                "has_lr_warmup_label": "学习率预热轮数" in text,
            },
        )
    except Exception as exc:
        _debug_log(
            hypothesis_id="H3",
            location="web/routers/ui.py:serve_dist_app_js",
            message="failed to inspect app.js label tokens",
            data={"error": str(exc)},
        )
    # endregion
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
