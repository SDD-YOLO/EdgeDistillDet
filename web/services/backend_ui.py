from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse

from web.core.paths import STATIC_DIR, TEMPLATE_FILE
from web.services.backend_common import _error

_DEBUG_LOG_PATH = STATIC_DIR.parent.parent / "debug-87ccac.log"


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    # region agent log
    try:
        payload = {
            "sessionId": "87ccac",
            "runId": str(run_id),
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

def favicon():
    favicon_path = STATIC_DIR / 'favicon.ico'
    if not favicon_path.exists():
        return _error('favicon 不存在', 404)
    return FileResponse(str(favicon_path))

def index():
    """读取 HTML 并手动替换 Jinja2 url_for（避免模板查找问题）"""
    with open(str(TEMPLATE_FILE), 'r', encoding='utf-8') as f:
        html = f.read()

    def replace_static_url(m):
        fname = m.group(1)
        return '/static/' + fname

    html = re.sub(
        r"\{\{\s*url_for\(\s*'static'\s*,\s*filename\s*=\s*'([^']+)'\s*\)\s*\}\}",
        replace_static_url,
        html
    )
    html = re.sub(
        r'\{\{\s*url_for\(\s*"static"\s*,\s*filename\s*=\s*"([^"]+)"\s*\)\s*\}\}',
        replace_static_url,
        html
    )

    # 为 dist 前端资源追加版本号，避免浏览器长期缓存旧 app.js（界面已更新仍看到旧 DOM）
    _dist_css = STATIC_DIR / 'dist' / 'app.css'
    _dist_js = STATIC_DIR / 'dist' / 'app.js'
    _v_css = int(_dist_css.stat().st_mtime) if _dist_css.is_file() else 0
    _v_js = int(_dist_js.stat().st_mtime) if _dist_js.is_file() else 0
    # region agent log
    _debug_log(
        run_id="repro-2",
        hypothesis_id="H2",
        location="web/services/backend_ui.py:index",
        message="index serves static dist bundle",
        data={"css_exists": _dist_css.is_file(), "js_exists": _dist_js.is_file(), "v_js": _v_js},
    )
    # endregion
    # region agent log
    try:
        bundle_text = _dist_js.read_text(encoding="utf-8", errors="ignore") if _dist_js.is_file() else ""
        _debug_log(
            run_id="repro-2",
            hypothesis_id="H1",
            location="web/services/backend_ui.py:index",
            message="bundle label tokens",
            data={
                "has_old_label": "预热轮数" in bundle_text,
                "has_target_label": "训练预热轮数" in bundle_text,
                "has_lr_label": "学习率预热轮数" in bundle_text,
            },
        )
    except Exception as exc:
        _debug_log(
            run_id="repro-2",
            hypothesis_id="H4",
            location="web/services/backend_ui.py:index",
            message="bundle read failed",
            data={"error": str(exc)},
        )
    # endregion
    html = html.replace('href="/static/dist/app.css"', f'href="/static/dist/app.css?v={_v_css}"')
    html = html.replace('src="/static/dist/app.js"', f'src="/static/dist/app.js?v={_v_js}"')

    return HTMLResponse(
        content=html,
        headers={'Cache-Control': 'no-store'},
    )
