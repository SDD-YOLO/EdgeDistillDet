from __future__ import annotations

import re

from fastapi.responses import FileResponse, HTMLResponse

from web.core.paths import STATIC_DIR, TEMPLATE_FILE
from web.services.backend_common import _error


def favicon():
    # 优先使用新的 png 图标，兼容浏览器默认请求 /favicon.ico
    favicon_path = STATIC_DIR / 'favicon.png'
    media_type = 'image/png'
    if not favicon_path.exists():
        favicon_path = STATIC_DIR / 'favicon.ico'
        media_type = 'image/x-icon'
    if not favicon_path.exists():
        return _error('favicon 不存在', 404)
    return FileResponse(str(favicon_path), media_type=media_type)

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
    html = html.replace('href="/static/dist/app.css"', f'href="/static/dist/app.css?v={_v_css}"')
    html = html.replace('src="/static/dist/app.js"', f'src="/static/dist/app.js?v={_v_js}"')

    return HTMLResponse(
        content=html,
        headers={'Cache-Control': 'no-store'},
    )
