"""
运行时配置：监听地址、CORS。

环境变量：
  EDGE_BACKEND_HOST   绑定地址，默认 127.0.0.1（仅本机）。局域网访问可设为 0.0.0.0。
  EDGE_BACKEND_PORT   端口，默认 5000（兼容旧名 EDGE_FLASK_PORT）。
  EDGE_CORS_ORIGINS   逗号分隔的允许来源；设为单个 * 表示任意来源（此时不携带 credentials）。
"""

from __future__ import annotations

import os


def get_bind_host() -> str:
    h = os.environ.get("EDGE_BACKEND_HOST", "127.0.0.1").strip()
    return h or "127.0.0.1"


def get_bind_port() -> int:
    raw = os.environ.get("EDGE_BACKEND_PORT", os.environ.get("EDGE_FLASK_PORT", "5000"))
    return int(raw)


def get_cors_middleware_kwargs() -> dict:
    """
    供 CORSMiddleware 使用的参数：allow_origins、allow_credentials。

    默认允许本机页面与 Vite 开发服务器（5173）访问 API；生产构建由同源 /static 提供，通常无需额外 CORS。
    """
    raw = os.environ.get("EDGE_CORS_ORIGINS", "").strip()
    if raw == "*":
        return {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    if raw:
        origins = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        origins = [
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
