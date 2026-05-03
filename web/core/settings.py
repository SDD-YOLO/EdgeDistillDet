"""
运行时配置：监听地址、CORS。

环境变量：
  EDGE_BACKEND_HOST   绑定地址，默认 127.0.0.1（仅本机）。局域网访问可设为 0.0.0.0。
  EDGE_BACKEND_PORT   端口，默认 5000（兼容旧名 EDGE_FLASK_PORT）。
  EDGE_CORS_ORIGINS   逗号分隔的允许来源；设为单个 * 表示任意来源（此时不携带 credentials）。
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_bind_host() -> str:
    return settings.host


def get_bind_port() -> int:
    return settings.port


def get_cors_middleware_kwargs() -> dict:
    """
    供 CORSMiddleware 使用的参数：allow_origins、allow_credentials。

    默认允许本机页面与 Vite 开发服务器（5173）访问 API；生产构建由同源 /static 提供，通常无需额外 CORS。
    """
    # Delegate to AppSettings implementation so behavior is centralized and testable.
    return settings.get_cors_middleware_kwargs()


class AppSettings(BaseSettings):
    host: str = Field("127.0.0.1", env="BACKEND_HOST")
    port: int = Field(5000, env="BACKEND_PORT")
    # Comma-separated list or single '*' for wildcard; parsed by helper below.
    cors_origins: str | None = Field(None, env="CORS_ORIGINS")
    frontend_dev_port: int = Field(5173, env="FRONTEND_DEV_PORT")

    model_config = SettingsConfigDict(env_prefix="EDGE_", env_file=".env")

    def get_cors_middleware_kwargs(self) -> dict:
        raw = (self.cors_origins or "").strip()
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
            backend_port = str(self.port)
            frontend_port = str(self.frontend_dev_port)
            origins = [
                f"http://127.0.0.1:{backend_port}",
                f"http://localhost:{backend_port}",
                f"http://127.0.0.1:{frontend_port}",
                f"http://localhost:{frontend_port}",
            ]
        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }


# Singleton settings instance for app imports
settings = AppSettings()
