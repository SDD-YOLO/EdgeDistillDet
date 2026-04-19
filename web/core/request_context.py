"""SaaS 请求级上下文（由 workbench_context 注入）。"""

from __future__ import annotations

from contextvars import ContextVar

current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)


def get_request_user_id() -> str | None:
    return current_user_id.get()
