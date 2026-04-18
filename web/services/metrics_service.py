"""指标查询 API 服务。"""

from __future__ import annotations

from web.services import backend_logic


def get_metrics(source: str):
    return backend_logic.get_metrics(source=source)
