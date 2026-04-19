"""pytest：默认关闭 SaaS，保证现有 API 测试无需数据库。"""

from __future__ import annotations

import os

os.environ.setdefault("EDGE_SAAS_ENABLED", "0")
