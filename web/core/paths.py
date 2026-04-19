"""路径与基础目录常量。"""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = WEB_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

TEMPLATE_FILE = WEB_DIR / "templates" / "index.html"
STATIC_DIR = WEB_DIR / "static"


def get_config_dir() -> Path:
    """当前工作区下的 configs/（多租户下为团队数据根）。"""
    from web.core.workspace import get_data_root

    return get_data_root() / "configs"


def get_agent_state_dir() -> Path:
    from web.core.workspace import get_data_root

    return get_data_root() / ".agent_state"


def get_agent_history_dir() -> Path:
    return get_agent_state_dir() / "run_history"


def train_lock_file() -> Path:
    from web.core.workspace import get_data_root

    return get_data_root() / ".training.lock"


# 兼容旧代码：默认等同仓库根下的路径（工作区未绑定时与 get_* 一致）
CONFIG_DIR = BASE_DIR / "configs"
AGENT_STATE_DIR = BASE_DIR / ".agent_state"
AGENT_HISTORY_DIR = AGENT_STATE_DIR / "run_history"
