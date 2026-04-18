"""路径与基础目录常量。"""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = WEB_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CONFIG_DIR = BASE_DIR / "configs"
TEMPLATE_FILE = WEB_DIR / "templates" / "index.html"
STATIC_DIR = WEB_DIR / "static"
AGENT_STATE_DIR = BASE_DIR / ".agent_state"
AGENT_HISTORY_DIR = AGENT_STATE_DIR / "run_history"
