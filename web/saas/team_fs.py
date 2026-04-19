"""团队工作区目录初始化。"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from web.core.paths import BASE_DIR
from web.core.saas_settings import get_team_data_root


def team_root_path(team_id: str) -> Path:
    return Path(get_team_data_root()) / team_id


def ensure_team_workspace(team_id: str | UUID) -> Path:
    tid = str(team_id)
    root = team_root_path(tid)
    for sub in ("configs", "runs", "datasets", ".agent_state", ".agent_state/run_history", ".job_logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # 首次：从仓库 configs 复制默认 YAML（不覆盖已有）
    dst_configs = root / "configs"
    src = BASE_DIR / "configs"
    if src.is_dir():
        for f in src.glob("*.yaml"):
            target = dst_configs / f.name
            if not target.exists():
                shutil.copy2(f, target)
            elif f.name == "distill_config.yaml" and target.stat().st_size == 0:
                shutil.copy2(f, target)
    return root
