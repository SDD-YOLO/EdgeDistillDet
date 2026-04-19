"""当前请求的数据工作区根目录（团队隔离）；未设置时回退到仓库根。"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from uuid import UUID

from web.core.paths import BASE_DIR

_data_root_ctx: ContextVar[Path | None] = ContextVar("data_root", default=None)
_team_id_ctx: ContextVar[UUID | None] = ContextVar("team_id", default=None)


def get_data_root() -> Path:
    """配置、runs、agent 状态等用户数据的根目录。"""
    p = _data_root_ctx.get()
    return p if p is not None else BASE_DIR


def get_current_team_id() -> UUID | None:
    return _team_id_ctx.get()


def set_workspace(team_id: UUID, data_root: Path):
    """返回两个 token，用于 finally 中 reset。"""
    return _team_id_ctx.set(team_id), _data_root_ctx.set(data_root.resolve())


def reset_workspace(team_token, root_token):
    _team_id_ctx.reset(team_token)
    _data_root_ctx.reset(root_token)


def team_workspace_path(team_id: UUID) -> Path:
    from web.core.saas_settings import get_team_data_root

    return Path(get_team_data_root()) / str(team_id)
