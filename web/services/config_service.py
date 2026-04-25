"""
web/services/config_service.py
==============================
配置文件相关业务逻辑（与路由解耦）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.distillation.common import w_feat_to_scalar
from utils import expand_env_vars

_last_saved_config: dict | None = None


def _normalize_w_feat_scalar(config: dict) -> None:
    """确保 distillation.w_feat 在内存与落盘时均为标量（兼容旧版列表）。"""
    d = config.get("distillation")
    if not isinstance(d, dict) or "w_feat" not in d:
        return
    d["w_feat"] = w_feat_to_scalar(d.get("w_feat"))


def list_config_names(config_dir: Path) -> list[str]:
    names: list[str] = []
    if config_dir.exists():
        for path in sorted(config_dir.iterdir()):
            if path.is_file() and path.suffix in {".yaml", ".yml"}:
                names.append(path.name)
    return names


def load_config(config_path: Path) -> dict | None:
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    data = expand_env_vars(data)
    if isinstance(data, dict):
        _normalize_w_feat_scalar(data)
    return data


def save_config(config_dir: Path, name: str, config: dict) -> tuple[str, int]:
    global _last_saved_config
    actual_name = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    target = config_dir / actual_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, dict):
        _normalize_w_feat_scalar(config)
    with open(target, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, allow_unicode=True, sort_keys=False)
    _last_saved_config = {"name": actual_name, "config": config}
    try:
        mtime_ns = int(target.stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return actual_name, mtime_ns


def get_recent_or_default(config_dir: Path, default_name: str = "distill_config.yaml") -> dict:
    if _last_saved_config is not None:
        return {"name": _last_saved_config["name"], "config": _last_saved_config["config"]}
    default_path = config_dir / default_name
    return {"name": default_name, "config": load_config(default_path) or {}}


def parse_uploaded_yaml(content: str) -> dict:
    config = expand_env_vars(yaml.safe_load(content) or {})
    if not isinstance(config, dict):
        raise ValueError("配置文件必须包含顶层映射对象")
    _normalize_w_feat_scalar(config)
    return config
