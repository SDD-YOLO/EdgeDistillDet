import os
import re
from typing import Any


def _expand_env_vars_str(value: str) -> str:
    if not isinstance(value, str) or not value:
        return value

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default if default is not None else '')

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\:-([^}]*))?\}", _replace, value)
    return os.path.expandvars(value)


def expand_env_vars(data: Any) -> Any:
    if isinstance(data, str):
        return _expand_env_vars_str(data)
    if isinstance(data, dict):
        return {key: expand_env_vars(value) for key, value in data.items()}
    if isinstance(data, list):
        return [expand_env_vars(value) for value in data]
    if isinstance(data, tuple):
        return tuple(expand_env_vars(value) for value in data)
    return data
