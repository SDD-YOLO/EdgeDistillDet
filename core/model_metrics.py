"""
core/model_metrics.py
=====================
模型参数量与 GFLOPs 统一解析工具。
"""

from __future__ import annotations

import importlib
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    suffix = text[-1].upper()
    scale = 1.0
    if suffix == "M":
        scale = 1e6
        text = text[:-1]
    elif suffix == "K":
        scale = 1e3
        text = text[:-1]
    try:
        return float(text) * scale
    except ValueError:
        return None


def estimate_params_m_from_checkpoint(weight_path: str | Path) -> float | None:
    import torch

    path = str(weight_path)
    raw = None
    try:
        raw = torch.load(path, map_location="cpu")
    except Exception:
        pass

    if raw is None:
        try:
            safe_globals = []
            ul_spec = importlib.util.find_spec("ultralytics")
            if ul_spec:
                ul = importlib.import_module("ultralytics")
                if hasattr(ul, "nn") and hasattr(ul.nn, "tasks") and hasattr(ul.nn.tasks, "DetectionModel"):
                    safe_globals.append(ul.nn.tasks.DetectionModel)
            if safe_globals and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals(safe_globals):
                    raw = torch.load(path, map_location="cpu", weights_only=False)
            else:
                raw = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None

    state_dict = None
    if hasattr(raw, "state_dict"):
        state_dict = raw.state_dict()
    elif isinstance(raw, dict):
        if "model" in raw:
            model_obj = raw["model"]
            if hasattr(model_obj, "state_dict"):
                state_dict = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                state_dict = model_obj
        elif "state_dict" in raw:
            inner = raw["state_dict"]
            if hasattr(inner, "state_dict"):
                state_dict = inner.state_dict()
            elif isinstance(inner, dict):
                state_dict = inner
        elif all(hasattr(v, "numel") for v in raw.values()):
            state_dict = raw

    if not isinstance(state_dict, dict):
        return None

    total_params = 0
    for value in state_dict.values():
        if hasattr(value, "numel"):
            total_params += int(value.numel())
    if total_params <= 0:
        return None
    return total_params / 1e6


def estimate_gflops_from_weight(weight_path: str | Path) -> float | None:
    from ultralytics import YOLO

    try:
        model = YOLO(str(weight_path))
        info = model.info(verbose=False)
        del model
        if isinstance(info, (list, tuple)) and len(info) >= 2:
            gflops = float(info[1])
            return gflops if gflops > 0 else None
    except Exception:
        return None
    return None


def extract_model_stats(model: Any) -> dict[str, str]:
    """
    从已加载的 YOLO 对象提取显示用参数量/GFLOPs 字段。
    返回：
      {"params": "12,345,678" | "N/A", "gflops": "8.7" | "N/A"}
    """
    params, gflops = "N/A", "N/A"
    captured = ""

    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            info_result = model.info(verbose=True)
        captured = buf.getvalue()

        if isinstance(info_result, (list, tuple)) and len(info_result) >= 2:
            n_params = _parse_number(info_result[1])
            g = _parse_number(info_result[3] if len(info_result) >= 4 else info_result[1])
            if n_params is not None and n_params > 0:
                params = f"{int(n_params):,}"
            if g is not None and g > 0:
                gflops = f"{g:.1f}".rstrip("0").rstrip(".")
        elif isinstance(info_result, dict):
            for key in ("params", "n_p", "n_params", "num_params"):
                n_params = _parse_number(info_result.get(key))
                if n_params is not None and n_params > 0:
                    params = f"{int(n_params):,}"
                    break
            for key in ("gflops", "flops", "gfloats"):
                g = _parse_number(info_result.get(key))
                if g is not None and g > 0:
                    gflops = f"{g:.1f}".rstrip("0").rstrip(".")
                    break
    except Exception:
        pass

    if params == "N/A" or gflops == "N/A":
        text = captured
        if not text:
            try:
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    model.info(verbose=False)
                text = buf.getvalue()
            except Exception:
                text = ""
        for line in text.splitlines():
            lower = line.lower()
            if "parameters" in lower and params == "N/A":
                for token in line.replace(",", "").split():
                    number = _parse_number(token)
                    if number is not None and number >= 1000:
                        params = f"{int(number):,}"
                        break
            if "gflops" in lower and gflops == "N/A":
                for token in line.split():
                    g = _parse_number(token)
                    if g is not None and g > 0:
                        gflops = f"{g:.1f}".rstrip("0").rstrip(".")
                        break

    if params == "N/A":
        try:
            total = sum(p.numel() for p in model.model.parameters())
            params = f"{int(total):,}"
        except Exception:
            pass

    return {"params": params, "gflops": gflops}
