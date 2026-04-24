"""LangGraph-based agent backend with approval gate."""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.responses import StreamingResponse

from web.agent_graph.runtime import execute_tool_graph, invoke_model_graph, invoke_model_graph_stream
from web.core.paths import AGENT_HISTORY_DIR, AGENT_STATE_DIR, CONFIG_DIR
from web.schemas import (
    AgentModelInvokeRequest,
    AgentPatchApplyRequest,
    AgentPatchPreviewRequest,
    AgentPatchValidateRequest,
    AgentRunHistoryRollbackRequest,
    AgentToolExecuteRequest,
)
from web.services import backend_state, config_service
from web.services.backend_common import _error

_ALLOWED_TOP_LEVEL = ("distillation", "training", "output")
_DEPRECATED_LEAF_PATHS = {
    "distillation.temperature",
    "distillation.schedule_type",
    "distillation.feat_layer",
    "distillation.alpha",
    "distillation.distill_type",
    "training.warm_epochs",
    "training.workers",
    "training.batch_size",
    "training.learning_rate",
    "training.grad_clip",
    "output.model_dir",
    "output.log_dir",
}
_DEFAULT_CONFIG_NAME = "distill_config.yaml"
_APPROVAL_TTL_SEC = 15 * 60
_HISTORY_LIMIT = 5

_approval_lock = threading.RLock()
_approval_store: dict[str, dict[str, Any]] = {}
_DEBUG_LOG_PATH = Path("debug-d5a26f.log")


def _now_ts() -> float:
    return time.time()


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "d5a26f",
            "runId": str(run_id or "default"),
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "message": str(message),
            "data": data if isinstance(data, dict) else {},
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


def _json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _distill_config_path() -> Path:
    return CONFIG_DIR / _DEFAULT_CONFIG_NAME


def _load_distill_config() -> dict[str, Any]:
    return config_service.load_config(_distill_config_path()) or {}


def _save_distill_config(config: dict[str, Any]) -> int:
    name, file_mtime_ns = config_service.save_config(CONFIG_DIR, _DEFAULT_CONFIG_NAME, config)
    backend_state.last_saved_config = {"name": name, "config": config}
    return file_mtime_ns


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _collect_leaf_paths(payload: dict[str, Any], prefix: str = "") -> list[str]:
    leaves: list[str] = []
    for key, value in (payload or {}).items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            leaves.extend(_collect_leaf_paths(value, path))
        else:
            leaves.append(path)
    return leaves


def _get_by_path(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    cur = data
    segs = path.split(".")
    for seg in segs[:-1]:
        node = cur.get(seg)
        if not isinstance(node, dict):
            node = {}
            cur[seg] = node
        cur = node
    cur[segs[-1]] = value


def _validate_top_level(patch: dict[str, Any]) -> list[str]:
    return [k for k in patch.keys() if k not in _ALLOWED_TOP_LEVEL]


def _collect_deprecated_leaf_paths(payload: dict[str, Any]) -> list[str]:
    return sorted(path for path in _collect_leaf_paths(payload or {}) if path in _DEPRECATED_LEAF_PATHS)


def _flatten_allowed(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
            return
        if prefix in _DEPRECATED_LEAF_PATHS:
            return
        flat[prefix] = val

    for top in _ALLOWED_TOP_LEVEL:
        if top in data:
            walk(top, data[top])
    return flat


def _merge_distill_patch(base_config: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(base_config or {})
    invalid = _validate_top_level(patch or {})
    if invalid:
        raise ValueError(f"patch includes disallowed top-level keys: {invalid}")
    return _deep_merge(base, patch or {})


def _leaf_config_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    bflat = _flatten_allowed(before or {})
    aflat = _flatten_allowed(after or {})
    all_paths = sorted(set(bflat.keys()) | set(aflat.keys()))
    rows: list[dict[str, Any]] = []
    for path in all_paths:
        in_before = path in bflat
        in_after = path in aflat
        if in_before and in_after:
            if bflat[path] != aflat[path]:
                rows.append({"path": path, "kind": "changed", "before": bflat[path], "after": aflat[path]})
        elif in_before:
            rows.append({"path": path, "kind": "removed", "before": bflat[path], "after": None})
        else:
            rows.append({"path": path, "kind": "added", "before": None, "after": aflat[path]})
    return {"paths": rows, "stats": {"changed": len(rows)}}


def _filter_change_summary_to_patch_declared(raw_summary: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    declared = set(_collect_leaf_paths(patch or {}))
    filtered = [row for row in list(raw_summary.get("paths") or []) if row.get("path") in declared]
    return {"paths": filtered, "stats": {"changed": len(filtered)}}


def _single_leaf_epsilon_declared_patch(base: dict[str, Any], declared_patch: dict[str, Any]) -> dict[str, Any]:
    declared_paths = _collect_leaf_paths(declared_patch or {})

    def priority(path: str) -> tuple[int, str]:
        if path.startswith("distillation."):
            return (0, path)
        if path.startswith("training."):
            return (1, path)
        return (2, path)

    # Prefer float-ish knobs first for safer epsilon adjustment.
    for path in sorted(declared_paths, key=priority):
        val = _get_by_path(base, path)
        if isinstance(val, float):
            out: dict[str, Any] = {}
            _set_by_path(out, path, val + 1e-12)
            return out
    for path in sorted(declared_paths, key=priority):
        val = _get_by_path(base, path)
        if isinstance(val, int) and not isinstance(val, bool):
            out = {}
            _set_by_path(out, path, val + 1)
            return out
    if declared_paths:
        path = sorted(declared_paths, key=priority)[0]
        val = _get_by_path(base, path)
        if isinstance(val, str):
            new_val: Any = val + "_updated"
        elif isinstance(val, bool):
            new_val = not val
        else:
            new_val = 1
        out = {}
        _set_by_path(out, path, new_val)
        return out
    return {}


def _history_path(run_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(run_id or "default"))
    AGENT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return AGENT_HISTORY_DIR / f"{safe}.json"


def _load_run_history(run_id: str) -> list[dict[str, Any]]:
    path = _history_path(run_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_run_history(run_id: str, rows: list[dict[str, Any]]) -> None:
    path = _history_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_history(
    *,
    run_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    operator: str | None,
    reason: str | None,
    request_hash: str | None,
) -> int:
    rows = _load_run_history(run_id)
    version = (rows[-1]["version"] + 1) if rows else 1
    rows.append(
        {
            "version": version,
            "timestamp": int(_now_ts()),
            "operator": operator or "agent",
            "reason": reason or "",
            "request_hash": request_hash or "",
            "before": before,
            "after": after,
        }
    )
    rows = rows[-_HISTORY_LIMIT:]
    _save_run_history(run_id, rows)
    return version


def _tool_get_context(args: dict[str, Any]) -> dict[str, Any]:
    run_id = str(args.get("run_id") or "default")
    config = _load_distill_config()
    try:
        mtime = int(_distill_config_path().stat().st_mtime_ns)
    except OSError:
        mtime = 0
    return {
        "status": "ok",
        "run_id": run_id,
        "config_name": _DEFAULT_CONFIG_NAME,
        "config": config,
        "file_mtime_ns": mtime,
    }


def _tool_analyze_params(args: dict[str, Any]) -> dict[str, Any]:
    objective = str(args.get("objective") or "").strip() or "stability"
    config = _load_distill_config()
    distill = config.get("distillation") if isinstance(config.get("distillation"), dict) else {}
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    tips: list[str] = []
    temp = distill.get("temperature")
    if isinstance(temp, (int, float)) and temp < 1.0:
        tips.append("temperature may be too small for smooth distillation logits.")
    lr0 = training.get("lr0")
    if isinstance(lr0, (int, float)) and lr0 > 0.01:
        tips.append("lr0 is relatively high; watch for unstable loss.")
    if not tips:
        tips.append("current parameters are generally reasonable; try incremental tuning.")
    return {"status": "ok", "objective": objective, "analysis": tips}


def _tool_propose_patch(args: dict[str, Any]) -> dict[str, Any]:
    goal = str(args.get("goal") or "").strip().lower()
    base = _load_distill_config()
    patch: dict[str, Any] = {"distillation": {}}
    if "稳定" in goal or "stability" in goal:
        val = base.get("distillation", {}).get("T_max", 6.0)
        if isinstance(val, (int, float)):
            patch["distillation"]["T_max"] = round(float(val) + 0.2, 6)
    else:
        patch["distillation"]["w_kd"] = 0.6
    if not patch["distillation"]:
        patch = _single_leaf_epsilon_declared_patch(base, {"distillation": {"w_kd": base.get("distillation", {}).get("w_kd", 0.5)}})
    return {
        "status": "ok",
        "goal": goal or "generic optimization",
        "need_approval": True,
        "patch": patch,
        "result": {"goal": goal or "generic optimization", "patch": patch, "need_approval": True},
    }


def _tool_validate_patch(args: dict[str, Any]) -> dict[str, Any]:
    patch = args.get("patch")
    if not isinstance(patch, dict) or not patch:
        return {"status": "error", "error": "patch must be a non-empty object"}
    invalid = _validate_top_level(patch)
    if invalid:
        return {"status": "error", "error": f"invalid top-level keys: {invalid}", "invalid_keys": invalid}
    deprecated = _collect_deprecated_leaf_paths(patch)
    if deprecated:
        return {"status": "error", "error": f"patch contains deprecated fields: {deprecated}", "deprecated_paths": deprecated}
    base = _load_distill_config()
    merged = _merge_distill_patch(base, patch)
    diff = _leaf_config_diff(base, merged)
    filtered = _filter_change_summary_to_patch_declared(diff, patch)
    return {"status": "ok", "valid": True, "change_summary": filtered}


def _tool_preview_patch(args: dict[str, Any]) -> dict[str, Any]:
    patch = args.get("patch")
    if not isinstance(patch, dict) or not patch:
        return {"status": "error", "error": "patch must be a non-empty object"}
    deprecated = _collect_deprecated_leaf_paths(patch)
    if deprecated:
        return {"status": "error", "error": f"patch contains deprecated fields: {deprecated}", "deprecated_paths": deprecated}
    run_id = str(args.get("run_id") or "default")
    operator = str(args.get("operator") or "agent")
    reason = str(args.get("reason") or "agent.preview_patch")
    base = _load_distill_config()
    merged = _merge_distill_patch(base, patch)
    raw_summary = _leaf_config_diff(base, merged)
    change_summary = _filter_change_summary_to_patch_declared(raw_summary, patch)
    changed_count = int(change_summary.get("stats", {}).get("changed", 0))
    if changed_count <= 0:
        return {
            "status": "ok",
            "run_id": run_id,
            "request_hash": _json_hash({"run_id": run_id, "patch": patch}),
            "change_summary": change_summary,
            "patch": patch,
            "need_approval": False,
        }
    request_hash = _json_hash({"run_id": run_id, "patch": patch})
    token = secrets.token_urlsafe(24)
    now = _now_ts()
    with _approval_lock:
        _approval_store[token] = {
            "run_id": run_id,
            "patch": patch,
            "request_hash": request_hash,
            "operator": operator,
            "reason": reason,
            "created_at": now,
            "expires_at": now + _APPROVAL_TTL_SEC,
            "used": False,
        }
    return {
        "status": "ok",
        "run_id": run_id,
        "approval_token": token,
        "request_hash": request_hash,
        "change_summary": change_summary,
        "patch": patch,
        "need_approval": True,
        "expires_at": int(now + _APPROVAL_TTL_SEC),
    }


def _tool_apply_patch_with_approval(args: dict[str, Any]) -> dict[str, Any]:
    token = str(args.get("approval_token") or "").strip()
    if not token:
        return {"status": "error", "error": "approval_token is required"}
    run_id = str(args.get("run_id") or "default")
    request_hash = str(args.get("request_hash") or "").strip()
    operator = str(args.get("operator") or "agent")
    reason = str(args.get("reason") or "agent.apply_patch_with_approval")
    with _approval_lock:
        rec = _approval_store.get(token)
        if not rec:
            return {"status": "error", "error": "E_APPROVAL_EXPIRED"}
        if rec.get("used"):
            return {"status": "error", "error": "E_APPROVAL_REPLAY"}
        if float(rec.get("expires_at") or 0.0) < _now_ts():
            return {"status": "error", "error": "E_APPROVAL_EXPIRED"}
        if str(rec.get("run_id") or "default") != run_id:
            return {"status": "error", "error": "run_id mismatch for approval token"}
        if request_hash and request_hash != str(rec.get("request_hash") or ""):
            return {"status": "error", "error": "request_hash mismatch"}
        patch = rec.get("patch") if isinstance(rec.get("patch"), dict) else {}
        rec["used"] = True
    base = _load_distill_config()
    merged = _merge_distill_patch(base, patch)
    file_mtime_ns = _save_distill_config(merged)
    version = _append_history(
        run_id=run_id,
        before=base,
        after=merged,
        operator=operator,
        reason=reason,
        request_hash=str(rec.get("request_hash") or request_hash),
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "history_version": version,
        "config": merged,
        "file_mtime_ns": file_mtime_ns,
    }


def _tool_list_run_history(args: dict[str, Any]) -> dict[str, Any]:
    run_id = str(args.get("run_id") or "default")
    rows = _load_run_history(run_id)
    rows = rows[-_HISTORY_LIMIT:]
    return {"status": "ok", "run_id": run_id, "history": rows}


def _tool_rollback_run_config(args: dict[str, Any]) -> dict[str, Any]:
    run_id = str(args.get("run_id") or "default")
    target_version = args.get("target_version")
    steps = int(args.get("steps") or 1)
    operator = str(args.get("operator") or "agent")
    reason = str(args.get("reason") or "agent.rollback_run_config")
    rows = _load_run_history(run_id)
    if not rows:
        return {"status": "error", "error": "E_RUN_HISTORY_EMPTY"}
    target: dict[str, Any] | None = None
    if isinstance(target_version, int):
        for item in rows:
            if int(item.get("version") or -1) == target_version:
                target = item
                break
        if target is None:
            return {"status": "error", "error": f"target_version {target_version} not found"}
    else:
        idx = max(0, len(rows) - 1 - max(1, steps))
        target = rows[idx]
    rollback_config = copy.deepcopy(target.get("after") or {})
    before = _load_distill_config()
    file_mtime_ns = _save_distill_config(rollback_config)
    version = _append_history(
        run_id=run_id,
        before=before,
        after=rollback_config,
        operator=operator,
        reason=reason,
        request_hash="",
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "rolled_back_to_version": int(target.get("version") or 0),
        "history_version": version,
        "config": rollback_config,
        "file_mtime_ns": file_mtime_ns,
    }


def _execute_tool(tool: str, args: dict[str, Any], _state: dict[str, Any]) -> Any:
    handlers = {
        "agent.get_context": _tool_get_context,
        "agent.analyze_params": _tool_analyze_params,
        "agent.propose_patch": _tool_propose_patch,
        "agent.validate_patch": _tool_validate_patch,
        "agent.preview_patch": _tool_preview_patch,
        "agent.apply_patch_with_approval": _tool_apply_patch_with_approval,
        "agent.list_run_history": _tool_list_run_history,
        "agent.rollback_run_config": _tool_rollback_run_config,
    }
    handler = handlers.get(tool)
    if handler is None:
        return {"status": "error", "error": f"unknown tool: {tool}"}
    return handler(args)


def _load_tools_contract() -> dict[str, Any]:
    contract_path = CONFIG_DIR / "agent_tools_contract.json"
    if contract_path.exists():
        try:
            return json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "version": "v1",
        "pipeline": [
            "agent.get_context",
            "agent.analyze_params",
            "agent.propose_patch",
            "agent.validate_patch",
            "agent.preview_patch",
            "user_approval",
            "agent.apply_patch_with_approval",
        ],
        "tools": [],
    }


def agent_tools_contract():
    return _load_tools_contract()


def agent_tools_execute(payload: AgentToolExecuteRequest):
    tool = str(payload.tool or "").strip()
    args = payload.args if isinstance(payload.args, dict) else {}
    run_id = str(args.get("run_id") or "default")
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H20",
        location="web/services/backend_agent.py:agent_tools_execute",
        message="Tool execute request received",
        data={"tool": tool, "arg_keys": sorted(args.keys())},
    )
    # endregion
    out = execute_tool_graph(tool=tool, args=args, executor=_execute_tool, run_id=run_id)
    result = out.get("result") if isinstance(out, dict) else None
    if isinstance(result, dict):
        return result
    return {"status": "error", "error": "tool_graph_failed"}


def agent_patch_validate(payload: AgentPatchValidateRequest):
    args = {"patch": payload.patch, "strict": payload.strict}
    return _tool_validate_patch(args)


def agent_patch_preview(payload: AgentPatchPreviewRequest):
    return _tool_preview_patch(
        {
            "run_id": payload.run_id,
            "patch": payload.patch,
            "operator": payload.operator,
            "reason": payload.reason,
        }
    )


def agent_patch_apply(payload: AgentPatchApplyRequest):
    return _tool_apply_patch_with_approval(
        {
            "run_id": payload.run_id,
            "approval_token": payload.approval_token,
            "request_hash": payload.request_hash,
            "operator": payload.operator,
            "reason": payload.reason,
        }
    )


def agent_run_history(run_id: str):
    return _tool_list_run_history({"run_id": run_id})


def agent_run_rollback(run_id: str, payload: AgentRunHistoryRollbackRequest):
    return _tool_rollback_run_config(
        {
            "run_id": run_id,
            "target_version": payload.target_version,
            "steps": payload.steps,
            "operator": payload.operator,
            "reason": payload.reason,
        }
    )


def agent_model_invoke(payload: AgentModelInvokeRequest):
    try:
        return invoke_model_graph(payload.model_dump())
    except Exception as exc:
        return _error(f"agent model invoke failed: {exc}", 500)


def agent_model_invoke_stream(payload: AgentModelInvokeRequest):
    try:
        return StreamingResponse(
            invoke_model_graph_stream(payload.model_dump()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )
    except Exception as exc:
        return _error(f"agent model stream failed: {exc}", 500)


__all__ = [
    "agent_model_invoke",
    "agent_model_invoke_stream",
    "agent_patch_apply",
    "agent_patch_preview",
    "agent_patch_validate",
    "agent_run_history",
    "agent_run_rollback",
    "agent_tools_contract",
    "agent_tools_execute",
    "_filter_change_summary_to_patch_declared",
    "_leaf_config_diff",
    "_merge_distill_patch",
    "_single_leaf_epsilon_declared_patch",
]
