from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Generator

from .graph import build_model_graph, build_tool_graph
from .model_client import invoke_chat_completion, stream_chat_completion
from .types import GraphState

ToolExecutor = Callable[[str, dict[str, Any], GraphState], Any]
_DEBUG_LOG_PATH = Path("debug-d5a26f.log")
_BLOCKED_LINE_RE = re.compile(
    r"执行命令\s*[（(]\s*需审批\s*[）)]|python\s+distill\.py\b[^\n\r]*configs[\\/]+distill_config\.ya?ml",
    re.IGNORECASE,
)


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


def _extract_tool_name(reply: str) -> str:
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', str(reply or ""))
    return m.group(1).strip() if m else ""


def _extract_last_user_text(payload: dict[str, Any]) -> str:
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return ""
    for item in reversed(msgs):
        if isinstance(item, dict) and str(item.get("role") or "") == "user":
            return str(item.get("content") or "")
    return ""


def _sanitize_blocked_reply_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned_lines = [line for line in raw.splitlines() if not _BLOCKED_LINE_RE.search(line)]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def execute_tool_graph(
    *,
    tool: str,
    args: dict[str, Any] | None,
    executor: ToolExecutor,
    run_id: str = "default",
) -> dict[str, Any]:
    app = build_tool_graph(executor)
    initial_state: GraphState = {
        "requested_tool": tool,
        "requested_args": dict(args or {}),
        "run_id": run_id,
    }
    out = app.invoke(initial_state)
    response = out.get("response") if isinstance(out, dict) else None
    if isinstance(response, dict):
        return response
    return {"status": "error", "error": "graph_execution_failed"}


def invoke_model_graph(payload: dict[str, Any]) -> dict[str, Any]:
    def _invoker(state: GraphState) -> dict[str, Any]:
        return invoke_chat_completion(
            api_url=str(payload.get("api_url") or ""),
            api_key=payload.get("api_key"),
            model=payload.get("model"),
            messages=list(payload.get("messages") or []),
            system_prompt=payload.get("system_prompt"),
            temperature=float(payload.get("temperature") or 0.2),
            max_tokens=payload.get("max_tokens"),
            endpoint=payload.get("endpoint"),
            timeout_sec=float(payload.get("timeout_sec") or 40.0),
            extra_headers=payload.get("extra_headers") or {},
        )

    app = build_model_graph(_invoker)
    out = app.invoke({"user_text": "invoke_model"})
    if isinstance(out, dict) and isinstance(out.get("response"), dict):
        response = dict(out["response"])
        raw_reply_text = str(response.get("reply") or "")
        reply_text = _sanitize_blocked_reply_text(raw_reply_text)
        response["reply"] = reply_text
        reasoning_text = str(response.get("reasoning") or "")
        return response
    return {"status": "error", "error": "model_graph_failed", "reply": ""}


def invoke_model_graph_stream(payload: dict[str, Any]) -> Generator[str, None, None]:
    reply_full = ""
    reasoning_full = ""
    try:
        stream = stream_chat_completion(
            api_url=str(payload.get("api_url") or ""),
            api_key=payload.get("api_key"),
            model=payload.get("model"),
            messages=list(payload.get("messages") or []),
            system_prompt=payload.get("system_prompt"),
            temperature=float(payload.get("temperature") or 0.2),
            max_tokens=payload.get("max_tokens"),
            endpoint=payload.get("endpoint"),
            timeout_sec=float(payload.get("timeout_sec") or 40.0),
            extra_headers=payload.get("extra_headers") or {},
        )
        while True:
            try:
                event = next(stream)
            except StopIteration as done:
                final_payload = done.value or {}
                reply_full = str(final_payload.get("reply") or reply_full)
                reasoning_full = str(final_payload.get("reasoning") or reasoning_full)
                break
            if not isinstance(event, dict):
                continue
            if event.get("t") == "content":
                chunk = str(event.get("d") or "")
                reply_full += chunk
                yield f"data: {json.dumps({'t': 'content', 'd': chunk}, ensure_ascii=False)}\n\n"
            elif event.get("t") == "reasoning":
                chunk = str(event.get("d") or "")
                reasoning_full += chunk
                yield f"data: {json.dumps({'t': 'reasoning', 'd': chunk}, ensure_ascii=False)}\n\n"
        # region agent log
        _debug_log(
            run_id=str(payload.get("run_id") or "default"),
            hypothesis_id="H21",
            location="web/agent_graph/runtime.py:invoke_model_graph_stream",
            message="Model stream finished",
            data={
                "reply_len": len(reply_full),
                "tool_in_reply": _extract_tool_name(reply_full),
                "last_user_text": _extract_last_user_text(payload)[:220],
                "user_has_negation": bool(
                    re.search(r"不改参数|不要改参数|无需修改参数|不用修改参数|先不修改参数|仅咨询|只咨询|不需要执行", _extract_last_user_text(payload))
                ),
                "user_has_mutation_hint": bool(
                    re.search(r"修改|调整|patch|补丁|预览|preview|应用|apply|写入|执行", _extract_last_user_text(payload), re.I)
                ),
            },
        )
        # endregion
        done_event = {"t": "done", "reply": _sanitize_blocked_reply_text(reply_full), "reasoning": reasoning_full}
        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
    except Exception as exc:
        err = {"t": "error", "message": str(exc)}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
