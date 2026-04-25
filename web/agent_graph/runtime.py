from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Generator

from web.agent_rag import retrieve_hybrid

from .graph import build_agentic_rag_graph, build_model_graph, build_tool_graph
from .model_client import invoke_chat_completion
from .types import GraphState

ToolExecutor = Callable[[str, dict[str, Any], GraphState], Any]
_BLOCKED_LINE_RE = re.compile(
    r"执行命令\s*[（(]\s*需审批\s*[）)]|python\s+distill\.py\b[^\n\r]*configs[\\/]+distill_config\.ya?ml",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_BRACED_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_tool_name(reply: str) -> str:
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', str(reply or ""))
    return m.group(1).strip() if m else ""


def _sanitize_blocked_reply_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned_lines = [line for line in raw.splitlines() if not _BLOCKED_LINE_RE.search(line)]
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_action_payload(reply: str) -> dict[str, Any]:
    raw = str(reply or "").strip()
    if not raw:
        return {}
    candidates = [raw]
    candidates.extend(m.group(1).strip() for m in _JSON_BLOCK_RE.finditer(raw))
    brace_match = _BRACED_JSON_RE.search(raw)
    if brace_match:
        candidates.append(brace_match.group(0))
    for item in candidates:
        try:
            parsed = json.loads(item)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(list(messages or [])):
        if isinstance(msg, dict) and str(msg.get("role") or "") == "user":
            return str(msg.get("content") or "")
    return ""


def _build_agentic_system_prompt(base_prompt: str | None) -> str:
    control_prompt = (
        "你是一个 Agentic RAG 助手。你可以在每一步自主选择：\n"
        "1) 检索（action=retrieve）\n"
        "2) 调用工具（action=tool）\n"
        "3) 直接回答（action=final）\n"
        "输出必须是 JSON 对象，格式之一：\n"
        '{"action":"retrieve","query":"...","top_k":5}\n'
        '{"action":"tool","tool":"agent.get_context","args":{"run_id":"default"}}\n'
        '{"action":"final","reply":"给用户的最终答复","reasoning":"可选"}\n'
        "若你引导用户到「批准修改训练配置」或需写入配置，须先以 action=tool 调用 agent.preview_patch 拿到 approval；禁止在从未调用过 preview 时仅在 final 中写去批准。\n"
        "当你修改 distillation.w_feat 时，必须使用数值标量（int/float），禁止输出数组/列表。\n"
        "如果工具返回需要审批（approval_token），不要自动执行 apply，改为告知用户在审批区执行。"
    )
    base = str(base_prompt or "").strip()
    if not base:
        return control_prompt
    return f"{base}\n\n{control_prompt}"


def _plan_once(payload: dict[str, Any], state: GraphState) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    resp = invoke_chat_completion(
        api_url=str(payload.get("api_url") or ""),
        api_key=payload.get("api_key"),
        model=payload.get("model"),
        messages=messages,
        system_prompt=_build_agentic_system_prompt(payload.get("system_prompt")),
        temperature=float(payload.get("temperature") or 0.2),
        max_tokens=payload.get("max_tokens"),
        endpoint=payload.get("endpoint"),
        timeout_sec=float(payload.get("timeout_sec") or 40.0),
        extra_headers=payload.get("extra_headers") or {},
    )
    reply = _sanitize_blocked_reply_text(str(resp.get("reply") or ""))
    reasoning = str(resp.get("reasoning") or "")
    parsed = _extract_action_payload(reply)
    action = str(parsed.get("action") or "").strip().lower()
    if action in {"rag", "search"}:
        action = "retrieve"
    if action not in {"retrieve", "tool", "final"}:
        action = "final"
    out: dict[str, Any] = {"status": "ok", "action": action, "reply": reply, "reasoning": reasoning}
    if action == "retrieve":
        out["query"] = str(parsed.get("query") or _extract_last_user_message(messages))
        out["top_k"] = int(parsed.get("top_k") or 5)
    if action == "tool":
        tool = str(parsed.get("tool") or _extract_tool_name(reply) or "").strip()
        args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
        out.update({"tool": tool, "args": args})
    if action == "final":
        # 允许模型仅输出 JSON 决策，真正答复由 reply 字段承载。
        final_reply = str(parsed.get("reply") or "").strip()
        if final_reply:
            out["reply"] = final_reply
        final_reasoning = str(parsed.get("reasoning") or "").strip()
        if final_reasoning:
            out["reasoning"] = final_reasoning
    return out


def _retrieve_once(state: GraphState) -> list[dict[str, Any]]:
    messages = list(state.get("messages") or [])
    query = str(state.get("retrieval_query") or _extract_last_user_message(messages))
    rag_opts = state.get("rag_options") if isinstance(state.get("rag_options"), dict) else {}
    run_id = str(state.get("run_id") or rag_opts.get("run_id") or "default")
    top_k = int(rag_opts.get("top_k") or state.get("retrieval_top_k") or 5)
    return retrieve_hybrid(query, run_id=run_id, top_k=top_k)


def _append_observation_message(messages: list[dict[str, Any]], role: str, content: str) -> list[dict[str, Any]]:
    out = list(messages or [])
    out.append({"role": role, "content": content})
    return out


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
    try:
        initial_messages = list(payload.get("messages") or [])
        initial_state: GraphState = {
            "messages": initial_messages,
            "run_id": str(payload.get("run_id") or "default"),
            "session_id": str(payload.get("session_id") or "default"),
            "max_steps": int(payload.get("max_steps") or 4),
            "rag_options": payload.get("rag_options") if isinstance(payload.get("rag_options"), dict) else {},
            "tool_policy": payload.get("tool_policy") if isinstance(payload.get("tool_policy"), dict) else {},
            "step_count": 0,
        }

        def _planner(state: GraphState) -> dict[str, Any]:
            step_out = _plan_once(payload, state)
            if step_out.get("action") == "retrieve":
                state["retrieval_query"] = str(step_out.get("query") or "")
                state["retrieval_top_k"] = int(step_out.get("top_k") or 5)
            return step_out

        def _executor(tool: str, args: dict[str, Any], state: GraphState) -> Any:
            # 非流式调用也复用 tools graph，保持审批语义一致。
            from web.services.backend_agent import _execute_tool  # local import to avoid circular

            return _execute_tool(tool, args, state)

        app = build_agentic_rag_graph(_planner, _executor, _retrieve_once)
        out = app.invoke(initial_state)
        if isinstance(out, dict) and isinstance(out.get("response"), dict):
            return dict(out["response"])
        return {"status": "error", "error": "model_graph_failed", "reply": ""}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "reply": ""}

def _build_continuation_hint(called_tools: list[str], tool_result: dict) -> str:
    called = set(called_tools)

    if "agent.get_context" in called and "agent.get_training_results" not in called:
        # get_context 已完成，强制下一步读指标
        # get_context 现在已合并指标，跳过这步直接给分析指令
        pass

    if "agent.get_context" in called and "agent.preview_patch" not in called:
        return (
            "你已获取配置与训练指标数据。"
            "现在必须：1) 用自然语言输出完整的调参分析与建议值；"
            "2) 紧接着以 action=tool 调用 agent.preview_patch 将建议打包进审批流程。"
            "禁止直接输出 action=final 而不调用 preview_patch。"
        )

    if "agent.preview_patch" in called:
        return (
            "配置预览已完成，审批票据已签发。"
            "请以 action=final 输出简短确认，提示用户在审批区点击「让 agent 执行」。"
            "禁止再次调用任何工具。"
        )

    return "请继续。若需工具则输出 action=tool；若已有足够信息则输出 action=final。"

def invoke_model_graph_stream(payload: dict[str, Any]) -> Generator[str, None, None]:
    reply_full = ""
    reasoning_full = ""
    trace: list[dict[str, Any]] = []
    try:
        run_id = str(payload.get("run_id") or "default")
        max_steps = max(1, int(payload.get("max_steps") or 4))
        state_messages = list(payload.get("messages") or [])
        step = 0

        from web.services.backend_agent import _execute_tool  # local import to avoid circular

        while step < max_steps:
            step += 1
            step_state: GraphState = {
                "messages": state_messages,
                "run_id": run_id,
                "max_steps": max_steps,
                "rag_options": payload.get("rag_options") if isinstance(payload.get("rag_options"), dict) else {},
                "tool_policy": payload.get("tool_policy") if isinstance(payload.get("tool_policy"), dict) else {},
                "step_count": step - 1,
            }
            planned = _plan_once(payload, step_state)
            action = str(planned.get("action") or "final")
            step_reply = str(planned.get("reply") or "")
            step_reasoning = str(planned.get("reasoning") or "")
            reply_full = step_reply
            reasoning_full = step_reasoning
            model_event = {
                "event_type": "model_output",
                "timestamp": int(time.time() * 1000),
                "step": step,
                "payload": {"reply": step_reply, "reasoning": step_reasoning, "action": action},
            }
            trace.append(model_event)
            yield f"data: {json.dumps(model_event, ensure_ascii=False)}\n\n"
            # 兼容旧前端：仍发送一次 content/done 语义。
            if step == 1:
                yield f"data: {json.dumps({'t': 'content', 'd': step_reply}, ensure_ascii=False)}\n\n"
                if step_reasoning:
                    yield f"data: {json.dumps({'t': 'reasoning', 'd': step_reasoning}, ensure_ascii=False)}\n\n"

            if action == "retrieve":
                q = str(planned.get("query") or _extract_last_user_message(state_messages))
                rag_options = payload.get("rag_options") if isinstance(payload.get("rag_options"), dict) else {}
                top_k = int(planned.get("top_k") or rag_options.get("top_k") or 5)
                hits = retrieve_hybrid(q, run_id=run_id, top_k=top_k)
                for hit in hits:
                    ev = {
                        "event_type": "retrieval_hit",
                        "timestamp": int(time.time() * 1000),
                        "step": step,
                        "payload": hit,
                    }
                    trace.append(ev)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                observe_text = json.dumps({"query": q, "hits": hits}, ensure_ascii=False)
                state_messages = _append_observation_message(state_messages, "assistant", step_reply)
                state_messages = _append_observation_message(
                    state_messages,
                    "user",
                    f"[retrieval_observation]\n{observe_text}",
                )
                continue

            if action == "tool":
                tool = str(planned.get("tool") or "").strip()
                args = planned.get("args") if isinstance(planned.get("args"), dict) else {}
                ev_start = {
                    "event_type": "tool_start",
                    "timestamp": int(time.time() * 1000),
                    "step": step,
                    "payload": {"tool": tool, "args": args},
                }
                trace.append(ev_start)
                yield f"data: {json.dumps(ev_start, ensure_ascii=False)}\n\n"
                tool_result = execute_tool_graph(tool=tool, args=args, executor=_execute_tool, run_id=run_id)
                ev_end = {
                    "event_type": "tool_end",
                    "timestamp": int(time.time() * 1000),
                    "step": step,
                    "payload": {"tool": tool, "args": args, "result": tool_result},
                }
                trace.append(ev_end)
                yield f"data: {json.dumps(ev_end, ensure_ascii=False)}\n\n"
                state_messages = _append_observation_message(state_messages, "assistant", step_reply)
                state_messages = _append_observation_message(
                    state_messages,
                    "user",
                    f"[tool_observation:{tool}]\n{json.dumps(tool_result, ensure_ascii=False)}",
                )
                # ← 加这段：注入续行指令
                called_so_far = [
                    ev["payload"]["tool"]
                    for ev in trace
                    if ev.get("event_type") == "tool_end"
                ]
                hint = _build_continuation_hint(called_so_far, tool_result)
                state_messages = _append_observation_message(
                    state_messages,
                    "user",
                    f"[system_hint]\n{hint}",
                )

                continue

            break

        done_event = {
            "event_type": "done",
            "timestamp": int(time.time() * 1000),
            "step": step,
            "payload": {
                "reply": _sanitize_blocked_reply_text(reply_full),
                "reasoning": reasoning_full,
                "trace": trace,
            },
        }
        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'t': 'done', 'reply': _sanitize_blocked_reply_text(reply_full), 'reasoning': reasoning_full, 'events': trace}, ensure_ascii=False)}\n\n"
    except Exception as exc:
        err = {"event_type": "error", "timestamp": int(time.time() * 1000), "payload": {"message": str(exc)}}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        # 兼容旧前端
        err = {"t": "error", "message": str(exc)}
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
