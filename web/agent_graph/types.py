from __future__ import annotations

from typing import Any, Literal, TypedDict

ApprovalStatus = Literal["not_required", "pending", "approved", "rejected", "expired"]


class ToolCall(TypedDict, total=False):
    tool: str
    args: dict[str, Any]


class ToolLog(TypedDict, total=False):
    call: ToolCall
    result: Any


class GraphState(TypedDict, total=False):
    # input context
    user_text: str
    run_id: str
    requested_tool: str
    requested_args: dict[str, Any]
    system_prompt: str

    # model interaction
    model_reply: str
    model_reasoning: str
    tool_call: ToolCall

    # tool execution
    tool_result: Any
    tool_logs: list[ToolLog]

    # patch / approval context
    patch: dict[str, Any]
    need_approval: bool
    approval_status: ApprovalStatus
    approval_token: str
    request_hash: str
    change_summary: dict[str, Any]

    # final output
    response: dict[str, Any]
    error: str
