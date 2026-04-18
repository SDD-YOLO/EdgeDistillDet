"""Agent 相关 API 服务。"""

from __future__ import annotations

from web.schemas import (
    AgentModelInvokeRequest,
    AgentPatchApplyRequest,
    AgentPatchPreviewRequest,
    AgentPatchValidateRequest,
    AgentRunHistoryRollbackRequest,
    AgentToolExecuteRequest,
)
from web.services import backend_logic


def agent_config_schema():
    return backend_logic.agent_config_schema()


def agent_patch_validate(payload: AgentPatchValidateRequest):
    return backend_logic.agent_patch_validate(payload)


def agent_patch_preview(payload: AgentPatchPreviewRequest):
    return backend_logic.agent_patch_preview(payload)


def agent_patch_apply(payload: AgentPatchApplyRequest):
    return backend_logic.agent_patch_apply(payload)


def agent_run_history(run_id: str):
    return backend_logic.agent_run_history(run_id)


def agent_run_rollback(run_id: str, payload: AgentRunHistoryRollbackRequest):
    return backend_logic.agent_run_rollback(run_id, payload)


def agent_tools_contract():
    return backend_logic.agent_tools_contract()


def agent_tools_execute(payload: AgentToolExecuteRequest):
    return backend_logic.agent_tools_execute(payload)


def agent_model_invoke(payload: AgentModelInvokeRequest):
    return backend_logic.agent_model_invoke(payload)


def agent_model_invoke_stream(payload: AgentModelInvokeRequest):
    return backend_logic.agent_model_invoke_stream(payload)
