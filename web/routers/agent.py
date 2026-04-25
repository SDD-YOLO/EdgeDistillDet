from __future__ import annotations

from fastapi import APIRouter

from web.schemas import (
    AgentModelInvokeRequest,
    AgentPatchApplyRequest,
    AgentPatchPreviewRequest,
    AgentRunHistoryRollbackRequest,
    AgentToolExecuteRequest,
)
from web.services import agent_service

router = APIRouter()


@router.post("/api/agent/patch/preview")
def agent_patch_preview(payload: AgentPatchPreviewRequest):
    return agent_service.agent_patch_preview(payload)


@router.post("/api/agent/patch/apply")
def agent_patch_apply(payload: AgentPatchApplyRequest):
    return agent_service.agent_patch_apply(payload)


@router.get("/api/agent/run/{run_id}/history")
def agent_run_history(run_id: str):
    return agent_service.agent_run_history(run_id)


@router.post("/api/agent/run/{run_id}/rollback")
def agent_run_rollback(run_id: str, payload: AgentRunHistoryRollbackRequest):
    return agent_service.agent_run_rollback(run_id, payload)


@router.get("/api/agent/tools")
def agent_tools_contract():
    return agent_service.agent_tools_contract()


@router.get("/api/agent/prompts")
def agent_prompts():
    return agent_service.agent_prompts()


@router.post("/api/agent/tools/execute")
def agent_tools_execute(payload: AgentToolExecuteRequest):
    return agent_service.agent_tools_execute(payload)


@router.post("/api/agent/model/invoke")
def agent_model_invoke(payload: AgentModelInvokeRequest):
    return agent_service.agent_model_invoke(payload)


@router.post("/api/agent/model/invoke-stream")
def agent_model_invoke_stream(payload: AgentModelInvokeRequest):
    return agent_service.agent_model_invoke_stream(payload)
