"""
web/schemas.py
==============
Web API 的请求模型定义，供路由层复用。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SaveConfigRequest(BaseModel):
    name: str = "distill_config.yaml"
    config: dict = Field(default_factory=dict)


class UploadConfigRequest(BaseModel):
    content: str


class DialogFilterItem(BaseModel):
    name: str = "All Files"
    patterns: list[str] = Field(default_factory=lambda: ["*.*"])


class DialogPickRequest(BaseModel):
    kind: str = "file"
    title: str = "选择路径"
    initial_path: str | None = None
    filters: list[DialogFilterItem] = Field(default_factory=list)


class AgentMessageItem(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    provider: str = "openai_compatible"
    api_url: str
    api_key: str | None = None
    model: str | None = None
    messages: list[AgentMessageItem] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    endpoint: str | None = None
    extra_headers: dict = Field(default_factory=dict)
    timeout_sec: float = 40.0


class AgentChatResponse(BaseModel):
    status: str = "ok"
    reply: str = ""
    reasoning: str | None = None
    route: str | None = None
    intent: str | None = None
    raw: dict = Field(default_factory=dict)


class AgentPatchValidateRequest(BaseModel):
    patch: dict = Field(default_factory=dict)
    strict: bool = True


class AgentPatchPreviewRequest(BaseModel):
    run_id: str = "default"
    patch: dict = Field(default_factory=dict)
    operator: str = "agent"
    reason: str = "agent.preview_patch"


class AgentPatchApplyRequest(BaseModel):
    run_id: str = "default"
    approval_token: str | None = None
    request_hash: str | None = None
    operator: str = "agent"
    reason: str = "agent.apply_patch_with_approval"


class AgentRunHistoryRollbackRequest(BaseModel):
    target_version: int | None = None
    steps: int = 1
    operator: str = "agent"
    reason: str = "agent.rollback_run_config"


class AgentToolExecuteRequest(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class AgentModelInvokeRequest(BaseModel):
    provider: str = "openai_compatible"
    api_url: str
    api_key: str | None = None
    model: str | None = None
    messages: list[AgentMessageItem] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    endpoint: str | None = None
    extra_headers: dict = Field(default_factory=dict)
    timeout_sec: float = 40.0


class TrainStartRequest(BaseModel):
    config: str = "distill_config.yaml"
    mode: str = "distill"
    checkpoint: str | None = None
    allow_overwrite: bool = False
