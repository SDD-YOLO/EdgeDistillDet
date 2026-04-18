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


class AgentPatchPreviewRequest(BaseModel):
    patch: dict = Field(default_factory=dict)
    run_id: str = "default"
    operator: str = "user"
    reason: str = ""


class AgentPatchApplyRequest(BaseModel):
    approval_token: str | None = None
    token: str | None = None
    run_id: str = "default"
    operator: str = "user"
    request_hash: str | None = None
    reason: str = ""


class AgentPatchValidateRequest(BaseModel):
    patch: dict = Field(default_factory=dict)
    strict: bool = True


class AgentRunHistoryRollbackRequest(BaseModel):
    run_id: str = "default"
    target_version: int | None = None
    steps: int | None = None
    operator: str = "user"
    reason: str = "manual rollback"


class AgentToolExecuteRequest(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class AgentMessageItem(BaseModel):
    role: str
    content: str


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
