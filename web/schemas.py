"""
web/schemas.py
==============
Web API 的请求模型定义，供路由层复用。
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    ok: bool = True
    data: T | None = None
    error: str | None = None
    meta: dict = Field(default_factory=dict)


class SaveConfigRequest(BaseModel):
    name: str = "distill_config.yaml"
    config: dict = Field(default_factory=dict)

    # 注意：SaveConfigRequest 不做强制验证
    # 目的：允许用户随时保存部分配置或草稿
    # 完整验证在 TrainStartRequest 中进行


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
    run_id: str = "default"
    session_id: str = "default"
    rag_options: dict = Field(default_factory=dict)
    tool_policy: dict = Field(default_factory=dict)
    max_steps: int = 4


class DisplayStartRequest(BaseModel):
    config: str = "distill_config.yaml"
    source: str | None = None
    weight: str | None = None
    device: str | None = None
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    visualize: bool | None = None
    show: bool | None = None
    save_txt: bool | None = None
    save_conf: bool | None = None
    save_crop: bool | None = None
    show_labels: bool | None = None
    show_conf: bool | None = None
    show_boxes: bool | None = None
    line_width: int | None = None
    output_dir: str | None = None


class ExportStartRequest(BaseModel):
    config: str = "distill_config.yaml"
    weight: str | None = None
    export_path: str | None = Field(default=None, min_length=1)
    format: str = "onnx"
    keras: bool = False
    optimize: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: int | None = None
    workspace: int | None = None
    nms: bool = False


class TrainStartRequest(BaseModel):
    config: str = "distill_config.yaml"
    mode: str = "distill"
    checkpoint: str | None = None
    allow_overwrite: bool = False

    @model_validator(mode="after")
    def validate_training_start(self) -> TrainStartRequest:
        """
        启动训练前的请求验证

        检查项：
        1. config 不能为空
        2. mode 必须是 'distill' 或 'resume'

        NOTE: 完整的配置一致性验证在 API 路由中执行
        """
        try:
            # 基础字段验证
            if not self.config or not isinstance(self.config, str):
                raise ValueError("config 必须是非空字符串")

            if self.mode not in ("distill", "resume"):
                raise ValueError(f"mode 必须是 'distill' 或 'resume'，当前值: {self.mode}")
        except ValueError as e:
            raise e
        except Exception:
            pass

        return self
