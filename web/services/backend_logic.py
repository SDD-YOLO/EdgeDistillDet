"""后端逻辑 facade：将领域实现拆分到子模块并保持旧导入兼容。"""

from __future__ import annotations

from web.services.backend_agent import (
    agent_model_invoke,
    agent_model_invoke_stream,
    agent_patch_apply,
    agent_patch_preview,
    agent_prompts,
    agent_run_history,
    agent_run_rollback,
    agent_tools_contract,
    agent_tools_execute,
)
from web.services.backend_config import (
    get_config,
    get_configs,
    get_recent_config,
    pick_path_dialog,
    save_config,
    upload_config,
)
from web.services.backend_metrics import get_metrics
from web.services.backend_train import (
    download_training_logs,
    export_weight_candidates,
    get_resume_candidates,
    get_training_logs,
    get_training_status,
    output_check,
    start_training,
    stop_training,
    stream_training_logs,
)
from web.services.backend_ui import favicon, index

__all__ = [
    'agent_model_invoke', 'agent_model_invoke_stream', 'agent_patch_apply',
    'agent_patch_preview', 'agent_prompts', 'agent_run_history', 'agent_run_rollback',
    'agent_tools_contract', 'agent_tools_execute', 'download_training_logs', 'favicon', 'get_config',
    'get_configs', 'get_metrics', 'get_recent_config', 'get_resume_candidates', 'export_weight_candidates', 'get_training_logs',
    'get_training_status', 'index', 'output_check', 'pick_path_dialog', 'save_config', 'start_training',
    'stop_training', 'stream_training_logs', 'upload_config',
]
