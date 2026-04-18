"""训练运行时状态集中管理。"""

from __future__ import annotations

import threading

training_process = None
training_status = {
    "running": False,
    "pid": None,
    "config": None,
    "mode": "distill",
    "start_time": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": [],
}
remote_training_state = {
    "active": False,
    "job_id": "",
    "api_base_url": "",
    "logs_offset": 0,
}
train_state_lock = threading.RLock()
train_log_cond = threading.Condition(train_state_lock)
train_thread_lock = threading.Lock()
train_fd = None
