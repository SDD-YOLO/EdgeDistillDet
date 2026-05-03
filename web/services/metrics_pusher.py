"""Lightweight metrics pusher used by training callbacks.

Writes a persisted JSONL record and attempts to broadcast to WebSocket manager
in a non-blocking way so it is safe to call from synchronous training code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

from web.services.ws_manager import manager as ws_manager

LOG = logging.getLogger(__name__)


def push_metrics(epoch: int, metrics: dict[str, Any], training_id: str = "default") -> None:
    """Persist metrics and attempt to broadcast to connected WS clients.

    This function is intentionally fire-and-forget so it can be invoked from
    synchronous training loops.
    """
    message = {
        "type": "metrics",
        "timestamp": time.time(),
        "data": {"training_id": training_id, "epoch": epoch, "metrics": metrics},
    }

    # Persist to a simple JSONL for durability (append-only)
    try:
        import os

        os.makedirs("data", exist_ok=True)
        with open("data/metrics_stream.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception:
        LOG.exception("Failed to persist metrics to data/metrics_stream.jsonl")

    # Try to broadcast asynchronously. If we're in an asyncio loop, schedule
    # a task; otherwise spawn a short-lived thread that runs the coroutine.
    try:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            asyncio.create_task(ws_manager.broadcast(message))
        else:

            def _runner():
                try:
                    asyncio.run(ws_manager.broadcast(message))
                except Exception:
                    LOG.exception("ws broadcast failed in background thread")

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
    except Exception:
        LOG.exception("Failed to enqueue metrics for broadcast")
