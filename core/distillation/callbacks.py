"""Training callbacks for distillation/training integration.

This module provides a minimal `MetricsCallback` that can be plugged into a
training loop (or Ultralytics callbacks) to push epoch metrics into the web
UI via `web.services.metrics_pusher.push_metrics`.
"""

from __future__ import annotations

from typing import Any

from web.services.metrics_pusher import push_metrics


class MetricsCallback:
    """Simple callback that pushes epoch-level metrics.

    Usage (pseudo):
        callback = MetricsCallback(training_id="run-123")
        callback.on_train_epoch_end(trainer)
    """

    def __init__(self, training_id: str = "default") -> None:
        self.training_id = training_id

    def on_train_epoch_end(self, trainer: Any) -> None:
        epoch = getattr(trainer, "epoch", None)
        metrics = getattr(trainer, "metrics", None) or {}
        # Ensure epoch is int when possible
        try:
            epoch = int(epoch) if epoch is not None else None
        except Exception:
            pass
        push_metrics(epoch, metrics, training_id=self.training_id)

    # Backwards-compatible alias
    def on_epoch_end(self, epoch: int, metrics: dict | None = None) -> None:
        push_metrics(epoch, metrics or {}, training_id=self.training_id)
