"""Centralized logging configuration and helper logger.

Use ``from core.logging import get_logger`` and call ``get_logger(__name__)``.
This module configures a single readable console format for both stdlib logging
and structlog-based loggers so terminal output stays consistent and detailed.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def _configure_structlog() -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Simple human-friendly renderer for console output
    def _simple_renderer(_, __, event_dict):
        ts = event_dict.get("timestamp") or event_dict.get("time") or ""
        level = str(event_dict.get("level") or "").upper()
        logger_name = event_dict.get("logger") or event_dict.get("logger_name") or ""
        event = event_dict.get("event", "")
        if isinstance(event, dict):
            event = event.get("message") or event.get("msg") or str(event)
        return f"{ts} [{level}] {logger_name} — {event}"

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, _simple_renderer],
        foreign_pre_chain=shared_processors,
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = next((h for h in root.handlers if isinstance(h, logging.StreamHandler)), None)
    if handler is None:
        handler = logging.StreamHandler(stream=sys.stdout)
        root.addHandler(handler)
    handler.setFormatter(formatter)


def init_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _configure_structlog()
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    try:
        return structlog.get_logger(name)
    except Exception:
        # Not configured or older structlog; initialize then return
        init_logging()
        return structlog.get_logger(name)


# module-level logger
logger = get_logger(__name__)
