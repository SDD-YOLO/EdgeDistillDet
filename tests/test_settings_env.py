from __future__ import annotations

from web.core import settings


def test_cors_defaults_follow_ports(monkeypatch):
    monkeypatch.delenv("EDGE_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("EDGE_BACKEND_PORT", "6000")
    monkeypatch.setenv("EDGE_FRONTEND_DEV_PORT", "5178")
    cfg = settings.get_cors_middleware_kwargs()
    origins = cfg.get("allow_origins") or []
    assert "http://127.0.0.1:6000" in origins
    assert "http://localhost:6000" in origins
    assert "http://127.0.0.1:5178" in origins
    assert "http://localhost:5178" in origins
