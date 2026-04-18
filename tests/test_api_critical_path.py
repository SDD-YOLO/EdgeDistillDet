"""最小关键路径：API 可导入、核心只读端点可用。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import api


def test_app_imports():
    assert api.title == "EdgeDistillDet Backend"


def test_get_configs_ok():
    client = TestClient(api)
    r = client.get("/api/configs")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "configs" in data


def test_get_train_status_ok():
    client = TestClient(api)
    r = client.get("/api/train/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_get_metrics_ok():
    client = TestClient(api)
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
