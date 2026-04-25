from __future__ import annotations

from pathlib import Path

import yaml

from web.schemas import AgentToolExecuteRequest
from web.services import backend_agent


def _write_yaml(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_agent_tmp(monkeypatch, tmp_path):
    config_dir = tmp_path / "configs"
    history_dir = tmp_path / ".agent_state" / "run_history"
    monkeypatch.setattr(backend_agent, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(backend_agent, "AGENT_HISTORY_DIR", history_dir)
    monkeypatch.setattr(backend_agent, "_approval_store", {})
    monkeypatch.setattr(backend_agent, "_APPROVAL_TTL_SEC", 3600)
    _write_yaml(
        config_dir / "distill_config.yaml",
        {"distillation": {"w_kd": 0.5, "temperature": 2.0}, "training": {"epochs": 100}},
    )
    return config_dir, history_dir


def test_preview_apply_replay_guard(monkeypatch, tmp_path):
    config_dir, _ = _prepare_agent_tmp(monkeypatch, tmp_path)

    preview = backend_agent._tool_preview_patch(
        {
            "run_id": "default",
            "patch": {"distillation": {"w_kd": 0.6}},
            "operator": "test",
            "reason": "unit-test",
        }
    )
    assert preview["status"] == "ok"
    assert preview["approval_token"]
    assert preview["request_hash"]
    assert preview["change_summary"]["stats"]["changed"] == 1

    bad = backend_agent._tool_apply_patch_with_approval({"run_id": "default", "approval_token": "bad-token"})
    assert bad["status"] == "error"

    ok = backend_agent._tool_apply_patch_with_approval(
        {
            "run_id": "default",
            "approval_token": preview["approval_token"],
            "request_hash": preview["request_hash"],
            "operator": "tester",
            "reason": "apply",
        }
    )
    assert ok["status"] == "ok"
    cfg = _read_yaml(config_dir / "distill_config.yaml")
    assert cfg["distillation"]["w_kd"] == 0.6

    replay = backend_agent._tool_apply_patch_with_approval(
        {
            "run_id": "default",
            "approval_token": preview["approval_token"],
            "request_hash": preview["request_hash"],
        }
    )
    assert replay["status"] == "error"
    assert replay["error"] == "E_APPROVAL_REPLAY"


def test_rollback_to_previous_version(monkeypatch, tmp_path):
    config_dir, _ = _prepare_agent_tmp(monkeypatch, tmp_path)

    p1 = backend_agent._tool_preview_patch({"run_id": "r1", "patch": {"distillation": {"w_kd": 0.6}}})
    backend_agent._tool_apply_patch_with_approval(
        {"run_id": "r1", "approval_token": p1["approval_token"], "request_hash": p1["request_hash"]}
    )
    p2 = backend_agent._tool_preview_patch({"run_id": "r1", "patch": {"distillation": {"w_kd": 0.7}}})
    backend_agent._tool_apply_patch_with_approval(
        {"run_id": "r1", "approval_token": p2["approval_token"], "request_hash": p2["request_hash"]}
    )
    cfg = _read_yaml(config_dir / "distill_config.yaml")
    assert cfg["distillation"]["w_kd"] == 0.7

    rollback = backend_agent._tool_rollback_run_config({"run_id": "r1", "steps": 1})
    assert rollback["status"] == "ok"
    cfg_after = _read_yaml(config_dir / "distill_config.yaml")
    assert cfg_after["distillation"]["w_kd"] == 0.6


def test_tools_execute_preview_contract(monkeypatch, tmp_path):
    _prepare_agent_tmp(monkeypatch, tmp_path)
    req = AgentToolExecuteRequest(
        tool="agent.preview_patch",
        args={"run_id": "graph-run", "patch": {"training": {"epochs": 120}}},
    )
    out = backend_agent.agent_tools_execute(req)
    assert out["status"] == "ok"
    assert out["approval_token"]
    assert out["request_hash"]


def test_preview_rejects_deprecated_fields(monkeypatch, tmp_path):
    _prepare_agent_tmp(monkeypatch, tmp_path)
    out = backend_agent._tool_preview_patch(
        {
            "run_id": "default",
            "patch": {"training": {"workers": 2}},
        }
    )
    assert out["status"] == "error"
    assert "deprecated" in out["error"]
    assert "training.workers" in out["deprecated_paths"]


