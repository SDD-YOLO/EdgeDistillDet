from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from web.services.backend_train import _validate_resume_locked_fields


def _write_args_yaml(run_dir: Path, data: str = "datasets/a.yaml", device: str = "0", imgsz: int = 640, batch: int = 16):
    payload = {
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "data": data,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_resume_locked_fields_match_ok(tmp_path: Path):
    run_dir = tmp_path / "runs" / "exp9"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "last.pt"
    ckpt.write_text("stub", encoding="utf-8")
    _write_args_yaml(run_dir)

    cfg = {
        "training": {"data_yaml": "datasets/a.yaml", "device": "0", "imgsz": 640, "batch": 16},
        "output": {"project": str(run_dir.parent), "name": run_dir.name},
    }
    _validate_resume_locked_fields(cfg, str(ckpt))


def test_resume_locked_fields_mismatch_raise(tmp_path: Path):
    run_dir = tmp_path / "runs" / "exp10"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "last.pt"
    ckpt.write_text("stub", encoding="utf-8")
    _write_args_yaml(run_dir, data="datasets/a.yaml", device="0", imgsz=640, batch=16)

    cfg = {
        "training": {"data_yaml": "datasets/b.yaml", "device": "cpu", "imgsz": 1280, "batch": 32},
        "output": {"project": str(run_dir.parent), "name": run_dir.name},
    }
    with pytest.raises(ValueError):
        _validate_resume_locked_fields(cfg, str(ckpt))
