from __future__ import annotations

from web.services.cache import csv_cache, result_manifest_cache
from web.services.cache.csv_cache import load_csv_summary_cached
from web.services.cache.result_manifest_cache import (
    load_cached_metrics_index,
    store_metrics_index,
)


def test_csv_summary_cache_refreshes_when_file_changes(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics_cache.sqlite3"
    monkeypatch.setattr(csv_cache, "DB_PATH", db_path)
    monkeypatch.setattr(csv_cache, "_SCHEMA_READY", False)

    csv_file = tmp_path / "results.csv"
    csv_file.write_text("epoch,metrics/mAP50\n0,0.1\n1,0.2\n", encoding="utf-8")

    columns_1, rows_1 = load_csv_summary_cached(csv_file)
    assert columns_1 == ["epoch", "metrics/mAP50"]
    assert len(rows_1) == 2

    csv_file.write_text("epoch,metrics/mAP50\n0,0.3\n", encoding="utf-8")
    columns_2, rows_2 = load_csv_summary_cached(csv_file)

    assert columns_2 == ["epoch", "metrics/mAP50"]
    assert len(rows_2) == 1
    assert rows_2[0]["metrics/mAP50"] == "0.3"


def test_result_manifest_cache_invalidates_on_mtime_change(tmp_path, monkeypatch):
    db_path = tmp_path / "metrics_cache.sqlite3"
    monkeypatch.setattr(result_manifest_cache, "DB_PATH", db_path)
    monkeypatch.setattr(result_manifest_cache, "_SCHEMA_READY", False)

    run_dir = tmp_path / "exp1"
    run_dir.mkdir()
    csv_file = run_dir / "results.csv"
    csv_file.write_text("epoch,metrics/mAP50\n0,0.1\n", encoding="utf-8")

    result_stat = csv_file.stat()
    run_stat = run_dir.stat()
    entries = [
        {
            "name": "exp1",
            "display_name": "exp1 @ 2026-05-01 00:00:00",
            "dir": str(run_dir),
            "run_dir": str(run_dir),
            "has_results": True,
            "path": str(csv_file),
            "columns": ["epoch", "metrics/mAP50"],
            "rows": 1,
            "result_mtime_ns": int(result_stat.st_mtime_ns),
            "run_mtime_ns": int(run_stat.st_mtime_ns),
            "modified_time": max(result_stat.st_mtime, run_stat.st_mtime),
        }
    ]

    store_metrics_index(entries)
    cached = load_cached_metrics_index(max_age_seconds=3600)
    assert cached is not None
    assert cached[0]["name"] == "exp1"

    csv_file.write_text("epoch,metrics/mAP50\n0,0.2\n", encoding="utf-8")
    assert load_cached_metrics_index(max_age_seconds=3600) is None
