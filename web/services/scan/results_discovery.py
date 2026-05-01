from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from web.core.paths import BASE_DIR
from web.services.cache.csv_cache import load_csv_summary_cached


def get_candidate_runs_directories(base_dir: Path | None = None) -> list[Path]:
    base_dir = Path(base_dir or BASE_DIR).resolve()
    candidates: list[Path] = []

    default_runs = base_dir / "runs"
    if default_runs.exists():
        candidates.append(default_runs)

    detect_runs = base_dir / "runs" / "detect"
    if detect_runs.exists() and detect_runs not in candidates:
        candidates.append(detect_runs)

    custom_runs = os.environ.get("EDGE_RUNS_DIRS", "").strip()
    if custom_runs:
        for path_str in custom_runs.split(";"):
            path_str = path_str.strip()
            if not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                path = base_dir / path
            if path.exists() and path not in candidates:
                candidates.append(path)

    return candidates


def discover_results_csvs(runs_directories: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for runs_dir in runs_directories:
        if not runs_dir.exists():
            continue
        for root, _dirs, files in os.walk(str(runs_dir)):
            if "results.csv" not in files:
                continue
            result_path = (Path(root) / "results.csv").resolve()
            if result_path in seen:
                continue
            seen.add(result_path)
            discovered.append(result_path)
    discovered.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)
    return discovered


def describe_result_csv(result_path: Path, base_resolved: Path) -> dict | None:
    try:
        run_dir = result_path.parent.resolve()
        display_name = f"{run_dir.name} @ {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        rel_dir = str(run_dir.relative_to(base_resolved))
        rel_path = str(result_path.resolve().relative_to(base_resolved))
        columns, rows = load_csv_summary_cached(result_path)
        result_stat = result_path.stat()
        run_stat = run_dir.stat()
    except Exception:
        return None

    return {
        "name": run_dir.name,
        "display_name": display_name,
        "dir": rel_dir,
        "run_dir": str(run_dir),
        "has_results": True,
        "path": rel_path,
        "columns": columns,
        "rows": len(rows),
        "result_mtime_ns": int(result_stat.st_mtime_ns),
        "run_mtime_ns": int(run_stat.st_mtime_ns),
        "modified_time": max(result_stat.st_mtime, run_stat.st_mtime),
    }
