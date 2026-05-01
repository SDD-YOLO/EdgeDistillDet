from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock

from web.core.paths import BASE_DIR


DB_PATH = BASE_DIR / "data" / "metrics_cache.sqlite3"
_SCHEMA_LOCK = RLock()
_SCHEMA_READY = False


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_manifest_cache (
                    path TEXT PRIMARY KEY,
                    run_dir TEXT NOT NULL,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    dir TEXT NOT NULL,
                    has_results INTEGER NOT NULL,
                    columns_json TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    result_mtime_ns INTEGER NOT NULL,
                    run_mtime_ns INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        _SCHEMA_READY = True


def _entry_signature(entry: dict) -> tuple[int, int]:
    return int(entry.get("result_mtime_ns") or 0), int(entry.get("run_mtime_ns") or 0)


def store_metrics_index(entries: list[dict]) -> None:
    if not entries:
        return
    try:
        _ensure_schema()
        with _connect() as conn:
            for entry in entries:
                conn.execute(
                    """
                    INSERT INTO result_manifest_cache (
                        path, run_dir, name, display_name, dir, has_results,
                        columns_json, rows, result_mtime_ns, run_mtime_ns, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        run_dir=excluded.run_dir,
                        name=excluded.name,
                        display_name=excluded.display_name,
                        dir=excluded.dir,
                        has_results=excluded.has_results,
                        columns_json=excluded.columns_json,
                        rows=excluded.rows,
                        result_mtime_ns=excluded.result_mtime_ns,
                        run_mtime_ns=excluded.run_mtime_ns,
                        updated_at=excluded.updated_at
                    """,
                    (
                        str(entry.get("path") or ""),
                        str(entry.get("run_dir") or entry.get("dir") or ""),
                        str(entry.get("name") or ""),
                        str(entry.get("display_name") or ""),
                        str(entry.get("dir") or ""),
                        1 if entry.get("has_results") else 0,
                        json.dumps(entry.get("columns") or [], ensure_ascii=False),
                        int(entry.get("rows") or 0),
                        int(entry.get("result_mtime_ns") or 0),
                        int(entry.get("run_mtime_ns") or 0),
                        time.time(),
                    ),
                )
            conn.commit()
    except Exception:
        return


def load_cached_metrics_index(max_age_seconds: int = 30) -> list[dict] | None:
    try:
        _ensure_schema()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM result_manifest_cache ORDER BY updated_at DESC"
            ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    now = time.time()
    cached = []
    for row in rows:
        entry = {
            "name": row["name"],
            "display_name": row["display_name"],
            "dir": row["dir"],
            "run_dir": row["run_dir"],
            "has_results": bool(row["has_results"]),
            "path": row["path"],
            "columns": json.loads(row["columns_json"]),
            "rows": int(row["rows"]),
            "result_mtime_ns": int(row["result_mtime_ns"]),
            "run_mtime_ns": int(row["run_mtime_ns"]),
            "modified_time": max(int(row["result_mtime_ns"]), int(row["run_mtime_ns"])) / 1_000_000_000,
        }
        try:
            result_path = Path(entry["path"])
            run_dir = Path(entry["run_dir"])
            if not result_path.exists() or not run_dir.exists():
                return None
            result_stat = result_path.stat()
            run_stat = run_dir.stat()
            if _entry_signature(entry) != (int(result_stat.st_mtime_ns), int(run_stat.st_mtime_ns)):
                return None
        except Exception:
            return None
        cached.append(entry)

    latest_updated = max((float(row["updated_at"]) for row in rows), default=0.0)
    if max_age_seconds > 0 and now - latest_updated > max_age_seconds:
        return None
    cached.sort(key=lambda item: item["modified_time"], reverse=True)
    return cached
