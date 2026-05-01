from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock

from web.core.paths import BASE_DIR
from web.services.io.csv_io import load_csv_summary_from_disk


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
                CREATE TABLE IF NOT EXISTS csv_summary_cache (
                    path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    columns_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        _SCHEMA_READY = True


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def invalidate_csv_summary(path: Path) -> None:
    try:
        _ensure_schema()
        with _connect() as conn:
            conn.execute("DELETE FROM csv_summary_cache WHERE path = ?", (str(path.resolve()),))
            conn.commit()
    except Exception:
        return


def load_csv_summary_cached(path: Path):
    normalized = Path(path).resolve()
    if not normalized.exists() or not normalized.is_file():
        return [], []

    try:
        signature = _file_signature(normalized)
    except Exception:
        return load_csv_summary_from_disk(normalized)

    try:
        _ensure_schema()
        with _connect() as conn:
            row = conn.execute(
                "SELECT mtime_ns, size, columns_json, rows_json FROM csv_summary_cache WHERE path = ?",
                (str(normalized),),
            ).fetchone()
            if row and int(row["mtime_ns"]) == signature[0] and int(row["size"]) == signature[1]:
                try:
                    return json.loads(row["columns_json"]), json.loads(row["rows_json"])
                except Exception:
                    pass
    except Exception:
        pass

    columns, rows = load_csv_summary_from_disk(normalized)
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO csv_summary_cache (path, mtime_ns, size, columns_json, rows_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mtime_ns=excluded.mtime_ns,
                    size=excluded.size,
                    columns_json=excluded.columns_json,
                    rows_json=excluded.rows_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(normalized),
                    signature[0],
                    signature[1],
                    json.dumps(columns, ensure_ascii=False),
                    json.dumps(rows, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.commit()
    except Exception:
        pass
    return columns, rows
