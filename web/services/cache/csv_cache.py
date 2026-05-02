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
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS csv_rows_cache (
                    path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    row_index INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    PRIMARY KEY (path, row_index)
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



def load_csv_rows_cached(path: Path, max_rows: int | None = None) -> list[dict]:
    """
    从SQLite缓存读取CSV行数据
    如果缓存过期或不存在，从磁盘读取并缓存
    """
    normalized = Path(path).resolve()
    if not normalized.exists() or not normalized.is_file():
        return []

    try:
        signature = _file_signature(normalized)
    except Exception:
        return load_csv_summary_from_disk(normalized)[1]

    try:
        _ensure_schema()
        with _connect() as conn:
            # 检查缓存是否有效
            rows = conn.execute(
                """
                SELECT row_json FROM csv_rows_cache
                WHERE path = ? AND mtime_ns = ?
                ORDER BY row_index
                """,
                (str(normalized), signature[0]),
            ).fetchall()

            if rows:
                cached_rows = [json.loads(row["row_json"]) for row in rows]
                if max_rows is None or len(cached_rows) <= max_rows:
                    return cached_rows
                return cached_rows[-max_rows:] if max_rows > 0 else []
    except Exception:
        pass

    # 从磁盘读取
    columns, rows = load_csv_summary_from_disk(normalized)

    # 缓存到SQLite
    try:
        _ensure_schema()
        with _connect() as conn:
            # 清理旧缓存
            conn.execute("DELETE FROM csv_rows_cache WHERE path = ?", (str(normalized),))

            # 插入新行
            for idx, row in enumerate(rows):
                conn.execute(
                    """
                    INSERT INTO csv_rows_cache (path, mtime_ns, row_index, row_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(normalized), signature[0], idx, json.dumps(row, ensure_ascii=False)),
                )
            conn.commit()
    except Exception:
        pass

    if max_rows is None or len(rows) <= max_rows:
        return rows
    return rows[-max_rows:] if max_rows > 0 else []


def invalidate_csv_rows(path: Path) -> None:
    """清空特定CSV的行缓存"""
    try:
        _ensure_schema()
        with _connect() as conn:
            conn.execute("DELETE FROM csv_rows_cache WHERE path = ?", (str(path.resolve()),))
            conn.commit()
    except Exception:
        return


def load_csv_rows_range_cached(path: Path, tail: int = 100) -> list[dict]:
    """
    从SQLite缓存读取CSV最后N行
    用于agent的增量查询
    """
    normalized = Path(path).resolve()
    if not normalized.exists() or not normalized.is_file():
        return []

    try:
        signature = _file_signature(normalized)
    except Exception:
        _, rows = load_csv_summary_from_disk(normalized)
        return rows[-tail:] if tail > 0 else []

    try:
        _ensure_schema()
        with _connect() as conn:
            # 获取总行数
            count_result = conn.execute(
                "SELECT COUNT(*) as cnt FROM csv_rows_cache WHERE path = ? AND mtime_ns = ?",
                (str(normalized), signature[0]),
            ).fetchone()

            total = count_result["cnt"] if count_result else 0
            offset = max(0, total - tail)

            # 获取后tail行
            rows = conn.execute(
                """
                SELECT row_json FROM csv_rows_cache
                WHERE path = ? AND mtime_ns = ?
                ORDER BY row_index
                LIMIT ? OFFSET ?
                """,
                (str(normalized), signature[0], tail, offset),
            ).fetchall()

            if rows:
                return [json.loads(row["row_json"]) for row in rows]
    except Exception:
        pass

    # 从磁盘读取并缓存
    _, rows = load_csv_summary_from_disk(normalized)
    try:
        _ensure_schema()
        with _connect() as conn:
            conn.execute("DELETE FROM csv_rows_cache WHERE path = ?", (str(normalized),))
            for idx, row in enumerate(rows):
                conn.execute(
                    """
                    INSERT INTO csv_rows_cache (path, mtime_ns, row_index, row_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(normalized), signature[0], idx, json.dumps(row, ensure_ascii=False)),
                )
            conn.commit()
    except Exception:
        pass

    return rows[-tail:] if tail > 0 else []
