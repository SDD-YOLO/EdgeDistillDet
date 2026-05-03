from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from web.core.paths import BASE_DIR, CONFIG_DIR
from web.services.cache.csv_cache import load_csv_rows_range_cached

_TEXT_FILE_PATTERNS = (
    "*.md",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.json",
    "*.py",
    "*.jsx",
    "*.js",
)
_MAX_FILES_PER_PATTERN = 20
_MAX_FILE_CHARS = 6000


def _tokenize(text: str) -> list[str]:
    parts = re.split(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", str(text or "").lower())
    return [p for p in parts if len(p) >= 2]


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_project_candidates(query: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pattern in _TEXT_FILE_PATTERNS:
        count = 0
        for path in BASE_DIR.rglob(pattern):
            if any(seg in (".git", "node_modules", "__pycache__", ".venv") for seg in path.parts):
                continue
            text = _read_text_safe(path)
            if not text:
                continue
            count += 1
            candidates.append(
                {
                    "source": "project",
                    "path": str(path.relative_to(BASE_DIR)),
                    "text": text[:_MAX_FILE_CHARS],
                }
            )
            if count >= _MAX_FILES_PER_PATTERN:
                break
    return candidates


def _collect_training_candidates(run_id: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    cfg_path = CONFIG_DIR / "distill_config.yaml"
    if cfg_path.exists():
        text = _read_text_safe(cfg_path)
        if text:
            candidates.append(
                {
                    "source": "training",
                    "path": "configs/distill_config.yaml",
                    "text": text[:_MAX_FILE_CHARS],
                }
            )

    results_path = BASE_DIR / "runs" / "detect" / "runs" / "distill" / str(run_id or "exp1") / "results.csv"
    if results_path.exists():
        rows = load_csv_rows_range_cached(results_path, tail=8)
        if rows:
            candidates.append(
                {
                    "source": "training",
                    "path": str(results_path.relative_to(BASE_DIR)),
                    "text": json.dumps(rows, ensure_ascii=False),
                }
            )
    return candidates


def _score(query_tokens: list[str], text: str, source: str) -> float:
    hay = str(text or "").lower()
    if not hay or not query_tokens:
        return 0.0
    freq = sum(hay.count(tok) for tok in query_tokens)
    if source == "training":
        return float(freq) + 1.2
    return float(freq) + 0.7


def retrieve_hybrid(query: str, *, run_id: str = "default", top_k: int = 5) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query)
    docs = _collect_training_candidates(run_id) + _collect_project_candidates(query)
    scored: list[dict[str, Any]] = []
    for doc in docs:
        score = _score(query_tokens, doc.get("text", ""), doc.get("source", "project"))
        if score <= 0:
            continue
        text = str(doc.get("text") or "")
        snippet = text[:220].replace("\n", " ").strip()
        scored.append(
            {
                "source": doc.get("source"),
                "path": doc.get("path"),
                "score": round(score, 3),
                "snippet": snippet,
            }
        )
    scored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return scored[: max(1, int(top_k or 5))]
