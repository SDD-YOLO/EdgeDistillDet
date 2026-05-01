from __future__ import annotations

import csv
from pathlib import Path


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_csv_summary_from_disk(path: Path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
        return list(reader.fieldnames or []), rows
    except Exception:
        return [], []
