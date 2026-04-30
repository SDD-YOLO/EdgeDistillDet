from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fastapi import Query

from web.core.paths import BASE_DIR
from web.services.backend_common import (
    _as_float,
    _build_metric_series,
    _estimate_run_stats,
    _load_csv_summary,
    _resolve_column_name,
    _summarize_series,
)


def _calc_total_time_with_resumes(rows: list[dict]) -> tuple[float | None, int]:
    """
    通过 time 列“重置”来分段累计总时长，覆盖多次断点续练场景。
    返回: (total_seconds, reset_count)
    """
    segment_max = 0.0
    total = 0.0
    reset_count = 0
    prev_time = None
    has_any = False
    for row in rows:
        time_val = _as_float(row.get("time"))
        if time_val is None:
            continue
        has_any = True
        tv = float(time_val)
        # time 显著回退视为进入新的续训片段；累计前一段最大值
        if prev_time is not None and tv + 1e-9 < prev_time:
            total += segment_max
            segment_max = 0.0
            reset_count += 1
        if tv > segment_max:
            segment_max = tv
        prev_time = tv
    if not has_any:
        return None, 0
    total += segment_max
    return total, reset_count


def get_metrics(source: str = Query('')):
    runs_dir = BASE_DIR / 'runs'
    base_resolved = Path(BASE_DIR).resolve()
    metrics_data = []
    if runs_dir.exists():
        for root, dirs, files in os.walk(str(runs_dir)):
            for file_name in files:
                if file_name != 'results.csv':
                    continue
                result_path = Path(root) / file_name
                run_dir = result_path.parent
                try:
                    display_name = f"{run_dir.name} @ {datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
                    rel_dir = str(run_dir.resolve().relative_to(base_resolved))
                    rel_path = str(result_path.resolve().relative_to(base_resolved))
                except (ValueError, OSError):
                    continue
                entry = {
                    "name": run_dir.name,
                    "display_name": display_name,
                    "dir": rel_dir,
                    "has_results": True,
                    "path": rel_path
                }
                try:
                    columns, rows = _load_csv_summary(result_path)
                    entry["columns"] = columns
                    entry["rows"] = len(rows)
                except Exception:
                    entry["columns"] = []
                    entry["rows"] = 0
                metrics_data.append(entry)
        try:
            metrics_data.sort(key=lambda item: os.path.getmtime(str(base_resolved / item['dir'])), reverse=True)
        except Exception:
            pass

    selected_path = source.strip()
    selected_data = None
    if selected_path:
        try:
            target = (BASE_DIR / selected_path).resolve()
            if target.exists() and target.is_file() and target.suffix == '.csv' and str(target).startswith(str(base_resolved)):
                columns, rows = _load_csv_summary(target)
                if rows:
                    chart_series = _build_metric_series(rows, columns, target.parent)
                    summary_metrics = {}
                    _summary_metric_map = [
                        ('box_loss', ['train/box_loss', 'box_loss', 'train_box_loss'], 'lower'),
                        ('cls_loss', ['train/cls_loss', 'cls_loss', 'train_cls_loss'], 'lower'),
                        ('dfl_loss', ['train/dfl_loss', 'dfl_loss', 'train_dfl_loss'], 'lower'),
                        ('map50', ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50', 'map50'], 'higher'),
                        ('map50_95', ['metrics/mAP50-95(B)', 'metrics/mAP50-95', 'mAP50-95', 'map50_95'], 'higher'),
                        ('precision', ['metrics/precision(B)', 'metrics/precision', 'precision'], 'higher'),
                        ('recall', ['metrics/recall(B)', 'metrics/recall', 'recall'], 'higher'),
                    ]
                    for key, aliases, better in _summary_metric_map:
                        actual_col = _resolve_column_name(columns, aliases)
                        if not actual_col:
                            continue
                        try:
                            s = _summarize_series(rows, actual_col, better=better)
                            if s is not None:
                                summary_metrics[key] = s
                        except Exception:
                            pass
                    total_time = None
                    for row in reversed(rows):
                        total_time = _as_float(row.get('time'))
                        if total_time is not None:
                            break

                    reset_count = 0
                    segment_time_sum = 0.0
                    segment_time_max = 0.0
                    prev_epoch = None
                    for row in rows:
                        epoch_val = _as_float(row.get("epoch"))
                        time_val = _as_float(row.get("time"))
                        if epoch_val is not None and prev_epoch is not None and epoch_val < prev_epoch:
                            reset_count += 1
                            segment_time_sum += segment_time_max
                            segment_time_max = 0.0
                        if time_val is not None:
                            segment_time_max = max(segment_time_max, float(time_val))
                        if epoch_val is not None:
                            prev_epoch = epoch_val
                    segment_time_sum += segment_time_max

                    run_stats = _estimate_run_stats(target.parent)
                    ov_map50 = '--'
                    if summary_metrics.get('map50'):
                        try:
                            ov_map50 = f"{(summary_metrics['map50']['best'] * 100):.2f}%"
                        except Exception:
                            pass
                    accumulated_time, time_reset_count = _calc_total_time_with_resumes(rows)
                    ov_time = '--'
                    if accumulated_time is not None:
                        ov_time = f"{int(accumulated_time // 60)}m {int(accumulated_time % 60)}s"
                    elif total_time is not None:
                        ov_time = f"{int(total_time // 60)}m {int(total_time % 60)}s"
                    selected_data = {
                        'source': selected_path,
                        'columns': columns,
                        'rows': len(rows),
                        'chart_series': chart_series,
                        'summary_metrics': summary_metrics,
                        'overview_stats': {
                            'ov-map50': ov_map50,
                            **run_stats,
                            'ov-time': ov_time
                        }
                    }
        except Exception as e:
            import traceback
            traceback.print_exc()
            selected_data = {'error': str(e)}

    response = {
        "status": "ok",
        "csv_metrics": metrics_data
    }
    if selected_data:
        response.update(selected_data)
    return response
