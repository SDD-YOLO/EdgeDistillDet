from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fastapi import Query

from web.core.workspace import get_data_root
from web.services.backend_common import _as_float, _build_metric_series, _estimate_run_stats, _load_csv_summary, _summarize_series

def get_metrics(source: str = Query('')):
    root = get_data_root()
    runs_dir = root / 'runs'
    base_resolved = Path(root).resolve()
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
            target = (root / selected_path).resolve()
            if target.exists() and target.is_file() and target.suffix == '.csv' and str(target).startswith(str(base_resolved)):
                columns, rows = _load_csv_summary(target)
                if rows:
                    chart_series = _build_metric_series(rows, columns, target.parent)
                    summary_metrics = {}
                    for key, col, better in [
                        ('box_loss', 'train/box_loss', 'lower'),
                        ('cls_loss', 'train/cls_loss', 'lower'),
                        ('dfl_loss', 'train/dfl_loss', 'lower'),
                        ('map50', 'metrics/mAP50(B)', 'higher'),
                        ('map50_95', 'metrics/mAP50-95(B)', 'higher'),
                        ('precision', 'metrics/precision(B)', 'higher'),
                        ('recall', 'metrics/recall(B)', 'higher'),
                    ]:
                        try:
                            s = _summarize_series(rows, col, better=better)
                            if s is not None:
                                summary_metrics[key] = s
                        except Exception:
                            pass
                    total_time = None
                    for row in reversed(rows):
                        total_time = _as_float(row.get('time'))
                        if total_time is not None:
                            break

                    run_stats = _estimate_run_stats(target.parent)
                    ov_map50 = '--'
                    if summary_metrics.get('map50'):
                        try:
                            ov_map50 = f"{(summary_metrics['map50']['best'] * 100):.2f}%"
                        except Exception:
                            pass
                    ov_time = '--'
                    if total_time is not None:
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
