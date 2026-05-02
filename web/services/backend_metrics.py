from __future__ import annotations

from pathlib import Path

from web.core.paths import BASE_DIR
from web.services.backend_common import (
    _as_float,
    _build_metric_series,
    _estimate_run_stats,
    _load_csv_summary,
    _resolve_column_name,
    _summarize_series,
)
from web.services.cache.result_manifest_cache import load_cached_metrics_index, store_metrics_index
from web.services.cache.csv_cache import load_csv_summary_cached, load_csv_rows_cached
from web.services.scan.results_discovery import describe_result_csv, discover_results_csvs, get_candidate_runs_directories


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


def get_metrics(source: str = ''):
    """
    获取所有训练结果的指标汇总及详细数据。
    
    Args:
        source: 指定具体的 results.csv 路径（相对于项目根目录）
    
    Returns:
        {
            "metrics": [...],  # 所有找到的 results.csv 摘要
            "selected": {...}  # 如果指定了 source，返回详细数据
        }
    """
    base_resolved = Path(BASE_DIR).resolve()
    metrics_data = []

    runs_directories = get_candidate_runs_directories(BASE_DIR)
    cached_metrics = load_cached_metrics_index()
    if cached_metrics is not None:
        metrics_data = cached_metrics
    else:
        for result_path in discover_results_csvs(runs_directories):
            entry = describe_result_csv(result_path, base_resolved)
            if entry is not None:
                metrics_data.append(entry)
        metrics_data.sort(key=lambda item: item.get('modified_time', 0.0), reverse=True)
        store_metrics_index(metrics_data)

    selected_path = source.strip()
    selected_data = None
    if selected_path:
        try:
            target = (BASE_DIR / selected_path).resolve()
            if target.exists() and target.is_file() and target.suffix == '.csv' and str(target).startswith(str(base_resolved)):
                columns, _ = load_csv_summary_cached(target)
                rows = load_csv_rows_cached(target)
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
