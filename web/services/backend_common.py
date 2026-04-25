from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import yaml
from fastapi.responses import JSONResponse

from core.model_metrics import estimate_gflops_from_weight, estimate_params_m_from_checkpoint
from utils import expand_env_vars
from web.core.paths import BASE_DIR

try:
    from utils.edge_profiler import EdgeProfiler
except Exception:
    EdgeProfiler = None

# Ultralytics 单次 run 下与 weights/ 并列的产物目录，不是「另一个 exp」；
# 若把它们算进 sibling_exp_dirs，会阻止把当前目录识别为 run 根，导致只跳过 weights 后漏检或误扫。
_CHECKPOINT_SCAN_SKIP_SUBDIRS = frozenset({
    'val',
    'predict',
    'weights',
    'plots',
    'wandb',
    'images',
    'assets',
    'inference_results',
    'test',
    'debug',
    'samples',
    'calibration',
    'confusion_matrix',
})

def _is_warning_like(text: str) -> bool:
    return bool(re.search(r'(\bwarn(ing)?\b|警告|告警|⚠|\[W\]|^\s*W\d*:|\bignoring\b|忽略|已忽略|\bdeprecated\b)', str(text or ''), re.IGNORECASE))

def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None

def _resolve_project_path(project: str, allow_external: bool = False) -> Path:
    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = (BASE_DIR / project_path).resolve()
    else:
        project_path = project_path.resolve()
    if allow_external:
        return project_path
    if str(project_path).startswith(str(Path(BASE_DIR).resolve())) is False and str(Path(BASE_DIR).resolve()) not in str(project_path):
        raise ValueError('项目目录必须在仓库根目录下')
    return project_path

def _normalize_compute_provider(value: str | None) -> str:
    """统一算力平台标识，兼容多种写法。"""
    v = str(value or '').strip().lower()
    if v in {'google colab', 'google_colab', 'colab'}:
        return 'colab'
    if v in {'autodl', 'auto_dl', 'auto-dl'}:
        return 'autodl'
    if v in {'remote_api', 'remote-api', 'cloud_api', 'cloud-api'}:
        return 'remote_api'
    return 'local'

def _candidate_output_roots(project_path: Path):
    candidates = [project_path.resolve()]
    try:
        rel = project_path.resolve().relative_to(BASE_DIR.resolve())
    except Exception:
        rel = None
    if rel is None:
        return candidates

    for prefix in ('runs/detect', 'detect/runs'):
        alt = (BASE_DIR / prefix / rel).resolve()
        if str(alt).startswith(str(BASE_DIR.resolve())) and alt not in candidates:
            candidates.append(alt)

    if str(rel).replace('\\', '/') == 'runs':
        fallback = (BASE_DIR / 'runs' / 'detect' / 'runs').resolve()
        if str(fallback).startswith(str(BASE_DIR.resolve())) and fallback not in candidates:
            candidates.append(fallback)
    # 兼容旧训练产物目录：当 project 是 runs/distill 时，历史结果常落在 runs/detect/runs
    if str(rel).replace('\\', '/') == 'runs/distill':
        legacy = (BASE_DIR / 'runs' / 'detect' / 'runs').resolve()
        if str(legacy).startswith(str(BASE_DIR.resolve())) and legacy not in candidates:
            candidates.append(legacy)

    return candidates

def _load_yaml_file(path: Path):
    last_error = None
    for encoding in ('utf-8', 'utf-8-sig', 'cp936', 'gbk', 'latin1'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                cfg = yaml.safe_load(f)
                return expand_env_vars(cfg) if cfg is not None else None
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except FileNotFoundError:
            return None
        except Exception:
            return None
    return None

def _save_yaml_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def _list_resume_candidates(project_path: Path):
    candidates = []
    if not project_path.exists() or not project_path.is_dir():
        for prefix in ['runs/detect', 'detect/runs', 'runs/detect/' + project_path.name, 'detect/' + project_path.name, 'detect']:
            fallback = (BASE_DIR / prefix).resolve()
            if fallback.exists() and fallback.is_dir():
                candidates.extend(_scan_run_dirs_for_checkpoints(fallback, project_path))
            if project_path.name:
                alt_fallback = (BASE_DIR / prefix).resolve()
                if alt_fallback.exists() and alt_fallback.is_dir() and alt_fallback != fallback:
                    candidates.extend(_scan_run_dirs_for_checkpoints(alt_fallback, project_path))
        # 宽泛搜索：直接在 runs/ 下递归查找含 checkpoint 的目录
        if not candidates:
            runs_dir = BASE_DIR / 'runs'
            if runs_dir.exists() and runs_dir.is_dir():
                candidates.extend(_scan_run_dirs_for_checkpoints(runs_dir, project_path))
        return candidates

    candidates.extend(_scan_run_dirs_for_checkpoints(project_path, project_path))

    for prefix in ['runs/detect', 'detect/runs', 'detect']:
        detect_project = (BASE_DIR / prefix / project_path.relative_to(BASE_DIR)).resolve()
        if detect_project != project_path and detect_project.exists() and detect_project.is_dir():
            existing_names = {c['name'] for c in candidates}
            new_candidates = _scan_run_dirs_for_checkpoints(detect_project, project_path)
            for nc in new_candidates:
                if nc['name'] not in existing_names:
                    candidates.append(nc)
                    existing_names.add(nc['name'])

    candidates.sort(key=lambda c: c['modified_time'], reverse=True)
    return candidates

def _scan_run_dirs_for_checkpoints(search_path: Path, logical_project: Path):
    candidates = []
    if not search_path.exists() or not search_path.is_dir():
        return candidates

    ckpt_rel = [
        ('last.pt', 'last.pt'),
        ('weights/last.pt', 'weights/last.pt'),
        ('weights/best.pt', 'weights/best.pt'),
    ]

    def first_checkpoints(run_root: Path):
        found = []
        for rel_path, label in ckpt_rel:
            candidate = run_root / rel_path
            if candidate.exists():
                found.append((candidate, label))
        return found

    def append_candidate(run_root: Path, checkpoint_path: Path, checkpoint_label: str):
        try:
            ck_mtime = checkpoint_path.stat().st_mtime
        except OSError:
            ck_mtime = 0.0
        try:
            dir_mtime = run_root.stat().st_mtime
        except OSError:
            dir_mtime = 0.0
        eff_mtime = max(ck_mtime, dir_mtime)
        candidates.append({
            'name': run_root.name,
            'project': str(logical_project.relative_to(BASE_DIR)),
            'dir': str(run_root.relative_to(BASE_DIR)),
            'display_name': f"{run_root.name} — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eff_mtime))} ({checkpoint_label})",
            'checkpoint': str(checkpoint_path.resolve()).replace('\\', '/'),
            'checkpoint_name': checkpoint_label,
            'modified_time': eff_mtime,
        })

    # 递归到项目根目录为止，不再限制深度
    try:
        items = sorted(search_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    except PermissionError:
        return candidates

    sibling_exp_dirs = [
        p for p in items
        if p.is_dir()
        and not p.name.startswith('.')
        and p.name not in _CHECKPOINT_SCAN_SKIP_SUBDIRS
    ]

    # project 指向「单次 run 根」且仅有 weights/ 等子目录时，用当前目录作为 run（正确关联 args.yaml / results.csv）
    root_hit = first_checkpoints(search_path)
    if root_hit and not sibling_exp_dirs:
        checkpoint_path, checkpoint_label = root_hit[0]
        if not _run_dir_training_fully_complete(search_path):
            append_candidate(search_path, checkpoint_path, checkpoint_label)
        return candidates

    for run_dir in items:
        if not run_dir.is_dir():
            continue
        # Ultralytics 将权重放在 run 下的 weights/ 中；勿把该子目录当作独立「运行名」，
        # 否则会出现名为 weights 的候选项，且 args.yaml/results.csv 在父级导致无法识别「已跑满」。
        if run_dir.name.startswith('.') or run_dir.name in _CHECKPOINT_SCAN_SKIP_SUBDIRS:
            continue
        checkpoint_files = first_checkpoints(run_dir)
        if checkpoint_files:
            checkpoint_path, checkpoint_label = checkpoint_files[0]
            # 已跑满配置轮数的 run 不再作为断点候选项（与 Ultralytics「已完成、无可 resume」一致）
            if _run_dir_training_fully_complete(run_dir):
                sub_candidates = _scan_run_dirs_for_checkpoints(run_dir, logical_project)
                candidates.extend(sub_candidates)
                continue
            append_candidate(run_dir, checkpoint_path, checkpoint_label)
        else:
            # 无深度限制地递归搜索子目录，直到遍历完所有层级
            sub_candidates = _scan_run_dirs_for_checkpoints(run_dir, logical_project)
            candidates.extend(sub_candidates)
    return candidates

def _resolve_model_path(run_dir: Path):
    args_file = run_dir / 'args.yaml'
    if not args_file.exists():
        return None
    try:
        payload = _load_yaml_file(args_file)
    except Exception:
        return None
    model_value = (payload or {}).get('model')
    if not model_value:
        return None
    path = Path(model_value)
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    return path if path.exists() else None

def _estimate_model_params(model_path: Path):
    try:
        return estimate_params_m_from_checkpoint(model_path)
    except Exception:
        return None

def _estimate_model_gflops(model_path: Path):
    try:
        return estimate_gflops_from_weight(model_path)
    except Exception:
        pass
    return None

def _estimate_run_stats(run_dir: Path):
    model_path = _resolve_model_path(run_dir)
    if model_path is None:
        return {'ov-fps': '--', 'ov-params': '--'}

    try:
        if EdgeProfiler is not None:
            profiler = EdgeProfiler(str(model_path), target_device='gpu')
            result = profiler.run_full_profile()
            return {
                'ov-fps': f"{result.theoretical_fps_fp16:.0f}",
                'ov-params': f"{result.params_m:.1f}"
            }
    except Exception:
        pass

    params_m = _estimate_model_params(model_path)
    fps_str = '--'

    gflops = _estimate_model_gflops(model_path)
    if gflops is None and params_m is not None:
        gflops = params_m * 4.5
    if gflops is not None and gflops > 0:
        gpu_tops_fp16 = 13.0
        efficiency = 0.6
        fps = (gpu_tops_fp16 * 1e12 * efficiency) / (gflops * 1e9)
        fps_str = f"{fps:.0f}"

    return {
        'ov-fps': fps_str if fps_str != '--' else 'N/A',
        'ov-params': f"{params_m:.1f}" if params_m is not None else '--'
    }

def _load_csv_summary(path: Path):
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
        return list(reader.fieldnames or []), rows
    except Exception:
        return [], []

def _run_dir_training_fully_complete(run_dir: Path) -> bool:
    """
    判断该 run 是否已按 args.yaml 中的 epochs 跑满（最后一行 epoch 达到终态）。
    用于从断点续训列表中排除「已正常结束」的 checkpoint。
    """
    args_path = run_dir / 'args.yaml'
    csv_path = run_dir / 'results.csv'
    if not args_path.exists() or not csv_path.exists():
        return False
    cfg = _load_yaml_file(args_path)
    if not isinstance(cfg, dict):
        return False
    try:
        total_epochs = int(cfg.get('epochs'))
    except Exception:
        return False
    if total_epochs <= 0:
        return False
    _, rows = _load_csv_summary(csv_path)
    if not rows:
        return False
    le = _as_float(rows[-1].get('epoch'))
    if le is None:
        return False
    te = float(total_epochs)
    # 兼容 0-based（末轮 epochs-1）、1-based（末轮等于 epochs）及 CSV 中末行已为满轮整数的情况
    if le + 1e-6 >= te:
        return True
    return bool(le + 1e-6 >= te - 1.0)

def _summarize_series(rows, key, better='higher'):
    values = [_as_float(row.get(key)) for row in rows if row.get(key) is not None]
    values = [v for v in values if v is not None]
    if not values:
        return None
    final = values[-1]
    best = max(values) if better == 'higher' else min(values)
    diff = final - best
    sign = '+' if diff >= 0 else '-'
    if abs(diff) <= 1e-12:
        trend = 'stable'
    else:
        trend = 'up' if diff > 0 else 'down'
    improvement = f"{sign}{abs(diff) * 100:.2f}%"
    return {
        'best': best,
        'final': final,
        'improvement': improvement,
        'trend': trend
    }

def _extract_class_performance(rows, columns, class_labels=None, run_dir=None):
    if not columns or not rows:
        return None

    if run_dir is not None:
        cache_path = Path(run_dir) / 'per_class_metrics.json'
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                labels = cached.get('labels') or []
                map_values = cached.get('map') or []
                recall_values = cached.get('recall') or []
                precision_values = cached.get('precision') or []
                if labels and (map_values or recall_values or precision_values):
                    def normalize(arr):
                        arr = list(arr or [])
                        if len(arr) < len(labels):
                            arr.extend([None] * (len(labels) - len(arr)))
                        return arr[:len(labels)]
                    return {
                        'labels': labels,
                        'map': normalize(map_values),
                        'precision': normalize(precision_values),
                        'recall': normalize(recall_values)
                    }
            except Exception:
                pass

    class_metrics = {}
    for column in columns:
        normalized = column.strip()
        match = re.search(r'(?i)(?:class|cls)[\s_/\\-]*(?P<index>\d+)', normalized)
        if not match:
            continue
        idx = int(match.group('index'))
        lower = normalized.lower()
        metric_key = None
        if 'precision' in lower:
            metric_key = 'precision'
        elif 'recall' in lower:
            metric_key = 'recall'
        elif 'map50-95' in lower or 'mAP50-95' in normalized:
            metric_key = 'map50'
        elif 'map50' in lower or 'ap50' in lower:
            metric_key = 'map50'
        elif 'ap' in lower:
            metric_key = 'map50'
        if not metric_key:
            continue
        class_metrics.setdefault(idx, {})[metric_key] = normalized

    if not class_metrics:
        return None

    final_row = rows[-1]
    labels = []
    map_values = []
    recall_values = []
    precision_values = []
    for idx in sorted(class_metrics.keys()):
        labels.append(class_labels[idx] if class_labels and idx < len(class_labels) else f'class{idx}')
        map_col = class_metrics[idx].get('map50')
        precision_col = class_metrics[idx].get('precision')
        recall_col = class_metrics[idx].get('recall')
        map_values.append(_as_float(final_row.get(map_col)) if map_col else None)
        precision_values.append(_as_float(final_row.get(precision_col)) if precision_col else None)
        recall_values.append(_as_float(final_row.get(recall_col)) if recall_col else None)

    return {
        'labels': labels,
        'map': map_values,
        'precision': precision_values,
        'recall': recall_values
    }

def _load_distill_log_json(run_dir):
    if not run_dir or not isinstance(run_dir, (str, Path)):
        return []
    try:
        log_path = Path(run_dir) / 'distill_log.json'
        if not log_path.exists():
            parent_log = Path(run_dir).parent / 'distill_log.json'
            if parent_log.exists():
                log_path = parent_log
            else:
                return []
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def _build_metric_series(rows, columns, run_dir):
    chart = {
        'epochs': [],
        'train_losses': {'box_loss': [], 'cls_loss': [], 'dfl_loss': []},
        'map_series': {'map50': [], 'map50_95': []},
        'lr_series': {'pg0': [], 'pg1': [], 'pg2': []},
        'precision_recall': {'precision': [], 'recall': []},
        'distill_series': {},
        'pr_curve': None,
        'class_performance': None
    }

    has_distill_columns = any(col.startswith('distill/') for col in (columns or []))
    distill_log_fallback = _load_distill_log_json(run_dir) if not has_distill_columns else []

    distill_by_epoch = {}
    for entry in distill_log_fallback:
        try:
            ep = int(entry.get('epoch', -1))
            if ep >= 0:
                distill_by_epoch[ep] = entry
        except (TypeError, ValueError):
            continue

    for row in rows:
        epoch = _as_float(row.get('epoch'))
        if epoch is None:
            continue
        epoch_int = int(epoch)
        chart['epochs'].append(epoch_int)
        chart['train_losses']['box_loss'].append(_as_float(row.get('train/box_loss')) or 0)
        chart['train_losses']['cls_loss'].append(_as_float(row.get('train/cls_loss')) or 0)
        chart['train_losses']['dfl_loss'].append(_as_float(row.get('train/dfl_loss')) or 0)
        chart['map_series']['map50'].append(_as_float(row.get('metrics/mAP50(B)')) or 0)
        chart['map_series']['map50_95'].append(_as_float(row.get('metrics/mAP50-95(B)')) or 0)
        chart['lr_series']['pg0'].append(_as_float(row.get('lr/pg0')) or 0)
        chart['lr_series']['pg1'].append(_as_float(row.get('lr/pg1')) or 0)
        chart['lr_series']['pg2'].append(_as_float(row.get('lr/pg2')) or 0)
        chart['precision_recall']['precision'].append(_as_float(row.get('metrics/precision(B)')) or 0)
        chart['precision_recall']['recall'].append(_as_float(row.get('metrics/recall(B)')) or 0)

        alpha_raw = row.get('distill/alpha')
        temp_raw = row.get('distill/temperature')
        kd_raw = row.get('distill/kd_loss')
        has_distill_values = any(v not in (None, '') for v in (alpha_raw, temp_raw, kd_raw))
        if has_distill_columns and has_distill_values:
            alpha_val = _as_float(alpha_raw)
            temp_val = _as_float(temp_raw)
            kd_val = _as_float(kd_raw)
            chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [alpha_val if alpha_val is not None else None]
            chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [temp_val if temp_val is not None else None]
            chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [kd_val if kd_val is not None else None]
        elif distill_by_epoch:
            de = distill_by_epoch.get(epoch_int) or distill_by_epoch.get(epoch_int - 1)
            if de:
                alpha_val = de.get('alpha')
                temp_val = de.get('temperature')
                kd_val = de.get('kd_loss') or de.get('avg_kd_loss')
                chart['distill_series']['alpha'] = chart['distill_series'].get('alpha', []) + [_as_float(alpha_val)]
                chart['distill_series']['temperature'] = chart['distill_series'].get('temperature', []) + [_as_float(temp_val)]
                chart['distill_series']['kd_loss'] = chart['distill_series'].get('kd_loss', []) + [_as_float(kd_val)]
            else:
                for k in ('alpha', 'temperature', 'kd_loss'):
                    chart['distill_series'][k] = chart['distill_series'].get(k, []) + [None]

    if chart['precision_recall']['precision'] and chart['precision_recall']['recall']:
        chart['pr_curve'] = {
            'precision': chart['precision_recall']['precision'],
            'recall': chart['precision_recall']['recall']
        }
    else:
        chart['pr_curve'] = None
    chart['class_performance'] = _extract_class_performance(rows, columns, None, run_dir)
    return chart

def _error(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={'error': message})
