from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import yaml
from ultralytics import YOLO


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    for encoding in ('utf-8', 'utf-8-sig', 'cp936', 'gbk', 'latin1'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                data = yaml.safe_load(f)
                return data or {}
        except UnicodeDecodeError:
            continue
        except Exception:
            return {}
    return {}


def _get_nested(cfg: dict, keys: list[str], default=None):
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is default:
            return default
    return node


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _sanitize_kwargs(model, candidate_kwargs: dict[str, object]) -> dict[str, object]:
    supported = set(inspect.signature(model).parameters)
    return {k: v for k, v in candidate_kwargs.items() if k in supported and v is not None}


def _print_line(message: str) -> None:
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def _get_box_count(result) -> int:
    try:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return 0
        return len(boxes)
    except Exception:
        return 0


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        _print_line(f"ERROR: 无法读取输入参数: {exc}")
        return 2

    config_path = _resolve_path(payload.get('config') or 'configs/distill_config.yaml')
    cfg = _load_yaml(config_path) if config_path else {}

    source = payload.get('source') or _get_nested(cfg, ['advanced', 'training', 'source']) or _get_nested(cfg, ['training', 'data_yaml'])
    if not source:
        _print_line('ERROR: 未指定推理数据源 source')
        return 2
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (BASE_DIR / source_path).resolve()

    student_weight = payload.get('weight') or _get_nested(cfg, ['distillation', 'student_weight'])
    if not student_weight:
        _print_line('ERROR: 未指定学生模型权重 weight')
        return 2
    weight_path = _resolve_path(student_weight)
    if not weight_path or not weight_path.exists():
        _print_line(f'ERROR: 模型权重不存在: {student_weight}')
        return 2

    device = payload.get('device') or _get_nested(cfg, ['training', 'device']) or 'cpu'
    imgsz = payload.get('imgsz') or _get_nested(cfg, ['training', 'imgsz']) or 640
    conf = payload.get('conf') or _get_nested(cfg, ['training', 'conf']) or 0.25
    iou = payload.get('iou') or _get_nested(cfg, ['training', 'iou']) or 0.45

    output_dir = payload.get('output_dir') or _get_nested(cfg, ['advanced', 'training', 'output_dir'])
    if not output_dir:
        output_dir = 'runs/inference_results'
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_txt = _coerce_bool(payload.get('save_txt') or _get_nested(cfg, ['advanced', 'training', 'save_txt']))
    save_conf = _coerce_bool(payload.get('save_conf') or _get_nested(cfg, ['advanced', 'training', 'save_conf']))
    save_crop = _coerce_bool(payload.get('save_crop') or _get_nested(cfg, ['advanced', 'training', 'save_crop']))
    show_labels = _coerce_bool(payload.get('show_labels') or _get_nested(cfg, ['advanced', 'training', 'show_labels']))
    show_conf = _coerce_bool(payload.get('show_conf') or _get_nested(cfg, ['advanced', 'training', 'show_conf']))
    show_boxes = _coerce_bool(payload.get('show_boxes') or _get_nested(cfg, ['advanced', 'training', 'show_boxes']))
    line_width = payload.get('line_width') or _get_nested(cfg, ['advanced', 'training', 'line_width']) or 2
    visualize = _coerce_bool(payload.get('visualize') or _get_nested(cfg, ['advanced', 'training', 'visualize']))
    show = _coerce_bool(payload.get('show') or _get_nested(cfg, ['advanced', 'training', 'show']))

    save_flag = visualize or save_txt or save_conf or save_crop
    if not save_flag:
        save_flag = False

    model = YOLO(str(weight_path))
    _print_line(f'INFO: 使用权重 {weight_path}')
    _print_line(f'INFO: 推理源 {source_path}')
    _print_line(f'INFO: 输出目录 {output_path}')

    predict_kwargs = {
        'imgsz': imgsz,
        'device': device,
        'conf': conf,
        'iou': iou,
        'save': save_flag,
        'save_txt': save_txt,
        'save_conf': save_conf,
        'save_crop': save_crop,
        'save_dir': str(output_path),
        'show': show,
        'visualize': visualize,
        'show_labels': show_labels,
        'show_conf': show_conf,
        'show_boxes': show_boxes,
        'line_width': line_width,
        'verbose': True,
    }
    predict_kwargs = _sanitize_kwargs(model.predict, predict_kwargs)

    results = model.predict(str(source_path), **predict_kwargs)
    total_images = len(results)
    total_boxes = sum(_get_box_count(res) for res in results)

    _print_line(f'INFO: 推理完成，总样本数={total_images}, 总检测框={total_boxes}')
    _print_line(f'INFO: 结果输出目录 {output_path}')

    for result in results:
        path = getattr(result, 'path', None) or getattr(result, 'orig_img', None)
        caption = str(path)
        boxes = _get_box_count(result)
        _print_line(f'INFO: 结果文件 {caption}  检测框={boxes}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
