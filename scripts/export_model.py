from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


SUPPORTED_EXPORT_FORMATS = {
    "onnx",
    "torchscript",
    "tflite",
    "saved_model",
    "coreml",
}
EXPECTED_OUTPUT_SUFFIX = {
    "onnx": ".onnx",
    "torchscript": ".torchscript",
    "tflite": ".tflite",
    "coreml": ".mlmodel",
}


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
    for encoding in ("utf-8", "utf-8-sig", "cp936", "gbk", "latin1"):
        try:
            with open(path, encoding=encoding) as f:
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


def _get_config_value(cfg: dict, paths: list[list[str]], default=None):
    for path in paths:
        value = _get_nested(cfg, path)
        if value is not None and (not isinstance(value, str) or value.strip() != ""):
            return value
    return default


def _sanitize_kwargs(model, candidate_kwargs: dict[str, object]) -> dict[str, object]:
    try:
        sig = inspect.signature(model)
        params = sig.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return {k: v for k, v in candidate_kwargs.items() if v is not None}
        supported = set(params)
        return {k: v for k, v in candidate_kwargs.items() if k in supported and v is not None}
    except Exception:
        return {k: v for k, v in candidate_kwargs.items() if v is not None}


def _supports_param(model, name: str) -> bool:
    try:
        sig = inspect.signature(model)
        params = sig.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            # 即使有 **kwargs，也要检查是否是已知参数
            known_yolo_export_args = {
                "format",
                "imgsz",
                "half",
                "int8",
                "dynamic",
                "simplify",
                "opset",
                "workspace",
                "nms",
                "batch",
                "device",
                "keras",
                "optimize",
                "verbose",
                "project",
                "name",
                "save_dir",
            }
            return name in known_yolo_export_args
        return name in params
    except Exception:
        return False


def _print_line(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON config string")
    parser.add_argument("--config-file", type=str, help="JSON config file path")
    args = parser.parse_args()

    if args.config_file:
        try:
            with open(args.config_file, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            _print_line(f"ERROR: 无法读取配置文件: {exc}")
            return 2
    elif args.config:
        try:
            payload = json.loads(args.config)
        except Exception as exc:
            _print_line(f"ERROR: 无效的 JSON 参数: {exc}")
            return 2
    else:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            _print_line(f"ERROR: 无法读取输入参数: {exc}")
            return 2

    config_path = _resolve_path(payload.get("config") or "configs/distill_config.yaml")
    cfg = _load_yaml(config_path) if config_path else {}

    student_weight = payload.get("weight") or _get_nested(cfg, ["distillation", "student_weight"])
    if not student_weight:
        _print_line("ERROR: 未指定学生模型权重 weight")
        return 2

    weight_path = _resolve_path(student_weight)
    if not weight_path or not weight_path.exists():
        _print_line(f"ERROR: 模型权重不存在: {student_weight}")
        return 2

    export_path = payload.get("export_path")
    if isinstance(export_path, str):
        export_path = export_path.strip() or None
    if not export_path:
        export_path = _get_config_value(
            cfg,
            [
                ["export_model", "export_path"],
                ["advanced", "training", "export_path"],
                ["training", "export_path"],
            ],
        )
        if isinstance(export_path, str):
            export_path = export_path.strip() or None
    if not export_path:
        export_path = "runs/exported_models"

    export_target = _resolve_path(export_path)
    if export_target is None:
        _print_line(f"ERROR: 无效的导出路径: {export_path}")
        return 2

    if export_target.suffix:
        output_dir = export_target.parent
        final_file = export_target
    else:
        output_dir = export_target
        final_file = None
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_line(f"INFO: 使用权重 {weight_path}")
    _print_line(f"INFO: 导出目录 {output_dir}")
    if final_file is not None:
        _print_line(f"INFO: 目标文件 {final_file}")

    fmt = str(
        payload.get("format")
        or _get_config_value(
            cfg,
            [
                ["export_model", "format"],
                ["advanced", "training", "format"],
                ["training", "format"],
            ],
        )
        or "onnx"
    ).lower()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        _print_line(f"ERROR: 不支持的导出格式: {fmt}")
        return 2

    keras = bool(
        payload.get("keras")
        or _get_config_value(
            cfg,
            [
                ["export_model", "keras"],
                ["advanced", "training", "keras"],
                ["training", "keras"],
            ],
        )
    )
    optimize = bool(
        payload.get("optimize")
        or _get_config_value(
            cfg,
            [
                ["export_model", "optimize"],
                ["advanced", "training", "optimize"],
                ["training", "optimize"],
            ],
        )
    )
    int8 = bool(
        payload.get("int8")
        or _get_config_value(
            cfg,
            [
                ["export_model", "int8"],
                ["advanced", "training", "int8"],
                ["training", "int8"],
            ],
        )
    )
    dynamic = bool(
        payload.get("dynamic")
        or _get_config_value(
            cfg,
            [
                ["export_model", "dynamic"],
                ["advanced", "training", "dynamic"],
                ["training", "dynamic"],
            ],
        )
    )
    simplify = bool(
        payload.get("simplify")
        or _get_config_value(
            cfg,
            [
                ["export_model", "simplify"],
                ["advanced", "training", "simplify"],
                ["training", "simplify"],
            ],
        )
    )
    opset = payload.get("opset") or _get_config_value(
        cfg,
        [
            ["export_model", "opset"],
            ["advanced", "training", "opset"],
            ["training", "opset"],
        ],
    )
    workspace = payload.get("workspace") or _get_config_value(
        cfg,
        [
            ["export_model", "workspace"],
            ["advanced", "training", "workspace"],
            ["training", "workspace"],
        ],
    )
    nms = bool(
        payload.get("nms")
        or _get_config_value(
            cfg,
            [
                ["export_model", "nms"],
                ["advanced", "training", "nms"],
                ["training", "nms"],
            ],
        )
    )
    save_dir = payload.get("save_dir") or _get_config_value(
        cfg,
        [
            ["export_model", "save_dir"],
            ["advanced", "training", "save_dir"],
            ["training", "save_dir"],
        ],
    )

    model = YOLO(str(weight_path))

    export_kwargs = {
        "format": fmt,
        "keras": keras,
        "optimize": optimize,
        "int8": int8,
        "dynamic": dynamic,
        "simplify": simplify,
        "opset": opset,
        "workspace": workspace,
        "nms": nms,
        "verbose": True,
    }
    if _supports_param(model.export, "project"):
        export_kwargs["project"] = str(output_dir)
    if _supports_param(model.export, "save_dir"):
        export_kwargs["save_dir"] = str(_resolve_path(save_dir) or output_dir) if save_dir else str(output_dir)

    # 修正：只用 name，不用 save_name（Ultralytics 不认识 save_name）
    if final_file is not None:
        if _supports_param(model.export, "name"):
            export_kwargs["name"] = final_file.stem
    elif _supports_param(model.export, "name"):
        export_kwargs["name"] = weight_path.stem

    if weight_path.suffix == ".torchscript" and fmt != "torchscript":
        _print_line("ERROR: 当前权重文件为 TorchScript 模型，不能导出为非 TorchScript 格式。请使用 torchscript 格式，或换用原始 PyTorch 权重 (.pt/.pth)。")
        return 2

    export_kwargs = _sanitize_kwargs(model.export, export_kwargs)

    # 强制移除已知无效参数（防止 _sanitize_kwargs 误判）
    invalid_yolo_export_args = {"save_name"}
    for key in invalid_yolo_export_args:
        export_kwargs.pop(key, None)

    exported = model.export(**export_kwargs)
    exported_paths = []
    if isinstance(exported, list | tuple):
        exported_paths = [str(p) for p in exported if p]
    elif exported:
        exported_paths = [str(exported)]

    expected_suffix = EXPECTED_OUTPUT_SUFFIX.get(fmt)
    if expected_suffix and exported_paths:
        mismatched = [p for p in exported_paths if not Path(p).name.lower().endswith(expected_suffix)]
        if mismatched:
            _print_line(f"ERROR: 导出格式为 {fmt}，但生成文件未匹配预期后缀 {expected_suffix}，当前输出: {exported_paths}")
            return 2

    if final_file and len(exported_paths) == 1:
        try:
            Path(exported_paths[0]).replace(final_file)
            exported_paths = [str(final_file)]
            _print_line(f"INFO: 已移动导出文件到 {final_file}")
        except Exception as exc:
            _print_line(f"WARNING: 无法移动导出文件: {exc}")
    elif output_dir is not None and exported_paths:
        moved_paths = []
        for exported_path in exported_paths:
            exported_file = Path(exported_path)
            if exported_file.parent.resolve() != output_dir.resolve():
                destination = output_dir / exported_file.name
                try:
                    exported_file.replace(destination)
                    moved_paths.append(str(destination))
                    _print_line(f"INFO: 已移动导出文件到 {destination}")
                except Exception as exc:
                    _print_line(f"WARNING: 无法移动导出文件: {exc}")
                    moved_paths.append(str(exported_file))
            else:
                moved_paths.append(str(exported_file))
        exported_paths = moved_paths

    if exported_paths:
        for path in exported_paths:
            _print_line(f"INFO: 导出完成: {path}")
        return 0

    _print_line("ERROR: 导出失败，未生成任何文件")
    return 2


if __name__ == "__main__":
    sys.exit(main())
