"""
EdgeDistillDet — 蒸馏训练唯一入口（CLI + Web 共用）

设计目标
--------
1. 单进程、单训练栈：禁止在此处再 spawn 第二套训练。
2. 断点续训与「新训」共用同一套 ultralytics + AdaptiveKDTrainer 逻辑。
3. 从 checkpoint 所在 run 目录读取 args.yaml，强制对齐 project / name / data，
   避免前端表单与磁盘上的真实 run 不一致导致写错目录或重复加载。
4. DataLoader workers 固定为 0（Windows / 小内存 / Web 场景），确保训练过程稳定可控。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


import yaml
from ultralytics import YOLO

from core.distillation.adaptive_kd_trainer import AdaptiveKDTrainer, _scatter_pr_from_ultralytics_box
from core.distillation.common import w_feat_to_scalar
from utils import expand_env_vars
from utils.device_detect import detect_best_device, setup_device_for_trainer
from utils.gpu_runtime import cleanup_gpu_resources

# 训练期 DataLoader 始终 0 workers，避免多进程复制数据集与「像两个训练同时跑」的内存形态
_TRAIN_LOADER_WORKERS = 0
_TRAINING_STRUCTURAL_KEYS = {"compute_provider", "cloud_api", "dataset_api", "data_yaml"}


def _flush_stdio():
    """管道/无 TTY 场景下尽量行缓冲，便于 Web 端实时读日志。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass


def _apply_web_friendly_tqdm_output() -> None:
    """
    Web 子进程 stdout 为管道时，ultralytics 自带 TQDM 默认用 \\r + CSI K 覆写同一行，
    readline() 几乎收不到「每步 loss 行」。将 \\r 与 ANSI 清屏转为换行并去掉 ESC，便于管道按行展示。
    """
    if os.environ.get("EDGE_WEB_LOG") != "1":
        return
    try:
        from ultralytics.utils.tqdm import TQDM
    except Exception:
        return
    if getattr(TQDM, "_edge_web_logged", False):
        return

    _orig_display = TQDM._display
    _orig_close = TQDM.close
    _orig_clear = TQDM.clear

    def _normalize(s: str) -> str:
        if not s:
            return s
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)

    class _CrNlWrapper:
        __slots__ = ("_real",)

        def __init__(self, real):
            self._real = real

        def write(self, s: str):
            return self._real.write(_normalize(s))

        def flush(self):
            return self._real.flush()

    def _display_patched(self, final=False):
        rf = self.file
        self.file = _CrNlWrapper(rf)
        try:
            return _orig_display(self, final=final)
        finally:
            self.file = rf

    def _close_patched(self):
        rf = self.file
        self.file = _CrNlWrapper(rf)
        try:
            return _orig_close(self)
        finally:
            self.file = rf

    def _clear_patched(self):
        rf = self.file
        self.file = _CrNlWrapper(rf)
        try:
            return _orig_clear(self)
        finally:
            self.file = rf

    TQDM._display = _display_patched
    TQDM.close = _close_patched
    TQDM.clear = _clear_patched
    TQDM._edge_web_logged = True


def _print_banner(root: Path) -> None:
    """与 main.py 一致的 ASCII BANNER（Web 子进程不再经过 main.py 时仍打印）。"""
    root_s = str(root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    try:
        import main as edge_main

        print(getattr(edge_main, "BANNER", "").rstrip(), flush=True)
    except Exception:
        print("EdgeDistillDet — 蒸馏训练", flush=True)
    print("", flush=True)


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return expand_env_vars(raw) if isinstance(raw, dict) else {}


def _resolve_under_root(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cand = (root / p).resolve()
    if cand.exists():
        return cand
    # Common local mapping: prefer repo-local configs/dataset_coco128.yaml when
    # user provided the bare `coco128.yaml` name.
    if p.name == "coco128.yaml":
        alt = (root / "configs" / "dataset_coco128.yaml").resolve()
        if alt.exists():
            return alt
    return p


def _get_nested(cfg: Dict[str, Any], keys: list[str], default=None):
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is default:
            return default
    return node


def _merge_advanced_training(train_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    advanced_training = _get_nested(cfg, ["advanced", "training"]) or {}
    merged = dict(train_cfg or {})
    for key, value in advanced_training.items():
        if value is not None:
            merged[key] = value
    return merged


def _run_dir_from_checkpoint(ckpt: Path) -> Path:
    """.../weights/last.pt -> .../run_dir ; .../last.pt -> parent"""
    ckpt = ckpt.resolve()
    if ckpt.parent.name.lower() == "weights":
        return ckpt.parent.parent
    return ckpt.parent


def _sync_from_args_yaml(run_dir: Path, train_cfg: Dict[str, Any], output_cfg: Dict[str, Any]) -> None:
    """用磁盘上真实 run 的 args.yaml 覆盖表单里可能过期的 output / data。"""
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        return
    try:
        with open(args_yaml, "r", encoding="utf-8") as f:
            disk = yaml.safe_load(f) or {}
    except Exception:
        return
    if not isinstance(disk, dict):
        return
    if disk.get("project") is not None:
        output_cfg["project"] = str(disk["project"])
    if disk.get("name") is not None:
        output_cfg["name"] = str(disk["name"])
    if disk.get("data") is not None:
        train_cfg["data_yaml"] = str(disk["data"])


def _find_auto_resume(project: str, name: str, root: Path) -> Optional[Path]:
    run_dir = (root / project / name).resolve()
    for rel in ("weights/last.pt", "last.pt", "weights/best.pt", "best.pt"):
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _pre_cuda_gc():
    cleanup_gpu_resources()

def _build_train_args(
    train_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    resume_path: Optional[Path],
    root: Path,
    allow_overwrite: bool = False,
) -> Dict[str, Any]:
    data_yaml = str(_resolve_under_root(train_cfg.get("data_yaml", "coco128.yaml"), root))

    # 【排查】先用最小参数集，确认数据加载是否正常
    train_args = {
        "data": data_yaml,
        "epochs": int(train_cfg.get("epochs", 10)),
        "imgsz": int(train_cfg.get("imgsz", 640)),
        "batch": int(train_cfg.get("batch", 16)),
        "workers": _TRAIN_LOADER_WORKERS,
        "project": output_cfg.get("project", "runs"),
        "name": output_cfg.get("name", "exp"),
        "exist_ok": bool(allow_overwrite),
        "verbose": True,
    }
    
    # 如果最小参数集正常，再逐个加回以下参数测试
    # optional = {
    #     "device": train_cfg.get("device", 0),
    #     "lr0": float(train_cfg.get("lr0", 0.01)),
    #     "lrf": float(train_cfg.get("lrf", 0.1)),
    #     "warmup_epochs": int(train_cfg.get("warmup_epochs", 3)),
    #     "mosaic": float(train_cfg.get("mosaic", 0.8)),      # ← 最可疑
    #     "mixup": float(train_cfg.get("mixup", 0.1)),        # ← 次可疑
    #     "close_mosaic": int(train_cfg.get("close_mosaic", 1)), # ← 次可疑
    #     "amp": bool(train_cfg.get("amp", True)),
    #     "plots": False,
    # }
    
    return train_args

def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return default


def _setup_wandb_env(
    wandb_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    resume_path: Optional[Path],
) -> None:
    """
    配置 W&B 环境变量。

    说明：
    - 默认关闭（兼容旧行为）；
    - 支持在 YAML 里显式开启并自定义 project/entity/name/tags。
    """
    enabled = _to_bool(wandb_cfg.get("enabled", False), default=False)
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return

    mode = str(wandb_cfg.get("mode", "online")).strip().lower() or "online"
    if mode not in {"online", "offline", "disabled"}:
        mode = "online"
    os.environ["WANDB_MODE"] = mode

    project = str(wandb_cfg.get("project") or output_cfg.get("project") or "edge-distilldet").strip()
    if project:
        os.environ["WANDB_PROJECT"] = project

    entity = str(wandb_cfg.get("entity", "")).strip()
    if entity:
        os.environ["WANDB_ENTITY"] = entity

    run_name = str(wandb_cfg.get("name") or output_cfg.get("name") or "").strip()
    if run_name:
        suffix = "_resume" if resume_path is not None else ""
        os.environ["WANDB_NAME"] = f"{run_name}{suffix}"

    group = str(wandb_cfg.get("group", "")).strip()
    if group:
        os.environ["WANDB_RUN_GROUP"] = group

    job_type = str(wandb_cfg.get("job_type", "")).strip()
    if job_type:
        os.environ["WANDB_JOB_TYPE"] = job_type

    notes = str(wandb_cfg.get("notes", "")).strip()
    if notes:
        os.environ["WANDB_NOTES"] = notes

    tags = wandb_cfg.get("tags")
    if isinstance(tags, (list, tuple)):
        clean_tags = [str(t).strip() for t in tags if str(t).strip()]
        if clean_tags:
            os.environ["WANDB_TAGS"] = ",".join(clean_tags)


def run_distill_training(config_path: str | Path, resume: str = "", allow_overwrite: bool = False) -> Any:
    """
    执行蒸馏训练（支持 resume 为空、'auto'、或 last.pt 等绝对/相对路径）。

    resume 语义
    -----------
    - ""     : 全新训练
    - "auto" : 在当前 output.project / output.name 下自动找 last.pt
    - 其他   : 指定 checkpoint；会先对齐该 run 的 args.yaml 再开训
    """
    config_path = Path(config_path)
    root = _project_root(config_path)
    _flush_stdio()
    _print_banner(root)
    _apply_web_friendly_tqdm_output()

    cfg = _load_config(config_path)

    distill_cfg = cfg.get("distillation", {}) or {}
    train_cfg = _merge_advanced_training(dict(cfg.get("training", {}) or {}), cfg)
    output_cfg = dict(cfg.get("output", {}) or {})
    wandb_cfg = dict(cfg.get("wandb", {}) or {})

    # ========== 自动设备检测与修复（必须在 train_cfg 创建之后）==========
    train_cfg = setup_device_for_trainer(train_cfg)
    # ================================================================

    student_weight = str(_resolve_under_root(str(distill_cfg.get("student_weight", "yolov8n.pt")), root))
    if not Path(student_weight).exists():
        raise FileNotFoundError(f"学生权重不存在: {student_weight}")

    resume_path: Optional[Path] = None
    resume_flag = (resume or "").strip()

    if resume_flag:
        if resume_flag.lower() == "auto":
            resume_path = _find_auto_resume(
                str(output_cfg.get("project", "runs")),
                str(output_cfg.get("name", "exp")),
                root,
            )
        else:
            rp = Path(resume_flag)
            if not rp.is_absolute():
                rp = (root / rp).resolve()
            if not rp.exists():
                raise FileNotFoundError(f"断点文件不存在: {rp}")
            resume_path = rp

    if resume_path is not None:
        _sync_from_args_yaml(_run_dir_from_checkpoint(resume_path), train_cfg, output_cfg)
        _pre_cuda_gc()
        time.sleep(0.5)

    teacher_raw = (distill_cfg.get("teacher_weight") or "").strip()
    teacher_weight = str(_resolve_under_root(teacher_raw, root)) if teacher_raw else ""
    if teacher_weight and not Path(teacher_weight).exists():
        raise FileNotFoundError(f"教师权重不存在: {teacher_weight}")

    known_kd_keys = {"teacher_weight", "alpha_init", "T_max", "T_min", "warm_epochs", "w_kd", "w_focal", "w_feat", "scale_boost", "focal_gamma"}
    extra_kd_params = {k: v for k, v in distill_cfg.items() if k not in known_kd_keys and v is not None}
    AdaptiveKDTrainer.set_kd_params(
        teacher_path=teacher_weight,
        alpha_init=float(distill_cfg.get("alpha_init", 0.5)),
        T_max=float(distill_cfg.get("T_max", 6.0)),
        T_min=float(distill_cfg.get("T_min", 1.5)),
        warm_epochs=int(distill_cfg.get("warm_epochs", 5)),
        w_kd=float(distill_cfg.get("w_kd", 0.5)),
        w_focal=float(distill_cfg.get("w_focal", 0.3)),
        w_feat=w_feat_to_scalar(distill_cfg.get("w_feat", 0.0)),
        scale_boost=float(distill_cfg.get("scale_boost", 2.0)),
        focal_gamma=float(distill_cfg.get("focal_gamma", 2.0)),
        **extra_kd_params,
    )

    # 始终用「架构权重」构造 YOLO；真正的断点由 train(resume=...) 内部单次加载
    student_model = YOLO(student_weight)
    if resume_path is not None:
        try:
            student_model.ckpt_path = str(resume_path)
        except Exception:
            pass

    project_raw = output_cfg.get("project", "runs")
    project_abs = str((root / project_raw).resolve())
    output_cfg_resolved = {**output_cfg, "project": project_abs}
    train_args = _build_train_args(train_cfg, output_cfg_resolved, resume_path, root, allow_overwrite=allow_overwrite)
    os.environ.setdefault("ULTRALYTICS_VERBOSE", "True")
    os.environ.setdefault("DATAMODULE_WORKERS", "0")
    os.environ.setdefault("NUM_WORKERS", "0")
    _setup_wandb_env(wandb_cfg, output_cfg, resume_path)

    try:
        # 【关键修复】直接实例化 AdaptiveKDTrainer，不通过 model.train()
        # 旧代码：results = student_model.train(AdaptiveKDTrainer, **train_args)
        if "model" not in train_args:
            train_args["model"] = student_weight
        
        trainer = AdaptiveKDTrainer(overrides=train_args)
        results = trainer.train()
        student_model.trainer = trainer
 
    except Exception as e:
        msg = str(e).lower()
        if resume_path and ("nothing to resume" in msg or "finished, nothing to resume" in msg):
            print("[RESUME] 检测到 checkpoint 已训练完成，自动切换为新训练模式并重建学生模型。", flush=True)
            train_args.pop("resume", None)
            student_model = YOLO(student_weight)
            # resume 也使用直接实例化
            if "model" not in train_args:
                train_args["model"] = student_weight
            trainer = AdaptiveKDTrainer(overrides=train_args)
            results = trainer.train()
            student_model.trainer = trainer
        else:
            raise

    trainer = getattr(student_model, "trainer", None)
    if trainer is not None and hasattr(trainer, "get_distill_log"):
        log_data = trainer.get_distill_log()
        if log_data:
            run_dir = Path(getattr(trainer, "save_dir", "") or "")
            if not run_dir or not run_dir.exists():
                run_dir = Path(train_args["project"]) / train_args["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            distill_dir = run_dir / "distill"
            distill_dir.mkdir(parents=True, exist_ok=True)
            with open(distill_dir / "distill_log.json", "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

    _maybe_auto_eval(root, train_cfg, output_cfg, train_args)
    return results


def _maybe_auto_eval(
    root: Path,
    train_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    train_args: Dict[str, Any],
) -> None:
    if not output_cfg.get("auto_eval", True):
        return
    run_dir = (Path(train_args["project"]) / train_args["name"]).resolve()    
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    model_to_eval = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)
    if not model_to_eval:
        return
    try:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            m = YOLO(str(model_to_eval), verbose=False)
            eval_data_yaml = str(_resolve_under_root(train_cfg.get("data_yaml", "coco128.yaml"), root))
            eval_results = m.val(
                data=eval_data_yaml,
                imgsz=int(train_cfg.get("imgsz", 640)),
                batch=int(train_cfg.get("batch", 16)),
                verbose=False,
                project=str(run_dir),   # ← 指向 runs/distill/exp1/
                name="val",             # ← 子目录名
                exist_ok=True,
            )
        if hasattr(eval_results, "box") and eval_results.box is not None:
            class_names = getattr(eval_results, "names", {}) or {}
            box = eval_results.box
            nc = len(class_names) if class_names else getattr(box, "nc", 0)
            per_class_data = None
            if nc > 0:
                import numpy as _np

                _maps = None
                if hasattr(box, "maps") and box.maps is not None:
                    _m = _np.asarray(box.maps)
                    _maps = list(_m.flatten()) if _m.ndim > 1 else list(_m)
                labels_list, map_list = [], []
                for i in range(nc):
                    labels_list.append(class_names.get(i, f"class{i}"))
                    map_list.append(float(_maps[i]) if _maps and i < len(_maps) else None)
                prec_list, rec_list = _scatter_pr_from_ultralytics_box(box, nc)
                per_class_data = {
                    "labels": labels_list,
                    "map": map_list,
                    "precision": prec_list,
                    "recall": rec_list,
                    "source": "auto_eval",
                }
            eval_data = {
                "model": str(model_to_eval),
                "map50": float(getattr(eval_results.box, "map50", 0) or 0)
                if hasattr(eval_results, "box")
                and eval_results.box is not None
                else None,
                "map50_95": float(getattr(eval_results.box, "map", 0) or 0)
                if hasattr(eval_results, "box")
                and eval_results.box is not None
                else None,
            }
            if per_class_data:
                eval_data["per_class"] = per_class_data
                val_dir = run_dir / "val"
                val_dir.mkdir(parents=True, exist_ok=True)
                with open(val_dir / "per_class_metrics.json", "w", encoding="utf-8") as pf:
                    from datetime import datetime as _dt

                    per_class_data["generated_at"] = _dt.now().isoformat()
                    per_class_data["epoch"] = int(train_args.get("epochs", 0))
                    json.dump(per_class_data, pf, indent=2, ensure_ascii=False)
            val_dir = run_dir / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            with open(val_dir / "eval_result.json", "w", encoding="utf-8") as ef:
                json.dump(eval_data, ef, indent=2, ensure_ascii=False)
        del m
        _pre_cuda_gc()
    except Exception:
        pass


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="EdgeDistillDet 蒸馏训练（唯一推荐入口）")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="断点：留空=新训；auto=按 output 下自动找；否则为 last.pt 等路径",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="允许覆盖同名运行目录（会禁用 Ultralytics 自动递增 exp 名称）",
    )
    args = parser.parse_args(argv)
    try:
        run_distill_training(args.config, resume=args.resume, allow_overwrite=bool(args.allow_overwrite))
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
