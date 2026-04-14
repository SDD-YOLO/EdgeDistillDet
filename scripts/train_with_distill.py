"""
scripts/train_with_distill.py
==============================
自适应知识蒸馏训练脚本（入口）

读取 distill_config.yaml → 构建 AdaptiveKDTrainer → 启动训练 → 保存蒸馏日志
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

from core.distillation.adaptive_kd_trainer import AdaptiveKDTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EdgeDistillDet.TrainScript")


def find_resume_checkpoint(project: str, name: str) -> Optional[Path]:
    run_dir = Path(project) / name
    candidates = [
        run_dir / 'last.pt',
        run_dir / 'weights' / 'last.pt',
        run_dir / 'weights' / 'best.pt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if run_dir.exists():
        pt_files = sorted(run_dir.rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if pt_files:
            return pt_files[0]

    normalized = str(run_dir).replace('\\', '/').lstrip('./')
    root = Path.cwd()
    for pattern in ['**/last.pt', '**/weights/last.pt', '**/weights/best.pt']:
        for candidate in sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate_rel = str(candidate.relative_to(root)).replace('\\', '/')
            if normalized in candidate_rel:
                return candidate

    return None


def run_distill_training(config_path: str, resume: Optional[str] = None):
    """从配置文件启动自适应蒸馏训练。"""

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    distill_cfg = cfg.get("distillation", {})
    train_cfg   = cfg.get("training", {})
    output_cfg  = cfg.get("output", {})

    student_path = distill_cfg.get("student_weight", "")
    teacher_path = distill_cfg.get("teacher_weight", "")

    if not os.path.exists(student_path):
        raise FileNotFoundError(f"学生模型权重不存在: {student_path}")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"教师模型权重不存在: {teacher_path}")

    logger.info(f"学生模型: {student_path}")
    logger.info(f"教师模型: {teacher_path}")

    student_model = YOLO(student_path)

    # 蒸馏参数（存储在模型对象上，不传入 train_args 避免新版ultralytics校验报错）
    kd_params = {
        "teacher_path":   teacher_path,
        "kd_alpha_init":  distill_cfg.get("alpha_init", 0.5),
        "T_max":          distill_cfg.get("T_max", 6.0),
        "T_min":          distill_cfg.get("T_min", 1.5),
        "warm_epochs":    distill_cfg.get("warm_epochs", 5),
        "w_kd":           distill_cfg.get("w_kd", 0.5),
        "w_focal":        distill_cfg.get("w_focal", 0.3),
        "w_feat":         distill_cfg.get("w_feat", 0.0),
        "scale_boost":    distill_cfg.get("scale_boost", 2.0),
        "focal_gamma":    distill_cfg.get("focal_gamma", 2.0),
    }
    # 挂载到模型对象上，AdaptiveKDTrainer 从这里读取
    student_model._kd_params = kd_params

    # 仅包含标准 YOLO 训练参数
    train_args = {
        "data":          train_cfg.get("data_yaml", "data.yaml"),
        "project":       output_cfg.get("project", "runs/distill"),
        "name":          output_cfg.get("name", "adaptive_kd"),
        "device":        train_cfg.get("device", 0),
        "epochs":        train_cfg.get("epochs", 150),
        "imgsz":         train_cfg.get("imgsz", 640),
        "batch":         train_cfg.get("batch", -1),
        "workers":       train_cfg.get("workers", 8),
        "lr0":           train_cfg.get("lr0", 0.01),
        "lrf":           train_cfg.get("lrf", 0.1),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
        "mosaic":        train_cfg.get("mosaic", 0.8),
        "mixup":         train_cfg.get("mixup", 0.1),
        "close_mosaic":  train_cfg.get("close_mosaic", 20),
        "amp":           train_cfg.get("amp", True),
    }

    if resume:
        if resume == 'auto':
            resume_path = find_resume_checkpoint(train_args['project'], train_args['name'])
            if resume_path is None:
                logger.warning("未找到可续训 checkpoint，改为从头训练。")
            else:
                train_args['resume'] = str(resume_path)
                logger.info(f"断点续训: 从 {resume_path} 恢复训练")
        else:
            resume_path = Path(resume)
            if resume_path.exists():
                train_args['resume'] = str(resume_path)
                logger.info(f"断点续训: 从指定 checkpoint {resume_path} 恢复训练")
            else:
                raise FileNotFoundError(f"指定的断点权重不存在: {resume}")

    student_model.TrainerClass = AdaptiveKDTrainer

    logger.info("=" * 55)
    logger.info("  自适应知识蒸馏训练启动")
    logger.info(f"  alpha_init={kd_params['kd_alpha_init']} | "
                f"T: [{kd_params['T_max']}→{kd_params['T_min']}] | "
                f"warm_epochs={kd_params['warm_epochs']}")
    logger.info("=" * 55)

    try:
        results = student_model.train(**train_args)
    except Exception as e:
        msg = str(e)
        if 'nothing to resume' in msg.lower() or 'finished, nothing to resume' in msg.lower():
            logger.warning("检测到 checkpoint 已完成训练，改为不使用 resume 参数重新启动训练")
            train_args.pop('resume', None)
            results = student_model.train(**train_args)
        else:
            raise

    # 保存蒸馏过程日志
    trainer: AdaptiveKDTrainer = student_model.trainer
    if hasattr(trainer, "get_distill_log"):
        log_data = trainer.get_distill_log()
        log_path = Path(output_cfg.get("project", "runs/distill")) / \
                   output_cfg.get("name", "adaptive_kd") / "distill_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        logger.info(f"蒸馏过程日志已保存: {log_path}")

    student_model.fuse()
    logger.info("✅ 自适应蒸馏训练完成，模型已 fuse。")
    return results
