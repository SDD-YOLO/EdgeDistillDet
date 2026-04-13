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

import yaml
from ultralytics import YOLO

from core.distillation.adaptive_kd_trainer import AdaptiveKDTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EdgeDistillDet.TrainScript")


def run_distill_training(config_path: str):
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

    # 将蒸馏参数注入 overrides，由 Trainer 读取
    kd_overrides = {
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

    # 合并 train 参数
    train_args = {
        "data":         train_cfg.get("data_yaml", "data.yaml"),
        "project":      output_cfg.get("project", "runs/distill"),
        "name":         output_cfg.get("name", "adaptive_kd"),
        "device":       train_cfg.get("device", 0),
        "epochs":       train_cfg.get("epochs", 150),
        "imgsz":        train_cfg.get("imgsz", 640),
        "batch":        train_cfg.get("batch", -1),
        "workers":      train_cfg.get("workers", 8),
        "lr0":          train_cfg.get("lr0", 0.01),
        "lrf":          train_cfg.get("lrf", 0.1),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3.0),
        "mosaic":       train_cfg.get("mosaic", 0.8),
        "mixup":        train_cfg.get("mixup", 0.1),
        "close_mosaic": train_cfg.get("close_mosaic", 20),
        "amp":          train_cfg.get("amp", True),
        **kd_overrides,
    }

    student_model.TrainerClass = AdaptiveKDTrainer

    logger.info("=" * 55)
    logger.info("  自适应知识蒸馏训练启动")
    logger.info(f"  alpha_init={kd_overrides['kd_alpha_init']} | "
                f"T: [{kd_overrides['T_max']}→{kd_overrides['T_min']}] | "
                f"warm_epochs={kd_overrides['warm_epochs']}")
    logger.info("=" * 55)

    results = student_model.train(**train_args)

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
