"""
scripts/train_with_distill.py
==============================
自适应知识蒸馏训练脚本 — 完整重写版

改进：
  1. 整洁的结构化日志输出（无冗余 banner）
  2. 完整的训练流程：warm-up → 蒸馏 → 自动验证评估
  3. 训练完成后自动运行 benchmark 评估
  4. 蒸馏日志和结果统一保存
  5. 所有输出到 stderr，由 web 后端统一处理
"""

import os
import sys
import json
import logging
import time
import io
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr, contextmanager

import yaml
from ultralytics import YOLO

from core.distillation.adaptive_kd_trainer import AdaptiveKDTrainer
from utils import expand_env_vars


# ═══════════════════════════════════════════════════════════════════════════════
# 日志系统 — 统一输出到 stderr，格式整洁
# ═══════════════════════════════════════════════════════════════════════════════
class _CleanLogHandler(logging.Handler):
    """整洁格式化日志输出到 stderr"""
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S"
        ))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg, file=sys.stderr, flush=True)
        except Exception:
            pass


# 清除根 logger 的已有 handler，避免重复输出
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 创建主 logger — 简洁名称
logger = logging.getLogger("DistillTrain")
logger.setLevel(logging.INFO)
logger.addHandler(_CleanLogHandler())
logger.propagate = False

# 配置 ultralytics 日志
ultralytics_logger = logging.getLogger("ultralytics")
ultralytics_logger.setLevel(logging.WARNING)
for handler in ultralytics_logger.handlers[:]:
    ultralytics_logger.removeHandler(handler)
ultralytics_logger.addHandler(_CleanLogHandler())
ultralytics_logger.propagate = False


def _summarize_model_arch(model: YOLO) -> str:
    """提取模型架构摘要"""
    try:
        m = model.model
        n_layers = len(list(m.modules()))
        total_params = sum(p.numel() for p in m.parameters())
        
        mod_counts = {}
        for mod in m.modules():
            tn = type(mod).__name__
            if tn not in ("Module", "Sequential", "ModuleList", "Conv2d", "BatchNorm2d", "SiLU", "Concat"):
                short = tn.split(".")[-1]
                mod_counts[short] = mod_counts.get(short, 0) + 1
        
        parts = [f"{c}×{n}" for n, c in sorted(mod_counts.items(), key=lambda x: -x[1])]
        params_m = f"{total_params / 1e6:.2f}M" if total_params > 1e6 else f"{total_params:,}"
        return f"Model: {n_layers} layers | {params_m} params | {', '.join(parts)}"
    except Exception:
        return ""


class _NullIO(io.TextIOBase):
    def write(self, _):
        return len(_) if _ else 0
    def flush(self):
        pass


_NULL_IO = _NullIO()


@contextmanager
def _suppress_ultralytics_output():
    """临时抑制 ultralytics 的 stdout/stderr 输出"""
    with redirect_stdout(_NULL_IO), redirect_stderr(_NULL_IO):
        yield


def find_resume_checkpoint(project: str, name: str) -> Optional[Path]:
    """查找可恢复的检查点"""
    run_dir = Path(project) / name
    candidates = [
        run_dir / 'last.pt',
        run_dir / 'weights' / 'last.pt',
        run_dir / 'weights' / 'best.pt',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_distill_training(config_path: str, resume: str = ""):
    """
    运行完整的蒸馏训练流程
    
    流程：
      1. 加载配置和学生模型
      2. 初始化蒸馏组件（教师模型 + 损失函数 + 调度器）
      3. 执行训练（warm-up → 蒸馏阶段）
      4. 自动保存蒸馏日志
      5. 可选：自动运行评估
    
    Args:
        config_path: YAML 配置文件路径
        resume: 'auto' 或指定 checkpoint 路径
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = expand_env_vars(yaml.safe_load(f) or {})
    
    distill_cfg = config.get('distillation', {})
    train_cfg = config.get('training', {})
    output_cfg = config.get('output', {})
    
    # Banner 由 main.py 统一输出，此处不再重复

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 1：配置信息输出（简洁版）
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"配置文件: {config_path}")
    logger.info("=" * 56)
    
    _PROJECT_ROOT = Path(config_path).resolve().parent.parent  # configs/ -> 项目根目录

    def _resolve_weight(path_str):
        """解析权重路径：绝对路径直接使用，相对路径相对于项目根目录"""
        p = Path(path_str)
        if p.is_absolute():
            return p
        candidate = (_PROJECT_ROOT / p).resolve()
        return candidate if candidate.exists() else p

    total_epochs = int(train_cfg.get('epochs', 10))
    warm_epochs = int(distill_cfg.get('warm_epochs', 5))

    # 输出数据集和训练配置概览（前端可解析）
    data_yaml = train_cfg.get('data_yaml', 'coco128.yaml')
    imgsz = int(train_cfg.get('imgsz', 640))
    batch_size = int(train_cfg.get('batch', 16))
    device = str(train_cfg.get('device', 0))
    
    # 尝试读取数据集信息
    try:
        import yaml as _yaml
        _data_path = _resolve_weight(data_yaml) if not Path(data_yaml).is_absolute() else Path(data_yaml)
        with open(_data_path, 'r', encoding='utf-8') as _df:
            _data_cfg = expand_env_vars(_yaml.safe_load(_df) or {})
        _train_list = _data_cfg.get('train', '')
        _nc = _data_cfg.get('nc', '?')
        _names = _data_cfg.get('names', [])
        _nimages = 0
        if _train_list:
            import glob as _glob
            _exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            _train_list_path = Path(_train_list)
            if not _train_list_path.is_absolute():
                _train_list_path = _data_path.parent / _train_list_path
            for _e in _exts:
                _nimages += len(_glob.glob(str(_train_list_path / _e))) if _train_list_path.is_dir() else 0
        logger.info(f"[TRAIN_INFO] dataset={Path(data_yaml).name} images={_nimages} classes={_nc} "
                    f"imgsz={imgsz} batch={batch_size} device={device} epochs={total_epochs}")
    except Exception:
        logger.info(f"[TRAIN_INFO] dataset={Path(data_yaml).name} imgsz={imgsz} "
                    f"batch={batch_size} device={device} epochs={total_epochs}")
    
    # 加载学生模型
    student_weight = distill_cfg.get('student_weight', 'yolov8n.pt')
    _base_student_weight = str(_resolve_weight(student_weight))  # 基础权重（架构定义）
    resume_path = None

    if resume:
        if resume == 'auto':
            resume_path = find_resume_checkpoint(output_cfg.get('project', 'runs/distill'), output_cfg.get('name', 'exp'))
            if resume_path is None:
                logger.warning("未找到可续训 checkpoint，改为从头训练")
            else:
                logger.info(f"[RESUME] 自动检测到 checkpoint: {resume_path}")
        else:
            resume_path = Path(resume)
            if not resume_path.is_absolute():
                resume_path = (_PROJECT_ROOT / resume_path).resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"指定的断点权重不存在: {resume_path}")
            logger.info(f"[RESUME] 加载指定 checkpoint: {resume_path}")

    
    # 【v4内存修复】resume 时使用基础权重创建 YOLO 对象（仅获取架构），
    # 让 ultralytics 内部通过 resume=True 加载 checkpoint。
    # 旧版错误：直接用 YOLO(checkpoint_pt) 预加载 → train(resume=True) 再次加载 → 双重内存占用！
    actual_load_weight = _base_student_weight if resume_path else _base_student_weight

    # 【OOM修复】resume 模式启动前主动释放 GPU 缓存
    # 场景：旧进程被 kill 后 GPU 显存可能未完全回收，新进程加载前先清理
    if resume_path is not None:
        import gc as _gc
        import torch as _torch
        logger.info("[RESUME] 清理残留 GPU 显存...")
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.reset_peak_memory_stats()
            # 等待 CUDA 异步操作完成，确保显存真正释放
            if hasattr(_torch.cuda, 'synchronize'):
                try:
                    _torch.cuda.synchronize()
                except Exception:
                    pass
        time.sleep(1)  # 给系统时间完成资源回收
    
    logger.info(f"[INIT 1/6] 加载学生模型: {actual_load_weight}"
                + (f" (resume from: {resume_path})" if resume_path else ""))
    
    with _suppress_ultralytics_output():
        student_model = YOLO(actual_load_weight)
    if resume_path is not None:
        try:
            # 仅设置 ckpt_path 引用，不预加载权重
            student_model.ckpt_path = str(resume_path)
        except Exception:
            pass
    
    logger.info("[INIT 2/6] 学生模型加载完成 ✓")
    arch_summary = _summarize_model_arch(student_model)
    if arch_summary:
        logger.info(f"         {arch_summary}")
    
    # 设置蒸馏参数
    teacher_weight_raw = distill_cfg.get('teacher_weight', '')
    teacher_weight = str(_resolve_weight(teacher_weight_raw)) if teacher_weight_raw else ''
    
    AdaptiveKDTrainer.set_kd_params(
        teacher_path=teacher_weight,
        alpha_init=float(distill_cfg.get('alpha_init', 0.5)),
        T_max=float(distill_cfg.get('T_max', 6.0)),
        T_min=float(distill_cfg.get('T_min', 1.5)),
        warm_epochs=int(distill_cfg.get('warm_epochs', 5)),
        w_kd=float(distill_cfg.get('w_kd', 0.5)),
        w_focal=float(distill_cfg.get('w_focal', 0.3)),
        w_feat=float(distill_cfg.get('w_feat', 0.0)),
        scale_boost=float(distill_cfg.get('scale_boost', 2.0)),
        focal_gamma=float(distill_cfg.get('focal_gamma', 2.0)),
    )
    
    total_epochs = int(train_cfg.get('epochs', 10))
    warm_epochs = int(distill_cfg.get('warm_epochs', 5))
    
    logger.info("蒸馏训练器: AdaptiveKDTrainer")
    logger.info("-" * 56)
    logger.info(f"  epochs={total_epochs} | warmup={warm_epochs} | "
               f"α={distill_cfg.get('alpha_init', 0.5)} | "
               f"T:[{distill_cfg.get('T_max', 6.0)}→{distill_cfg.get('T_min', 1.5)}]")
    logger.info(f"  w_kd={distill_cfg.get('w_kd', 0.5)} | "
               f"w_focal={distill_cfg.get('w_focal', 0.3)} | "
               f"teacher={os.path.basename(str(distill_cfg.get('teacher_weight', '')))}")
    logger.info("=" * 56)

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 2：构建训练参数
    # ══════════════════════════════════════════════════════════════════════
    train_args = {
        "data": train_cfg.get('data_yaml', 'coco128.yaml'),
        "epochs": total_epochs,
        "imgsz": int(train_cfg.get('imgsz', 640)),
        "batch": int(train_cfg.get('batch', 16)),
        "workers": int(train_cfg.get('workers', 8)),
        "device": train_cfg.get('device', 0),
        "lr0": float(train_cfg.get('lr0', 0.01)),
        "lrf": float(train_cfg.get('lrf', 0.1)),
        "warmup_epochs": int(train_cfg.get('warmup_epochs', 3)),
        "mosaic": float(train_cfg.get('mosaic', 0.8)),
        "mixup": float(train_cfg.get('mixup', 0.1)),
        "close_mosaic": int(train_cfg.get('close_mosaic', 1)),
        "amp": bool(train_cfg.get('amp', True)),
        "project": output_cfg.get('project', 'runs/distill'),
        "name": output_cfg.get('name', 'exp'),
        "verbose": False,
        "plots": False,
    }
    
    logger.info(f"[INIT 3/6] 蒸馏参数配置完成 ✓")
    
    # 处理 resume 参数
    if resume:
        if resume == 'auto' and resume_path is None:
            logger.warning("未找到可续训 checkpoint，改为从头训练")
        elif resume_path is not None:
            # 【关键修复】传 checkpoint 路径而非布尔值！
            # ultralytics check_resume() 对布尔值 True 会 fallback 到 get_latest_run() 全局搜索，
            # 可能找不到目标 checkpoint 或找到错误文件。
            # 直接传路径字符串才能确保加载指定断点。
            train_args['resume'] = str(resume_path)
            logger.info(f"断点续训: {resume_path}")
    
    # 环境变量设置
    os.environ['ULTRALYTICS_VERBOSE'] = 'False'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['DATAMODULE_WORKERS'] = '0'
    os.environ['NUM_WORKERS'] = '0'
    
    # ══════════════════════════════════════════════════════════════════════
    # 阶段 3：执行训练
    # ══════════════════════════════════════════════════════════════════════
    _distill_trainer_cls = AdaptiveKDTrainer
    
    logger.info(f"[INIT 4/6] 启动训练引擎... (数据集: {train_args['data']} | imgsz={train_args['imgsz']} | batch={train_args['batch']})")
    
    try:
        results = student_model.train(trainer=_distill_trainer_cls, **train_args)
    except Exception as e:
        msg = str(e)
        if 'nothing to resume' in msg.lower() or 'finished, nothing to resume' in msg.lower():
            logger.warning("checkpoint 已完成训练，改为从头开始")
            train_args.pop('resume', None)
            results = student_model.train(trainer=_distill_trainer_cls, **train_args)
        else:
            raise
    
    # ══════════════════════════════════════════════════════════════════════
    # 阶段 4：保存蒸馏日志
    # ══════════════════════════════════════════════════════════════════════
    trainer = getattr(student_model, 'trainer', None)
    if trainer is None:
        logger.warning("未获取到 student_model.trainer，蒸馏日志保存失败")
    elif not hasattr(trainer, 'get_distill_log'):
        logger.warning("trainer 对象不支持 get_distill_log，无法保存蒸馏日志")
    else:
        log_data = trainer.get_distill_log()
        if log_data:
            run_dir = Path(getattr(trainer, 'save_dir', ''))
            if not run_dir or not run_dir.exists():
                run_dir = Path(train_args['project']) / train_args['name']
                logger.warning(f"trainer.save_dir 不可用，回退到配置目录: {run_dir}")
            else:
                logger.info(f"使用 trainer.save_dir 保存蒸馏日志: {run_dir}")

            # 保存 distill_log.json（详细蒸馏数据）
            log_path = run_dir / 'distill_log.json'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"蒸馏日志已保存: {log_path}")
            
            # 输出最终蒸馏统计
            distill_entries = [e for e in log_data if e.get('phase') == 'distill']
            warm_entries = [e for e in log_data if e.get('phase') == 'warm']
            logger.info("-" * 56)
            logger.info("训练完成统计:")
            logger.info(f"  Warm-up epochs:   {len(warm_entries)}")
            logger.info(f"  Distill epochs:   {len(distill_entries)}")
            if distill_entries:
                final_task = distill_entries[-1].get('task_loss', '--')
                final_kd = distill_entries[-1].get('kd_loss', '--')
                final_alpha = distill_entries[-1].get('alpha', '--')
                final_temp = distill_entries[-1].get('temperature', '--')
                logger.info(f"  Final task_loss:  {final_task:.4f}" if isinstance(final_task, float) else f"  Final task_loss:  {final_task}")
                logger.info(f"  Final kd_loss:    {final_kd:.4f}" if isinstance(final_kd, float) else f"  Final kd_loss:    {final_kd}")
                logger.info(f"  Final alpha:       {final_alpha:.3f}" if isinstance(final_alpha, float) else f"  Final alpha:       {final_alpha}")
                logger.info(f"  Final temperature: {final_temp:.2f}" if isinstance(final_temp, float) else f"  Final temperature: {final_temp}")
            logger.info("=" * 56)
        else:
            logger.warning("trainer.get_distill_log() 返回空，未生成蒸馏日志")

    # ══════════════════════════════════════════════════════════════════════
    # 阶段 4：自动评估（如果配置了）
    # ══════════════════════════════════════════════════════════════════════
    eval_enabled = output_cfg.get('auto_eval', True)
    if eval_enabled:
        run_dir = Path(train_args['project']) / train_args['name']
        best_pt = run_dir / 'weights' / 'best.pt'
        last_pt = run_dir / 'weights' / 'last.pt'
        
        model_to_eval = None
        if best_pt.exists():
            model_to_eval = best_pt
        elif last_pt.exists():
            model_to_eval = last_pt
        
        if model_to_eval:
            logger.info(f"[EVAL] 开始自动评估: {model_to_eval}")
            try:
                with _suppress_ultralytics_output():
                    eval_model = YOLO(str(model_to_eval))
                    eval_results = eval_model.val(
                        data=train_cfg.get('data_yaml', 'coco128.yaml'),
                        imgsz=int(train_cfg.get('imgsz', 640)),
                        batch=int(train_cfg.get('batch', 16)),
                        verbose=False,
                    )
                
                # 提取关键指标
                if hasattr(eval_results, 'box') and eval_results.box is not None:
                    map50 = getattr(eval_results.box, 'map50', None)
                    map50_95 = getattr(eval_results.box, 'map', None)
                    
                    logger.info("[EVAL] 评估完成:")
                    if map50 is not None:
                        logger.info(f"  mAP@50:     {map50:.4f}")
                    if map50_95 is not None:
                        logger.info(f"  mAP@50-95:  {map50_95:.4f}")
                    
                    # 提取并保存每类指标
                    ap_per_cls = getattr(eval_results.box, 'ap_per_class', None)
                    class_names = getattr(eval_results, 'names', {}) or {}
                    nc = len(class_names) if class_names else (len(ap_per_cls) if ap_per_cls is not None else 0)

                    per_class_data = None
                    if ap_per_cls is not None and nc > 0:
                        p_arr = getattr(eval_results.box, 'p', None)
                        r_arr = getattr(eval_results.box, 'r', None)
                        labels_list, map_list, prec_list, rec_list = [], [], [], []
                        for i in range(nc):
                            labels_list.append(class_names.get(i, f'class{i}'))
                            map_list.append(float(ap_per_cls[i]))
                            prec_list.append(float(p_arr[i]) if p_arr is not None and i < len(p_arr) else 0.0)
                            rec_list.append(float(r_arr[i]) if r_arr is not None and i < len(r_arr) else 0.0)
                        per_class_data = {
                            'labels': labels_list,
                            'map': map_list,
                            'precision': prec_list,
                            'recall': rec_list,
                            'source': 'auto_eval',
                        }
                        logger.info(f"  每类指标已提取 ({nc} 个类别)")
                    
                    # 保存评估结果
                    eval_result_path = run_dir / 'eval_result.json'
                    eval_data = {
                        'model': str(model_to_eval),
                        'map50': float(map50) if map50 is not None else None,
                        'map50_95': float(map50_95) if map50_95 is not None else None,
                    }
                    if per_class_data:
                        eval_data['per_class'] = per_class_data
                        pc_path = run_dir / 'per_class_metrics.json'
                        with open(pc_path, 'w', encoding='utf-8') as _pf:
                            import json as _json
                            from datetime import datetime as _dt
                            per_class_data['generated_at'] = _dt.now().isoformat()
                            per_class_data['epoch'] = total_epochs
                            _json.dump(per_class_data, _pf, indent=2, ensure_ascii=False)

                    with open(eval_result_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"  结果已保存: {eval_result_path}")
            except Exception as eval_err:
                logger.warning(f"[EVAL] 评估失败（不影响训练）: {eval_err}")
    
    logger.info("训练全部完成！")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeDistillDet 蒸馏训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default="", 
                        help="断点续训: 'auto' 自动查找或指定路径")
    args = parser.parse_args()
    
    try:
        results = run_distill_training(args.config, resume=args.resume)
    except Exception as e:
        logger.error(f"[FATAL] 训练失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
