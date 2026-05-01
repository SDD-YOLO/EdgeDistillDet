"""
EdgeDistillDet — 面向边缘计算的微小目标自适应蒸馏与检测评估系统
====================================================================
主入口 CLI，统一调度训练 / 评估 / 可视化 / 边缘剖析四大子系统。

用法示例：
    python main.py train   --config configs/distill_config.yaml
    python main.py eval    --config configs/eval_config.yaml
    python main.py analyze --dataset /path/to/dataset
    python main.py profile --weight model.pt --device rk3588
"""
import argparse
import sys
import time
from pathlib import Path
from core.logging import get_logger
logger = get_logger(__name__)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
__version__ = '1.1.0'
BANNER = f'\n _____    _           ______ _     _   _ _ _______     _   \n|  ___|  | |          |  _  (_)   | | (_) | |  _  \\   | |  \n| |__  __| | __ _  ___| | | |_ ___| |_ _| | | | | |___| |_ \n|  __|/ _` |/ _` |/ _ \\ | | | / __| __| | | | | | / _ \\ __|\n| |__| (_| | (_| |  __/ |/ /| \\__ \\ |_| | | | |/ /  __/ |_ \n\\____/\\__,_|\\__, |\\___|___/ |_|___/\\__|_|_|_|___/ \\___|\\___|\n             __/ |                                          \n            |___/                                           \n\n  面向边缘计算的微小目标自适应蒸馏与检测评估系统  v{__version__}\n  Edge-Oriented Micro Small-Target Adaptive Distillation & Detection Evaluation System\n'

def cmd_train(args):
    from scripts.train_with_distill import run_distill_training
    run_distill_training(args.config, resume=args.resume)

def cmd_eval(args):
    from scripts.evaluate import run_evaluation
    run_evaluation(args.config)

def cmd_analyze(args):
    from scripts.visualize_dataset import run_dataset_analysis
    run_dataset_analysis(args.dataset, args.output)

def cmd_profile(args):
    from utils.edge_profiler import EdgeProfiler
    profiler = EdgeProfiler(args.weight, target_device=args.device)
    profiler.run_full_profile()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='EdgeDistillDet', description='面向边缘计算的微小目标自适应蒸馏与检测评估系统', formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p_train = sub.add_parser('train', help='启动自适应知识蒸馏训练')
    p_train.add_argument('--config', default='configs/distill_config.yaml', help='蒸馏训练配置文件路径')
    p_train.add_argument('--resume', nargs='?', const='auto', default=None, help='断点续训，传递 checkpoint 路径或使用 auto 自动续训')
    p_eval = sub.add_parser('eval', help='运行多设备综合性能评估')
    p_eval.add_argument('--config', default='configs/eval_config.yaml', help='评估配置文件路径')
    p_ana = sub.add_parser('analyze', help='数据集统计分析与可视化')
    p_ana.add_argument('--dataset', required=True, help='数据集根目录')
    p_ana.add_argument('--output', default='outputs/figures', help='图表输出目录')
    p_prof = sub.add_parser('profile', help='边缘设备部署剖析')
    p_prof.add_argument('--weight', required=True, help='模型权重 .pt 路径')
    p_prof.add_argument('--device', default='rk3588', choices=['rk3588', 'ascend310', 'cpu', 'gpu'], help='目标边缘设备类型')
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.time()
    logger.info(f'EdgeDistillDet CLI started | version={__version__} command={args.command}')
    dispatch = {'train': cmd_train, 'eval': cmd_eval, 'analyze': cmd_analyze, 'profile': cmd_profile}
    dispatch[args.command](args)
    logger.info(f'EdgeDistillDet CLI finished | elapsed_s={time.time() - t0:.1f} command={args.command}')
if __name__ == '__main__':
    main()