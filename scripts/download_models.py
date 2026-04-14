"""
模型下载脚本 - 下载 YOLOv8 预训练权重
=====================================
运行: python scripts/download_models.py

下载的模型会保存到项目根目录下的 models/ 文件夹
同时 ultralytics 也会缓存一份到 ~/.cache/ultralytics/
"""

import os
import sys
import shutil
from pathlib import Path

# 项目根目录（硬编码避免路径问题）
ROOT = Path(r'D:\Personal_Files\Projects\EdgeDistillDet')
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 模型配置列表
MODELS = {
    "yolov8n.pt": ("YOLOv8 Nano",   "~6MB",  "学生模型(推荐)"),
    "yolov8s.pt": ("YOLOv8 Small",   "~22MB", "学生模型备选"),
    "yolov8m.pt": ("YOLOv8 Medium",  "~50MB", "教师模型(推荐)"),
    "yolov8l.pt": ("YOLOv8 Large",   "~80MB", "教师模型备选"),
    "yolov8x.pt": ("YOLOv8 X-Large", "~130MB","强教师模型"),
}

# ultralytics 缓存目录
ULTRALYTICS_CACHE = Path.home() / '.cache' / 'ultralytics'

def find_cached_model(model_name):
    """在 ultralytics 缓存中查找已下载的模型"""
    # 常见缓存路径模式
    search_paths = [
        ULTRALYTICS_CACHE,
        Path.home() / '.cache' / 'huggingface' / 'hub',
        Path.home(),
    ]

    for base in search_paths:
        if not base.exists():
            continue
        for p in base.rglob(model_name):
            if p.is_file() and p.stat().st_size > 100000:  # >100KB 才是真正的模型文件
                return p
    return None


def download_model(model_name):
    """通过 ultralytics 下载并复制到本地 models/ 目录"""
    dest = MODEL_DIR / model_name
    if dest.exists() and dest.stat().st_size > 1000000:
        print(f"  [已存在] {model_name} ({dest.stat().st_size/1024/1024:.1f}MB)")
        return True

    desc, size, role = MODELS.get(model_name, (model_name, "?", ""))
    print(f"\n{'='*50}")
    print(f"  正在下载: {desc} ({size})")
    print(f"  用途: {role}")
    print(f"{'='*50}")

    try:
        from ultralytics import YOLO
        # 触发 ultralytics 自动从 GitHub 下载
        m = YOLO(model_name)
        # 获取实际文件路径 (pt_path 属性)
        src = getattr(m.model, 'pt_path', None)

        if src and Path(src).exists():
            shutil.copy2(src, dest)
            sz_mb = dest.stat().st_size / 1024 / 1024
            print(f"  [完成] 已复制到: {dest} ({sz_mb:.1f}MB)")
            return True
        else:
            print(f"  [提示] 模型已由 ultralytics 加载，但未找到缓存路径")
            print(f"  可直接在代码中使用 YOLO('{model_name}')")
            return True

    except Exception as e:
        print(f"  [错误] 失败: {e}")
        return False


def main():
    print("=" * 55)
    print("  EdgeDistillDet - 模型下载工具")
    print("=" * 55)
    print(f"\n  输出目录: {MODEL_DIR}\n")

    recommended = ["yolov8n.pt", "yolov8m.pt"]

    print("  推荐组合:")
    print(f"    学生模型: yolov8n.pt  (轻量 ~6MB)")
    print(f"    教师模型: yolov8m.pt  (中等 ~50MB)")
    print()

    success_count = 0
    for model_name in recommended:
        if download_model(model_name):
            success_count += 1

    print("\n" + "=" * 55)
    print(f"  结果: {success_count}/{len(recommended)} 个模型就绪")
    print("=" * 55)

    # 显示可用模型
    pt_files = list(MODEL_DIR.glob("*.pt"))
    txt_files = list(MODEL_DIR.glob("*.txt"))
    if pt_files:
        print(f"\n  本地模型文件:")
        for f in sorted(pt_files):
            sz = f.stat().st_size / 1024 / 1024
            desc = MODELS.get(f.name, ("", ""))[0]
            print(f"    {f.name:<15} {sz:>7.1f}MB  {desc}")

    # 尝试从缓存补充显示
    cache_found = []
    for name in recommended:
        cached = find_cached_model(name)
        if cached and not (MODEL_DIR / name).exists():
            cache_found.append((name, cached))

    if cache_found:
        print(f"\n  ultralytics 缓存中已有(可直接使用):")
        for name, path in cache_found:
            sz = path.stat().st_size / 1024 / 1024
            print(f"    {name:<15} {sz:>7.1f}MB  路径: {path}")


if __name__ == '__main__':
    main()
