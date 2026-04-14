"""
数据集下载脚本
=================
支持以下数据集:
  1. coco128      - COCO子集(128张图, 快速验证训练流程)
  2. visdrone     - VisDrone无人机数据集(需手动下载或提供链接)
  3. VOC          - Pascal VOC数据集

运行: python scripts/download_dataset.py [coco128|visdrone|voc]
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录（硬编码）
ROOT = Path(r'D:\Personal_Files\Projects\EdgeDistillDet')
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def download_coco128():
    """
    下载 COCO128 - ultralytics 内置的小型数据集
    128张图片, COCO 80类, 适合快速验证训练流程是否正常
    """
    target = DATASETS_DIR / "coco128"
    if (target / "images" / "train2017").exists():
        print(f"\n  COCO128 已存在: {target}")
        return str(target)

    print("\n" + "=" * 55)
    print("  下载 COCO128 数据集")
    print("  大小: ~17MB (128张图像)")
    print("  类别: COCO 80类")
    print("=" * 55)

    # 使用 ultralytics 自带的 coco128.yaml
    # 它会自动从 GitHub 下载到 ~/.cache/ultralytics/datasets/
    from ultralytics.data.utils import check_det_dataset

    try:
        info = check_det_dataset("coco128.yaml")
        src_path = Path(info.get("path", ""))
        print(f"\n  ultralytics 下载位置: {src_path}")

        # 复制到项目 datasets 目录
        if src_path.exists() and src_path != target:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src_path, target)
            print(f"  已复制到: {target}")

        # 统计文件数
        train_imgs = list((target / "images" / "train2017").glob("*")) if (target / "images" / "train2017").exists() else []
        val_imgs = list((target / "images" / "val2017").glob("*")) if (target / "images" / "val2017").exists() else []
        print(f"\n  训练集: {len(train_imgs)} 张图像")
        print(f"  验证集: {len(val_imgs)} 张图像")

        return str(target)

    except Exception as e:
        print(f"  错误: {e}")
        return None


def download_visdrone():
    """VisDrone 数据集下载指引"""
    target = DATASETS_DIR / "visdrone"
    if target.exists():
        print(f"\n  VisDrone 目录已存在: {target}")
        return str(target)

    print("\n" + "=" * 55)
    print("  VisDrone 数据集下载说明")
    print("=" * 55)
    print("""
  VisDrone 是无人机视角的小目标检测数据集，非常适合本项目。

  【下载步骤】
  1. 注册账号: https://aiskyeye.com/
  2. 下载数据集:
     - VisDrone2019-DET-train.zip   (~5.4GB, 6471张)
     - VisDrone2019-DET-val.zip     (~1.4GB,  548张)
     - VisDrone2019-DET-test-dev.zip(~1.6GB,  1618张)

  【解压后目录结构】
  VisDrone/
  ├── images/
  │   ├── 10000001234.jpg
  │   └── ...
  └── annotations/
      ├── 10000001234.txt   (YOLO格式标注)
      └── ...

  【转换工具】项目包含 VisDrone→YOLO 格式转换器
  运行: python scripts/convert_visdrone.py <解压路径>
""")

    # 尝试自动转换已有的 VisDrone 格式数据
    common_paths = [
        Path("F:/VisDrone"),
        Path("F:/drone_dataset"),
        Path("D:/VisDrone"),
        Path.home() / "Downloads" / "VisDrone",
    ]
    for p in common_paths:
        if p.exists():
            print(f"\n  发现可能的数据集目录: {p}")
            ans = input("  是否使用此目录? (y/n): ").strip().lower()
            if ans == 'y':
                return str(p)

    print("\n  请先按上述说明下载数据集，然后重新运行此命令。")
    return None


def create_minidataset():
    """
    创建最小测试数据集（仅用于验证训练流程能否跑通）
    包含10张合成的简单图像和对应标注
    """
    import json
    import numpy as np

    name = "mini_test"
    target = DATASETS_DIR / name
    images_train = target / "images" / "train"
    images_val = target / "images" / "val"
    labels_train = target / "labels" / "train"
    labels_val = target / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  创建最小测试数据集: {name}")
    print(f"{'='*55}")

    # 创建简单的合成图像和标注
    np.random.seed(42)
    nc = 2  # 2个类别: car, person
    names = ["car", "person"]

    for split, img_dir, lbl_dir, count in [
        ("train", images_train, labels_train, 8),
        ("val", images_val, labels_val, 4),
    ]:
        for i in range(count):
            # 生成随机图像 (RGB噪声)
            img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            img_path = str(img_dir / f"img_{i:04d}.jpg")
            
            # 用 PIL 或 opencv 保存
            try:
                from PIL import Image
                Image.fromarray(img).save(img_path)
            except ImportError:
                import cv2
                cv2.imwrite(img_path, img)

            # 生成随机标注 (YOLO格式: class cx cy w h, 归一化)
            n_objs = np.random.randint(1, 4)
            with open(str(lbl_dir / f"img_{i:04d}.txt"), 'w') as f:
                for _ in range(n_objs):
                    cls_id = np.random.randint(0, nc)
                    cx, cy = np.random.uniform(0.15, 0.85, 2)
                    w, h = np.random.uniform(0.05, 0.25, 2)
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 写入 data.yaml
    yaml_content = f"""# 最小测试数据集 - 仅用于验证训练流程
path: ../datasets/{name}
train: images/train
val: images/val

nc: {nc}
names:
"""
    for nm in names:
        yaml_content += f"  - {nm}\n"

    yaml_path = target / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # 同时复制一份到 configs
    config_yaml = ROOT / "configs" / f"dataset_{name}.yaml"
    shutil.copy(yaml_path, config_yaml)

    print(f"\n  数据集创建完成:")
    print(f"    路径: {target}")
    print(f"    训练: {count} 张 | 验证: {4 if split=='train' else 0} 张")
    print(f"    类别: {names}")
    print(f"    YAML: {config_yaml}")
    print(f"\n  配置已写入: configs/dataset_{name}.yaml")

    return str(config_yaml)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 55)
    print("  EdgeDistillDet - 数据集下载工具")
    print("=" * 55)
    print(f"\n  数据集保存到: {DATASETS_DIR}\n")

    options = {
        "coco128": ("COCO128 子集", "~17MB, 80类, 128张, 快速验证"),
        "visdrone": ("VisDrone 无人机", "~8GB, 10类, 小目标专用"),
        "mini": ("合成迷你数据集", "<1MB, 2类, 仅验证流程"),
    }

    if not dataset:
        print("  可用选项:")
        for k, (_, desc) in options.items():
            print(f"    python scripts/download_dataset.py {k:<12} # {desc}")
        print()
        print("  推荐新手: python scripts/download_dataset.py mini")
        print("  推荐验证: python scripts/download_dataset.py coco128")
        return

    dataset = dataset.lower()

    if dataset == "coco128":
        result = download_coco128()
    elif dataset == "visdrone":
        result = download_visdrone()
    elif dataset in ("mini", "minidataset", "test"):
        result = create_minidataset()
    else:
        print(f"  未知选项: {dataset}")
        print(f"  可用: {', '.join(options.keys())}")
        return

    if result:
        print(f"\n  结果: 成功 -> {result}")


if __name__ == '__main__':
    main()
