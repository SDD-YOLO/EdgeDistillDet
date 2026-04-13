"""
core/detection/small_target_detector.py
=========================================
小目标检测推理封装器（SmallTargetDetector）

原创功能：
  1. 自适应置信度校准   —— 对小框（面积 < small_thresh）单独降低阈值，提升召回
  2. 多尺度滑窗融合     —— 对高分辨率输入自动分片推理后 NMS 融合
  3. 目标尺寸分级统计   —— 将检测结果按 tiny/small/medium/large 分级汇报
  4. 推理结果结构化输出  —— 返回标准 DetResult 数据类，便于下游处理
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SingleDetection:
    xyxy: Tuple[float, float, float, float]  # 像素坐标
    conf: float
    cls:  int
    area: float                              # 像素面积
    size_category: str                       # tiny / small / medium / large


@dataclass
class DetResult:
    image_path: str
    detections: List[SingleDetection] = field(default_factory=list)
    inference_ms: float = 0.0
    imgsz: Tuple[int, int] = (0, 0)

    @property
    def tiny_count(self):
        return sum(1 for d in self.detections if d.size_category == "tiny")

    @property
    def small_count(self):
        return sum(1 for d in self.detections if d.size_category == "small")

    def summary(self) -> str:
        lines = [
            f"图像: {self.image_path}",
            f"推理耗时: {self.inference_ms:.1f} ms  |  检测总数: {len(self.detections)}",
            f"  tiny(<16²): {self.tiny_count}  small(<32²): {self.small_count}",
        ]
        for d in self.detections:
            x1,y1,x2,y2 = d.xyxy
            lines.append(
                f"  [{d.size_category:6s}] cls={d.cls} conf={d.conf:.3f} "
                f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) area={d.area:.0f}px²"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 尺寸分级
# ─────────────────────────────────────────────────────────────────────────────
def _categorize_size(area_px: float) -> str:
    if area_px < 16 * 16:
        return "tiny"
    elif area_px < 32 * 32:
        return "small"
    elif area_px < 96 * 96:
        return "medium"
    else:
        return "large"


# ─────────────────────────────────────────────────────────────────────────────
# 多尺度滑窗分片（针对超高分辨率图像）
# ─────────────────────────────────────────────────────────────────────────────
def _tile_image(
    img: np.ndarray, tile_size: int = 640, overlap: float = 0.2
) -> List[Tuple[np.ndarray, int, int]]:
    """
    将大图切分为若干 tile_size×tile_size 的分片，返回 (片段, x_offset, y_offset)。
    """
    H, W = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            tile = img[y:y2, x:x2]
            if tile.shape[0] < tile_size // 4 or tile.shape[1] < tile_size // 4:
                continue
            tiles.append((tile, x, y))
    return tiles


def _nms_boxes(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5
) -> List[int]:
    """简洁 NMS，返回保留的索引列表。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou <= iou_thresh]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────
class SmallTargetDetector:
    """
    小目标自适应检测推理器。

    参数：
      weight_path        : YOLO 权重路径
      base_conf          : 基础置信度阈值（大目标）
      small_conf_factor  : 小目标置信度降低因子（对 tiny/small 类别乘以此系数）
      base_iou           : NMS IOU 阈值
      device             : 推理设备
      tile_large         : 是否启用大图滑窗分片
      tile_size          : 分片尺寸
      small_area_px      : 判定为"小目标"的像素面积阈值（用于自适应置信度）
    """

    # 尺寸阈值（px²）
    TINY_AREA  = 16 * 16
    SMALL_AREA = 32 * 32

    def __init__(
        self,
        weight_path: str,
        base_conf: float = 0.25,
        small_conf_factor: float = 0.6,
        base_iou: float = 0.45,
        device: str = "cpu",
        tile_large: bool = False,
        tile_size: int = 640,
        small_area_px: float = 32 * 32,
    ):
        self.model = YOLO(weight_path)
        self.base_conf = base_conf
        self.small_conf = max(0.05, base_conf * small_conf_factor)
        self.base_iou = base_iou
        self.device = device
        self.tile_large = tile_large
        self.tile_size = tile_size
        self.small_area_px = small_area_px

    # ── 单张图像推理 ──────────────────────────────────────────────────────────
    def infer(self, image_path: str, imgsz: int = 640) -> DetResult:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        H, W = img.shape[:2]
        result = DetResult(image_path=image_path, imgsz=(W, H))

        t0 = time.perf_counter()
        if self.tile_large and (H > self.tile_size * 1.5 or W > self.tile_size * 1.5):
            dets = self._infer_tiled(img, imgsz)
        else:
            dets = self._infer_single(img, imgsz)
        result.inference_ms = (time.perf_counter() - t0) * 1000
        result.detections = dets
        return result

    def _infer_single(self, img: np.ndarray, imgsz: int) -> List[SingleDetection]:
        raw = self.model.predict(
            img,
            imgsz=imgsz,
            conf=self.small_conf,  # 先用低阈值，后续再过滤
            iou=self.base_iou,
            device=self.device,
            verbose=False,
        )
        return self._parse_raw(raw[0], img.shape[:2])

    def _infer_tiled(self, img: np.ndarray, imgsz: int) -> List[SingleDetection]:
        tiles = _tile_image(img, self.tile_size, overlap=0.2)
        all_boxes, all_scores, all_cls = [], [], []

        for tile, ox, oy in tiles:
            raw = self.model.predict(
                tile,
                imgsz=imgsz,
                conf=self.small_conf,
                iou=self.base_iou,
                device=self.device,
                verbose=False,
            )
            if raw[0].boxes is None or len(raw[0].boxes) == 0:
                continue
            boxes = raw[0].boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] += ox
            boxes[:, [1, 3]] += oy
            all_boxes.append(boxes)
            all_scores.append(raw[0].boxes.conf.cpu().numpy())
            all_cls.append(raw[0].boxes.cls.cpu().numpy().astype(int))

        if not all_boxes:
            return []

        boxes  = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        clses  = np.concatenate(all_cls)

        keep = _nms_boxes(boxes, scores, self.base_iou)
        H, W = img.shape[:2]
        dets = []
        for idx in keep:
            x1,y1,x2,y2 = boxes[idx]
            area = (x2-x1)*(y2-y1)
            # 自适应置信度过滤
            thresh = self.small_conf if area < self.small_area_px else self.base_conf
            if scores[idx] < thresh:
                continue
            dets.append(SingleDetection(
                xyxy=(float(x1),float(y1),float(x2),float(y2)),
                conf=float(scores[idx]),
                cls=int(clses[idx]),
                area=float(area),
                size_category=_categorize_size(area),
            ))
        return dets

    def _parse_raw(self, raw, img_shape) -> List[SingleDetection]:
        dets = []
        if raw.boxes is None or len(raw.boxes) == 0:
            return dets
        for box in raw.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            area  = (x2-x1)*(y2-y1)
            thresh = self.small_conf if area < self.small_area_px else self.base_conf
            if conf < thresh:
                continue
            dets.append(SingleDetection(
                xyxy=(x1,y1,x2,y2),
                conf=conf,
                cls=cls,
                area=area,
                size_category=_categorize_size(area),
            ))
        return dets

    # ── 批量推理（返回汇总统计）────────────────────────────────────────────────
    def batch_infer_summary(
        self, image_paths: List[str], imgsz: int = 640
    ) -> dict:
        results = []
        for p in image_paths:
            try:
                results.append(self.infer(p, imgsz))
            except Exception as e:
                print(f"  ⚠ 推理失败 {p}: {e}")

        total = len(results)
        if total == 0:
            return {}

        all_dets     = [d for r in results for d in r.detections]
        tiny_total   = sum(1 for d in all_dets if d.size_category == "tiny")
        small_total  = sum(1 for d in all_dets if d.size_category == "small")
        avg_ms       = sum(r.inference_ms for r in results) / total

        return {
            "total_images":    total,
            "total_dets":      len(all_dets),
            "tiny_dets":       tiny_total,
            "small_dets":      small_total,
            "avg_infer_ms":    round(avg_ms, 2),
            "avg_fps":         round(1000 / avg_ms, 2) if avg_ms > 0 else 0,
        }
