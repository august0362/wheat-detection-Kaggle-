"""
src/utils.py
------------
Các hàm dùng chung cho toàn bộ pipeline:
    - Đọc file cấu hình YAML.
    - Cố định seed (để kết quả lặp lại được).
    - Parse chuỗi bbox thô từ train.csv.
    - Convert bbox qua lại giữa 3 định dạng toạ độ:
        COCO   [x_min, y_min, w, h]            (pixel, giống train.csv)
        xyxy   [x_min, y_min, x_max, y_max]    (pixel, dùng để tính IoU/metric)
        YOLO   [x_center, y_center, w, h]      (normalized [0, 1], dùng để ghi file nhãn)
"""
from __future__ import annotations

import ast
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Đọc file cấu hình YAML (vd: configs/kaggle_config.yaml) và trả về dict Python."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho `random`, NumPy và PyTorch (CPU + toàn bộ GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bbox_string(bbox_str: str) -> List[float]:
    """
    Parse 1 chuỗi bbox trong cột "bbox" của train.csv, vd "[834.0, 222.0, 56.0, 36.0]"
    (COCO: x_min, y_min, width, height). Dùng `ast.literal_eval` thay vì `eval()`
    để an toàn hơn (chỉ đọc dữ liệu, không thực thi mã).
    """
    x, y, w, h = ast.literal_eval(bbox_str)
    return [float(x), float(y), float(w), float(h)]


def coco_to_xyxy_clipped(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert box COCO [x_min, y_min, w, h] (pixel) -> xyxy (pixel), đồng thời kéo
    box về nằm trong biên ảnh [0, img_w] x [0, img_h] — vì một số box gốc trong
    train.csv bị lệch nhẹ ra ngoài ảnh.

    Trả về None nếu sau khi kéo về biên, box không còn diện tích (box này phải bị
    loại bỏ, không đưa vào label YOLO hay tính metric).
    """
    x, y, w, h = bbox
    x1 = float(np.clip(x, 0, img_w))
    y1 = float(np.clip(y, 0, img_h))
    x2 = float(np.clip(x + w, 0, img_w))
    y2 = float(np.clip(y + h, 0, img_h))

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return None
    return x1, y1, x2, y2


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """[x_min, y_min, x_max, y_max] (pixel) -> [x_center, y_center, w, h] (normalized [0, 1])."""
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    norm_w = (x2 - x1) / img_w
    norm_h = (y2 - y1) / img_h
    return x_center, y_center, norm_w, norm_h


def yolo_to_xyxy(
    x_center: float, y_center: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """[x_center, y_center, w, h] (normalized [0, 1]) -> [x_min, y_min, x_max, y_max] (pixel)."""
    x1 = (x_center - w / 2.0) * img_w
    y1 = (y_center - h / 2.0) * img_h
    x2 = (x_center + w / 2.0) * img_w
    y2 = (y_center + h / 2.0) * img_h
    return x1, y1, x2, y2


def coco_to_yolo(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert box COCO [x_min, y_min, w, h] (pixel) -> YOLO [x_center, y_center, w, h]
    (normalized [0, 1]). Trả về None nếu box nằm ngoài ảnh (xem `coco_to_xyxy_clipped`).
    """
    clipped = coco_to_xyxy_clipped(bbox, img_w, img_h)
    if clipped is None:
        return None
    return xyxy_to_yolo(*clipped, img_w, img_h)
