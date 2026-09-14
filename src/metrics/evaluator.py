"""
src/metrics/evaluator.py
-------------------------
Cài đặt metric chính thức của cuộc thi Global Wheat Detection (chỉ dùng NumPy,
không cần Ultralytics/PyTorch).

Tài liệu chính thức:
https://www.kaggle.com/competitions/global-wheat-detection/overview/evaluation

Cách tính, cho MỖI ảnh:
    1. Sắp xếp box dự đoán giảm dần theo confidence.
    2. Với mỗi ngưỡng IoU trong [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]: ghép từng box
       dự đoán với box thật (Ground Truth) chưa bị ghép, theo thứ tự confidence
       giảm dần:
         - Ghép được (IoU lớn hơn ngưỡng) -> TP.
         - Không ghép được -> FP.
         - Box thật không được box nào ghép -> FN.
       Score = TP / (TP + FP + FN).
    3. Điểm của 1 ảnh = trung bình Score trên 6 ngưỡng.
Điểm cuối cùng = trung bình điểm của tất cả ảnh trong tập đánh giá.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

IOU_THRESHOLDS: np.ndarray = np.round(np.arange(0.50, 0.76, 0.05), 2)  # [0.50, 0.55, ..., 0.75]


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Tính IoU giữa mọi cặp (box dự đoán, box thật).

    Args:
        pred_boxes: (P, 4) [x_min, y_min, x_max, y_max], đơn vị pixel.
        gt_boxes: (G, 4) [x_min, y_min, x_max, y_max], đơn vị pixel.

    Returns:
        Ma trận IoU kích thước (P, G).
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    px1, py1, px2, py2 = (pred_boxes[:, i][:, None] for i in range(4))
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i][None, :] for i in range(4))

    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    pred_area = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    gt_area = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union_area = pred_area + gt_area - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0).astype(np.float32)


def _greedy_match_at_threshold(iou_matrix: np.ndarray, threshold: float) -> float:
    """
    Ghép box dự đoán với box thật tại 1 ngưỡng IoU. Giả định các hàng của
    `iou_matrix` (pred) đã được sắp xếp giảm dần theo confidence từ trước.

    Score = TP / (TP + FP + FN). Quy ước 2 trường hợp biên:
        - Không có box thật, không có box dự đoán -> 1.0 (đúng hoàn toàn).
        - Không có box thật, có box dự đoán -> 0.0 (mọi box dự đoán đều là FP).
    """
    n_pred, n_gt = iou_matrix.shape

    if n_gt == 0 and n_pred == 0:
        return 1.0

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    fp = 0

    for pred_idx in range(n_pred):
        if n_gt == 0:
            fp += 1
            continue
        ious = iou_matrix[pred_idx].copy()
        ious[gt_matched] = -1.0  # box thật đã bị ghép rồi thì không cho ghép lại
        best_gt = int(np.argmax(ious))
        best_iou = ious[best_gt]

        if best_iou > threshold:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1

    fn = int(np.sum(~gt_matched))
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 1.0


def image_score(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> float:
    """
    Tính điểm của 1 ảnh = trung bình Score trên toàn bộ `thresholds`.

    Args:
        pred_boxes: (P, 4) xyxy pixel, không cần sắp xếp sẵn theo confidence.
        pred_scores: (P,) confidence tương ứng từng box dự đoán.
        gt_boxes: (G, 4) xyxy pixel.
        thresholds: danh sách ngưỡng IoU, mặc định [0.50, 0.55, ..., 0.75].

    Returns:
        Điểm của ảnh, giá trị trong [0, 1].
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if len(pred_boxes) > 0:
        order = np.argsort(-pred_scores)  # sắp xếp giảm dần theo confidence
        pred_boxes = pred_boxes[order]

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    scores = [_greedy_match_at_threshold(iou_matrix, t) for t in thresholds]
    return float(np.mean(scores))


def competition_score(
    all_pred_boxes: Sequence[np.ndarray],
    all_pred_scores: Sequence[np.ndarray],
    all_gt_boxes: Sequence[np.ndarray],
    thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> float:
    """
    Metric cuối cùng của cuộc thi = trung bình điểm các ảnh trong tập đánh giá.

    Args:
        all_pred_boxes: list, mỗi phần tử là (P_i, 4) xyxy pixel của 1 ảnh.
        all_pred_scores: list, mỗi phần tử là (P_i,) confidence của 1 ảnh.
        all_gt_boxes: list, mỗi phần tử là (G_i, 4) xyxy pixel của 1 ảnh.

    Returns:
        Điểm cuối cùng, giá trị trong [0, 1].
    """
    n = len(all_gt_boxes)
    assert len(all_pred_boxes) == len(all_pred_scores) == n, (
        "Số lượng ảnh giữa pred_boxes/pred_scores/gt_boxes phải khớp nhau."
    )
    if n == 0:
        return 0.0

    image_scores = [
        image_score(p, s, g, thresholds)
        for p, s, g in zip(all_pred_boxes, all_pred_scores, all_gt_boxes)
    ]
    return float(np.mean(image_scores))
