"""
src/metrics/evaluator.py
-------------------------
Thành phần 3 của pipeline: triển khai ĐỘC LẬP (không phụ thuộc Ultralytics/PyTorch,
chỉ NumPy) đúng metric chính thức của cuộc thi Global Wheat Detection — trung bình
"Score" theo Greedy Matching tại 6 ngưỡng IoU [0.50, 0.55, 0.60, 0.65, 0.70, 0.75].

Tài liệu chính thức:
https://www.kaggle.com/competitions/global-wheat-detection/overview/evaluation

Thuật toán, cho MỖI ảnh:
    1. Sắp xếp pred_boxes giảm dần theo confidence.
    2. Với MỖI ngưỡng t trong {0.50, ..., 0.75}: Greedy Matching giữa pred (theo thứ
       tự confidence giảm dần) và GT chưa bị nhận:
         - pred match được 1 GT còn trống có IoU > t (ưu tiên IoU cao nhất) -> TP.
         - pred không match được GT nào -> FP.
         - GT nào không được match bởi pred nào -> FN.
       Score(t) = TP / (TP + FP + FN).
    3. Image_Score = trung bình cộng Score(t) trên 6 ngưỡng.
Metric cuối cùng = trung bình cộng Image_Score trên toàn bộ ảnh của tập đánh giá.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

IOU_THRESHOLDS: np.ndarray = np.round(np.arange(0.50, 0.76, 0.05), 2)  # [0.50, 0.55, ..., 0.75]


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    IoU giữa MỌI cặp (pred, gt), vector hoá bằng NumPy broadcasting.

    Args:
        pred_boxes: (P, 4) [x_min, y_min, x_max, y_max], đơn vị pixel.
        gt_boxes: (G, 4) [x_min, y_min, x_max, y_max], đơn vị pixel.

    Returns:
        (P, G) ma trận IoU, dtype float32.
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
    Greedy Matching tại MỘT ngưỡng IoU. Giả định các HÀNG của `iou_matrix` (pred)
    ĐÃ được sắp xếp giảm dần theo confidence — hàm này chỉ duyệt tuần tự theo thứ
    tự hàng, không tự sắp xếp lại.

    Score(t) = TP / (TP + FP + FN), với quy ước 2 trường hợp biên của đề bài:
        - Không GT, không pred -> 1.0 (ảnh không có wheat head, model cũng không
          dự đoán gì -> đúng hoàn toàn).
        - Không GT, có pred -> 0.0 (mọi pred đều là FP, denom = FP > 0 -> 0/FP = 0,
          công thức tự nhiên cho ra 0, không cần case riêng).
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
        ious[gt_matched] = -1.0  # loại GT đã bị nhận, không cho match lại (trùng lặp)
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
    Image_Score = trung bình cộng Score(t) trên toàn bộ `thresholds` cho MỘT ảnh.

    Args:
        pred_boxes: (P, 4) xyxy pixel, CHƯA CẦN sắp xếp sẵn theo confidence.
        pred_scores: (P,) confidence tương ứng từng pred_box.
        gt_boxes: (G, 4) xyxy pixel.
        thresholds: danh sách ngưỡng IoU, mặc định [0.50, 0.55, ..., 0.75].

    Returns:
        Image_Score, float trong [0, 1].
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    if len(pred_boxes) > 0:
        order = np.argsort(-pred_scores)  # sắp xếp GIẢM DẦN theo confidence
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
    Metric cuối cùng của cuộc thi = trung bình cộng Image_Score trên toàn bộ ảnh
    của tập đánh giá.

    Args:
        all_pred_boxes: list, mỗi phần tử là (P_i, 4) xyxy pixel của 1 ảnh.
        all_pred_scores: list, mỗi phần tử là (P_i,) confidence của 1 ảnh.
        all_gt_boxes: list, mỗi phần tử là (G_i, 4) xyxy pixel của 1 ảnh.

    Returns:
        Metric cuộc thi, float trong [0, 1].
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
