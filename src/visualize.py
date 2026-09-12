"""
src/visualize.py
------------------
Phần "Tracking" của thành phần 4: vẽ biểu đồ theo dõi quá trình train (loss +
custom metric) và vẽ ảnh so sánh bounding box dự đoán vs nhãn thực trên 1 batch
mẫu của tập validation.
"""
from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(results_csv: str, custom_metric_csv: Optional[str], out_path: str) -> None:
    """
    Vẽ 2 biểu đồ cạnh nhau:
      (1) Train/Val loss (box, cls, dfl) — đọc từ `results.csv` mà Ultralytics tự
          sinh trong thư mục run sau mỗi epoch.
      (2) Custom Competition Score theo epoch — đọc từ `custom_metric_csv` do
          callback trong `src/train.py` ghi ra (xem `make_metric_callback`).
    """
    if not os.path.isfile(results_csv):
        print(f"[WARN] Không tìm thấy {results_csv}, bỏ qua vẽ biểu đồ loss.")
        return

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    loss_cols = {
        "train/box_loss": ("train box", "-"),
        "train/cls_loss": ("train cls", "-"),
        "train/dfl_loss": ("train dfl", "-"),
        "val/box_loss": ("val box", "--"),
        "val/cls_loss": ("val cls", "--"),
        "val/dfl_loss": ("val dfl", "--"),
    }
    for col, (label, style) in loss_cols.items():
        if col in df.columns:
            axes[0].plot(df["epoch"], df[col], style, label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Train / Val Loss (box, cls, dfl)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if custom_metric_csv and os.path.isfile(custom_metric_csv):
        metric_df = pd.read_csv(custom_metric_csv)
        axes[1].plot(metric_df["epoch"], metric_df["custom_score"], "o-", color="tab:green")
        axes[1].set_ylim(0, 1)
    else:
        axes[1].text(0.5, 0.5, "Chưa có dữ liệu custom metric", ha="center", va="center")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Score (mean, IoU 0.50:0.75:0.05)")
    axes[1].set_title("Custom Competition Metric trên tập Validation")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Đã lưu biểu đồ theo dõi training: {out_path}")


def _read_yolo_label_boxes(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    if not os.path.isfile(label_path):
        return np.empty((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def visualize_val_predictions(
    weights_path: str,
    images_dir: str,
    labels_dir: str,
    out_path: str,
    n_images: int = 16,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1024,
    seed: int = 42,
) -> None:
    """
    Chạy `weights_path` trên `n_images` ảnh NGẪU NHIÊN của `images_dir`, vẽ:
        - Box Ground Truth: màu XANH LÁ (đọc từ file .txt tương ứng trong `labels_dir`).
        - Box dự đoán: màu ĐỎ, kèm confidence.
    rồi ghép thành 1 lưới ảnh duy nhất, lưu ra `out_path`.
    """
    from ultralytics import YOLO  # import cục bộ: module này import được ngay cả khi chưa cài ultralytics

    all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not all_images:
        print(f"[WARN] Không tìm thấy ảnh nào trong {images_dir} để visualize.")
        return

    random.Random(seed).shuffle(all_images)
    sample_images = all_images[:n_images]

    model = YOLO(weights_path)
    results = model.predict(source=sample_images, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

    n = len(results)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, result in zip(axes, results):
        img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB).copy()
        img_h, img_w = img.shape[:2]

        image_id = Path(result.path).stem
        label_path = os.path.join(labels_dir, f"{image_id}.txt")
        gt_boxes = _read_yolo_label_boxes(label_path, img_w, img_h)
        for x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(pred_boxes, pred_confs):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(
                    img, f"{c:.2f}", (int(x1), max(0, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA,
                )

        ax.imshow(img)
        ax.set_title(image_id, fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Xanh lá = Ground Truth | Đỏ = Prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Đã lưu ảnh visualize prediction vs GT: {out_path}")
