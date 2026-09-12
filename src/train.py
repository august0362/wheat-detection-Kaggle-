"""
src/train.py
-------------
Thành phần 4 của pipeline: huấn luyện YOLOv8 (Ultralytics) cho Global Wheat
Detection. Thiết kế để chạy trên Kaggle GPU (code chỉ lưu ở VS Code/GitHub):

    !git pull
    !python -m src.train --config configs/kaggle_config.yaml

Chạy thử nhanh ở local để kiểm tra pipeline không lỗi trước khi đẩy lên Kaggle:

    python -m src.train --config configs/local_config.yaml

Luồng xử lý:
    1. `src/data_prep.py`      chuẩn bị dữ liệu YOLO (bỏ qua nếu đã có sẵn).
    2. `src/augmentations.py`  gắn augmentation Albumentations tuỳ biến vào Ultralytics.
    3. `ultralytics.YOLO.train()` train model thật, có gắn callback `on_model_save`
       tính custom competition metric (`src/metrics/evaluator.py`) trên tập
       validation mỗi `metric_eval_interval` epoch.
    4. `src/visualize.py`      vẽ biểu đồ loss + custom metric, và ảnh so sánh
       prediction vs Ground Truth trên 1 batch mẫu của tập validation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from src.augmentations import patch_ultralytics_albumentations
from src.data_prep import build_yolo_dataset
from src.metrics.evaluator import competition_score
from src.utils import load_config, set_seed, yolo_to_xyxy
from src.visualize import plot_training_history, visualize_val_predictions

# Console mặc định trên Windows không encode được tiếng Việt có dấu -> ép UTF-8 để
# log không crash với UnicodeEncodeError (áp dụng cho cả chạy trên Kaggge/Linux).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _load_val_ground_truth(labels_dir: str, image_size: int) -> Dict[str, np.ndarray]:
    """
    Đọc TOÀN BỘ label .txt của tập val 1 lần duy nhất, denormalize về pixel xyxy,
    dùng làm Ground Truth cố định cho callback tính custom metric mỗi epoch (tránh
    đọc lại + parse lại nhiều lần).

    Toàn bộ ảnh gốc của cuộc thi này đều có kích thước `image_size` x `image_size`
    (mặc định 1024x1024, xem `data.original_image_size` trong config) nên có thể
    denormalize trực tiếp mà không cần đọc từng file ảnh để lấy shape.
    """
    gt_map: Dict[str, np.ndarray] = {}
    for label_path in sorted(Path(labels_dir).glob("*.txt")):
        image_id = label_path.stem
        boxes: List[List[float]] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = map(float, parts)
                boxes.append(list(yolo_to_xyxy(xc, yc, w, h, image_size, image_size)))
        gt_map[image_id] = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
    return gt_map


def make_metric_callback(
    val_images_dir: str,
    gt_map: Dict[str, np.ndarray],
    metric_csv_path: str,
    eval_interval: int,
    conf: float,
    iou: float,
    imgsz: int,
    total_epochs: int,
) -> Callable[[Any], None]:
    """
    Tạo callback gắn vào Ultralytics qua `model.add_callback("on_model_save", ...)`.

    `on_model_save` được Ultralytics gọi NGAY SAU KHI `last.pt`/`best.pt` đã ghi
    xong ra đĩa cho epoch hiện tại (xem `BaseTrainer.save_model`/`_do_train`) —
    đảm bảo trọng số load lên để predict luôn khớp đúng epoch vừa train xong.

    Cách tính (Custom Validation Loop):
        1. Load lại `trainer.last` bằng `YOLO(...)` (model độc lập với model đang
           train, không ảnh hưởng gradient/optimizer state).
        2. Predict toàn bộ ảnh validation với `conf` THẤP (giữ đủ box để Greedy
           Matching xét đúng theo thứ tự confidence, không phải ngưỡng dùng lúc submit).
        3. So khớp với Ground Truth cố định (`gt_map`) bằng
           `src/metrics/evaluator.py::competition_score`.
        4. Ghi đè `metric_csv_path` (epoch, custom_score) — ghi lại mỗi lần để
           không mất tiến độ nếu training bị ngắt giữa chừng.

    Việc load lại model + predict toàn bộ tập val khá tốn thời gian trên dataset
    lớn -> điều chỉnh tần suất bằng `eval_interval` (epoch cuối LUÔN được tính).
    """
    from ultralytics import YOLO

    history: List[Dict[str, float]] = []

    def _callback(trainer: Any) -> None:
        try:
            epoch = int(trainer.epoch) + 1  # trainer.epoch là 0-indexed
            is_final_epoch = epoch >= total_epochs
            if eval_interval > 1 and epoch % eval_interval != 0 and not is_final_epoch:
                return

            weights_path = str(trainer.last)
            metric_model = YOLO(weights_path)
            results = metric_model.predict(
                source=val_images_dir, conf=conf, iou=iou, imgsz=imgsz, verbose=False, stream=True
            )

            all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
            for result in results:
                image_id = Path(result.path).stem
                if image_id not in gt_map:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
                scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))
                all_pred_boxes.append(boxes)
                all_pred_scores.append(scores)
                all_gt_boxes.append(gt_map[image_id])

            score = competition_score(all_pred_boxes, all_pred_scores, all_gt_boxes)
            history.append({"epoch": epoch, "custom_score": score})

            with open(metric_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "custom_score"])
                writer.writeheader()
                writer.writerows(history)

            print(f"[METRIC] Epoch {epoch}: Custom Competition Score (IoU 0.50:0.75:0.05) = {score:.4f}")
        except Exception as e:  # không để lỗi tính metric phụ làm hỏng cả quá trình train chính
            print(f"[WARN] Lỗi khi tính custom metric ở epoch {getattr(trainer, 'epoch', '?')}: {e}")

    return _callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 cho Global Wheat Detection")
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Đã load config: {args.config}")
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ─── Bước 1: Chuẩn bị dữ liệu YOLO ───────────────────────────────────────
    data_yaml_path = build_yolo_dataset(cfg)

    # ─── Bước 2: Gắn augmentation Albumentations tuỳ biến ────────────────────
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("enabled", True):
        patch_ultralytics_albumentations(
            hflip_p=aug_cfg.get("hflip_p", 0.5),
            vflip_p=aug_cfg.get("vflip_p", 0.5),
            brightness_contrast_p=aug_cfg.get("brightness_contrast_p", 0.3),
            hue_sat_value_p=aug_cfg.get("hue_sat_value_p", 0.3),
            blur_p=aug_cfg.get("blur_p", 0.2),
            shift_scale_rotate_p=aug_cfg.get("shift_scale_rotate_p", 0.3),
            min_visibility=aug_cfg.get("min_visibility", 0.3),
        )

    from ultralytics import YOLO

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    output_dir = train_cfg["output_dir"]
    experiment_name = cfg.get("experiment_name", "wheat_yolov8")

    yolo_dataset_dir = Path(data_cfg["yolo_dataset_dir"])
    val_images_dir = str(yolo_dataset_dir / "images" / "val")
    val_labels_dir = str(yolo_dataset_dir / "labels" / "val")

    # ─── Bước 3: Train + custom metric callback ──────────────────────────────
    gt_map = _load_val_ground_truth(val_labels_dir, data_cfg.get("original_image_size", 1024))
    print(f"[INFO] Đã nạp Ground Truth cho {len(gt_map)} ảnh validation (dùng để tính custom metric).")

    run_dir = Path(output_dir) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_csv_path = str(run_dir / "custom_metric_history.csv")

    model = YOLO(model_cfg["variant"])
    model.add_callback(
        "on_model_save",
        make_metric_callback(
            val_images_dir=val_images_dir,
            gt_map=gt_map,
            metric_csv_path=metric_csv_path,
            eval_interval=train_cfg.get("metric_eval_interval", 1),
            conf=train_cfg.get("metric_conf_threshold", 0.1),
            iou=train_cfg.get("metric_nms_iou", 0.6),
            imgsz=train_cfg.get("imgsz", 1024),
            total_epochs=train_cfg["epochs"],
        ),
    )

    # Augmentation gốc của Ultralytics (hsv/degrees/translate/scale/shear/flip) dễ
    # TRÙNG LẶP với pipeline Albumentations tuỳ biến ở Bước 2 -> tắt (=0) để không
    # augment 2 lần chồng chéo lên cùng 1 loại biến đổi. Mosaic/Mixup vẫn giữ
    # nguyên vì không trùng với bất kỳ transform nào trong danh sách yêu cầu.
    native_aug_overrides: Dict[str, float] = (
        dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
             shear=0.0, fliplr=0.0, flipud=0.0)
        if train_cfg.get("disable_native_augmentation", True)
        else {}
    )

    model.train(
        data=data_yaml_path,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg.get("imgsz", 1024),
        batch=train_cfg.get("batch", 16),
        device=train_cfg.get("device", 0),
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=train_cfg.get("lr0", 0.01),
        patience=train_cfg.get("patience", 20),
        seed=seed,
        project=output_dir,
        name=experiment_name,
        exist_ok=True,
        pretrained=model_cfg.get("pretrained", True),
        verbose=True,
        **native_aug_overrides,
    )

    actual_run_dir = Path(model.trainer.save_dir)
    best_weights = actual_run_dir / "weights" / "best.pt"
    results_csv = actual_run_dir / "results.csv"

    # ─── Bước 4: Biểu đồ + visualize prediction vs GT ────────────────────────
    plot_training_history(
        results_csv=str(results_csv),
        custom_metric_csv=metric_csv_path,
        out_path=str(actual_run_dir / "training_history.png"),
    )

    infer_cfg = cfg.get("inference", {})
    visualize_val_predictions(
        weights_path=str(best_weights),
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        out_path=str(actual_run_dir / "val_predictions_vs_gt.png"),
        n_images=cfg.get("visualize", {}).get("n_images", 16),
        conf=infer_cfg.get("conf_threshold", 0.25),
        iou=infer_cfg.get("iou_threshold", 0.5),
        imgsz=train_cfg.get("imgsz", 1024),
        seed=seed,
    )

    print("\n[DONE] Training hoàn tất.")
    print(f"[DONE] Best weights: {best_weights}")
    print(f"[DONE] Biểu đồ training: {actual_run_dir / 'training_history.png'}")
    print(f"[DONE] Ảnh visualize: {actual_run_dir / 'val_predictions_vs_gt.png'}")


if __name__ == "__main__":
    main()
