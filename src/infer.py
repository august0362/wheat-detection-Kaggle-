"""
src/infer.py
-------------
Thành phần 5 của pipeline: script inference ĐỘC LẬP, chạy trên tập test THẬT của
Kaggle (ẩn lúc submit) để sinh file `submission.csv` đúng định dạng cuộc thi.

    python -m src.infer --config configs/kaggle_config.yaml --output submission.csv

Hoặc truyền tham số trực tiếp (không cần config):

    python -m src.infer \\
        --weights runs/wheat_kaggle_run/weights/best.pt \\
        --source /kaggle/input/competitions/global-wheat-detection/test \\
        --sample-submission /kaggle/input/competitions/global-wheat-detection/sample_submission.csv \\
        --output submission.csv --conf 0.3 --iou 0.5

Định dạng submission (mỗi ảnh 1 dòng, bắt buộc kể cả ảnh không có dự đoán nào):
    image_id,PredictionString
    ce4833752,1.0 0 0 50 50
    6ca7b2650,
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.utils import load_config


def _list_test_image_ids(source_dir: str, sample_submission: Optional[str]) -> List[str]:
    """
    Danh sách image_id CHÍNH THỨC cần dự đoán. Ưu tiên lấy theo `sample_submission.csv`
    (đúng bộ ảnh + đúng thứ tự Kaggle chấm điểm) nếu có; nếu không, quét trực tiếp
    thư mục ảnh test. Đảm bảo MỌI ảnh test đều có đúng 1 dòng trong submission.csv —
    thiếu dòng nào cũng khiến Kaggle từ chối/chấm sai submission.
    """
    if sample_submission and os.path.isfile(sample_submission):
        return pd.read_csv(sample_submission)["image_id"].tolist()

    image_paths = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    return [Path(p).stem for p in image_paths]


def format_prediction_string(boxes_xyxy: Sequence[Sequence[float]], scores: Sequence[float]) -> str:
    """
    "confidence x_min y_min width height" nối nhau bằng dấu cách, mỗi box 1 cụm,
    SẮP XẾP GIẢM DẦN theo confidence (đúng thứ tự chấm điểm: box confidence cao
    được xét match trước — xem `src/metrics/evaluator.py`). Toạ độ/kích thước làm
    tròn về số nguyên gần nhất (pixel), đúng ví dụ định dạng của cuộc thi.
    """
    if len(boxes_xyxy) == 0:
        return ""

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    parts = []
    for i in order:
        x1, y1, x2, y2 = boxes_xyxy[i]
        w = x2 - x1
        h = y2 - y1
        parts.append(f"{scores[i]:.4f} {x1:.0f} {y1:.0f} {w:.0f} {h:.0f}")
    return " ".join(parts)


def run_inference(
    weights_path: str,
    source_dir: str,
    output_csv: str,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    imgsz: int = 1024,
    device: Optional[str] = None,
    sample_submission: Optional[str] = None,
) -> str:
    """
    Load `weights_path` (best.pt), predict toàn bộ ảnh trong `source_dir`, và ghi
    `output_csv` đúng định dạng submission.

    NMS (loại box trùng lặp) + lọc theo ngưỡng confidence đã được `YOLO.predict()`
    áp dụng nội bộ thông qua 2 tham số `conf`/`iou` — không cần cài đặt NMS thủ công.
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)

    all_image_ids = _list_test_image_ids(source_dir, sample_submission)
    print(f"[INFO] Tổng {len(all_image_ids)} ảnh test cần dự đoán.")

    results = model.predict(
        source=source_dir, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz,
        device=device, verbose=False, stream=True,
    )

    predictions_by_id = {}
    n_images_predicted = 0
    for result in results:
        image_id = Path(result.path).stem
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []
        scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else []
        predictions_by_id[image_id] = format_prediction_string(boxes, scores)
        n_images_predicted += 1
    print(f"[INFO] Đã chạy inference trên {n_images_predicted} ảnh.")

    rows = []
    n_missing = 0
    for image_id in all_image_ids:
        if image_id not in predictions_by_id:
            n_missing += 1
        rows.append({"image_id": image_id, "PredictionString": predictions_by_id.get(image_id, "")})
    if n_missing:
        print(f"[WARN] {n_missing} image_id không tìm thấy file ảnh tương ứng trong {source_dir} -> PredictionString rỗng.")

    submission_df = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    submission_df.to_csv(output_csv, index=False)
    print(f"[DONE] Đã ghi submission: {output_csv} ({len(submission_df)} dòng).")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference + sinh submission.csv cho Global Wheat Detection")
    parser.add_argument("--config", type=str, default=None, help="Đọc tham số mặc định từ file config YAML.")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn best.pt (ghi đè config).")
    parser.add_argument("--source", type=str, default=None, help="Thư mục ảnh test (ghi đè config).")
    parser.add_argument("--sample-submission", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--conf", type=float, default=None, help="Ngưỡng confidence (ghi đè config).")
    parser.add_argument("--iou", type=float, default=None, help="Ngưỡng NMS IoU (ghi đè config).")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    infer_cfg = cfg.get("inference", {})
    data_cfg = cfg.get("data", {})

    weights_path = args.weights or infer_cfg.get("weights_path")
    source_dir = args.source or data_cfg.get("test_image_dir")
    sample_submission = args.sample_submission or data_cfg.get("sample_submission")
    conf = args.conf if args.conf is not None else infer_cfg.get("conf_threshold", 0.3)
    iou = args.iou if args.iou is not None else infer_cfg.get("iou_threshold", 0.5)
    imgsz = args.imgsz or infer_cfg.get("imgsz", 1024)
    device = args.device or infer_cfg.get("device")

    if not weights_path or not source_dir:
        raise ValueError(
            "Thiếu --weights/--source (hoặc thiếu inference.weights_path / data.test_image_dir trong --config)."
        )

    run_inference(
        weights_path=weights_path,
        source_dir=source_dir,
        output_csv=args.output,
        conf_threshold=conf,
        iou_threshold=iou,
        imgsz=imgsz,
        device=device,
        sample_submission=sample_submission,
    )


if __name__ == "__main__":
    main()
