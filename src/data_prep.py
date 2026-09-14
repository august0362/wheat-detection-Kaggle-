"""
src/data_prep.py
------------------
Chuẩn bị dữ liệu chuẩn YOLOv8 từ file train.csv gốc của cuộc thi Global Wheat
Detection.

Các bước xử lý:
    1. build_image_metadata()    gom danh sách ảnh, kể cả ảnh "nền" (không có box
                                  nào trong train.csv), để không bị bỏ sót khi chia tập.
    2. stratified_image_split()  chia train/val (80/20), mỗi ảnh chỉ thuộc 1 tập
                                  duy nhất (tránh 1 ảnh vừa ở train vừa ở val).
    3. build_yolo_dataset()      convert bbox (dùng src/utils.py), ghi file nhãn
                                  .txt, copy ảnh, xuất file data.yaml.

Chạy độc lập (không cần train.py):
    python -m src.data_prep --config configs/kaggle_config.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.utils import coco_to_yolo, load_config, parse_bbox_string, set_seed

# Nhãn dùng cho ảnh có trong thư mục nhưng không có box nào trong train.csv (ảnh nền)
NO_ANNOTATION_LABEL = "__no_annotation__"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _list_image_ids(image_dir: str) -> List[str]:
    """Lấy danh sách image_id (tên file, không đuôi) có thật trong `image_dir`."""
    return sorted(
        p.stem for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def build_image_metadata(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    Tạo bảng (image_id, source), mỗi ảnh 1 dòng, dùng để chia train/val.

    Ảnh nền (có trong `image_dir` nhưng không có dòng nào trong `df`) cũng được
    thêm vào với nhãn `NO_ANNOTATION_LABEL`, để không bị bỏ sót khi chia tập.
    """
    annotated = df.groupby("image_id")["source"].first().reset_index()
    annotated_ids = set(annotated["image_id"])

    all_ids = _list_image_ids(image_dir)
    background_ids = [i for i in all_ids if i not in annotated_ids]
    if background_ids:
        background_df = pd.DataFrame(
            {"image_id": background_ids, "source": NO_ANNOTATION_LABEL}
        )
        annotated = pd.concat([annotated, background_df], ignore_index=True)
        print(f"[INFO] {len(background_ids)} ảnh không có annotation (ảnh nền) được thêm vào tập dữ liệu.")

    missing_on_disk = annotated_ids - set(all_ids)
    if missing_on_disk:
        print(f"[WARN] {len(missing_on_disk)} image_id trong train.csv không tìm thấy file ảnh trên đĩa, sẽ bị loại.")
        annotated = annotated[~annotated["image_id"].isin(missing_on_disk)].reset_index(drop=True)

    return annotated


def stratified_image_split(
    image_meta: pd.DataFrame, val_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chia train/val (mặc định 80/20) theo cột 'source', mỗi image_id chỉ rơi vào
    đúng 1 tập — tránh rò rỉ dữ liệu khi các box của cùng 1 ảnh bị xé lẻ ra 2 tập.
    """
    train_ids, val_ids = train_test_split(
        image_meta["image_id"].values,
        test_size=val_size,
        random_state=seed,
        stratify=image_meta["source"].values,
        shuffle=True,
    )
    return train_ids, val_ids


def _print_source_distribution(image_meta: pd.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray) -> None:
    meta_indexed = image_meta.set_index("image_id")
    train_dist = meta_indexed.loc[train_ids, "source"].value_counts(normalize=True).round(3)
    val_dist = meta_indexed.loc[val_ids, "source"].value_counts(normalize=True).round(3)
    dist_table = pd.DataFrame({"train": train_dist, "val": val_dist}).fillna(0.0)
    print("[INFO] Phân phối 'source' (tỉ lệ) — Train vs Val (càng gần nhau càng tốt):")
    print(dist_table.to_string())


def _write_yolo_label(label_path: str, rows: List[Tuple[int, float, float, float, float]]) -> None:
    """
    Ghi 1 file .txt nhãn YOLO: mỗi dòng là "class_id x_center y_center width height"
    (normalized [0, 1]). Ảnh không có box nào thì vẫn ghi file, chỉ để rỗng — đây là
    quy ước của YOLO cho ảnh nền, giúp model học cả trường hợp ảnh không có vật thể.
    """
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in rows:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _place_image(src_path: str, dst_path: str, copy_images: bool) -> None:
    if os.path.exists(dst_path):
        return
    if copy_images:
        shutil.copy2(src_path, dst_path)
    else:
        os.symlink(os.path.abspath(src_path), dst_path)


def build_yolo_dataset(cfg: Dict) -> str:
    """
    Chạy toàn bộ bước chuẩn bị dữ liệu: convert bbox, chia train/val, tạo thư mục
    ảnh/nhãn chuẩn YOLOv8, xuất file `data.yaml`.

    Nếu `<yolo_dataset_dir>/data.yaml` đã tồn tại thì bỏ qua, dùng lại luôn — nhờ
    vậy `src/train.py` có thể gọi hàm này mỗi lần train mà không build lại từ đầu.

    Cấu trúc thư mục được tạo ra:
        <yolo_dataset_dir>/
        ├── images/{train,val}/<image_id>.jpg
        ├── labels/{train,val}/<image_id>.txt
        └── data.yaml

    Returns:
        Đường dẫn tới file `data.yaml`.
    """
    data_cfg = cfg["data"]
    out_dir = Path(data_cfg["yolo_dataset_dir"])
    data_yaml_path = out_dir / "data.yaml"

    if data_yaml_path.exists():
        print(f"[INFO] Dataset YOLO đã tồn tại tại {out_dir}, bỏ qua bước chuẩn bị dữ liệu.")
        return str(data_yaml_path)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    df = pd.read_csv(data_cfg["csv_path"])
    image_dir = data_cfg["train_image_dir"]

    image_meta = build_image_metadata(df, image_dir)
    train_ids, val_ids = stratified_image_split(
        image_meta, val_size=data_cfg.get("val_size", 0.2), seed=seed
    )
    print(f"[INFO] Tổng {len(image_meta)} ảnh -> Train: {len(train_ids)} ảnh | Val: {len(val_ids)} ảnh")
    _print_source_distribution(image_meta, train_ids, val_ids)

    boxes_by_image = {
        image_id: group["bbox"].tolist() for image_id, group in df.groupby("image_id")
    }
    dims_by_image = df.drop_duplicates("image_id").set_index("image_id")[["width", "height"]]

    copy_images = bool(data_cfg.get("copy_images", True))
    n_dropped_boxes = 0
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        img_out_dir = out_dir / "images" / split_name
        lbl_out_dir = out_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for image_id in ids:
            src_img = os.path.join(image_dir, f"{image_id}.jpg")
            _place_image(src_img, str(img_out_dir / f"{image_id}.jpg"), copy_images)

            rows: List[Tuple[int, float, float, float, float]] = []
            if image_id in boxes_by_image:
                img_w = int(dims_by_image.loc[image_id, "width"])
                img_h = int(dims_by_image.loc[image_id, "height"])
                for bbox_str in boxes_by_image[image_id]:
                    yolo_box = coco_to_yolo(parse_bbox_string(bbox_str), img_w, img_h)
                    if yolo_box is not None:
                        rows.append((0, *yolo_box))  # class_id=0: "wheat" (chỉ 1 lớp duy nhất)
                    else:
                        n_dropped_boxes += 1
            _write_yolo_label(str(lbl_out_dir / f"{image_id}.txt"), rows)

    if n_dropped_boxes:
        print(f"[WARN] {n_dropped_boxes} bbox bị loại bỏ vì suy biến/nằm ngoài biên ảnh sau khi clip.")

    class_name = data_cfg.get("class_name", "wheat")
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [class_name],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"[INFO] Đã xuất data.yaml: {data_yaml_path}")

    return str(data_yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu YOLOv8 cho Global Wheat Detection")
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml_path = build_yolo_dataset(cfg)
    print(f"[DONE] data.yaml: {data_yaml_path}")


if __name__ == "__main__":
    main()
