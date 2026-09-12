"""
src/data_prep.py
------------------
Thành phần 1 của pipeline: chuẩn bị dữ liệu chuẩn YOLOv8 từ train.csv gốc của
cuộc thi Global Wheat Detection.

    train.csv (image_id, width, height, bbox="[x,y,w,h]" COCO, source)
            │
            ▼  build_image_metadata()      gộp cả ảnh "nền" (có file ảnh, KHÔNG có
            │                              dòng nào trong train.csv) vào nhóm riêng
            │                              để không bị bỏ sót khi chia tập.
            ▼  stratified_image_split()    train_test_split 80/20, stratify theo
            │                              cột 'source', chia Ở CẤP ĐỘ ẢNH (tránh
            │                              1 ảnh có bbox vừa lọt train vừa lọt val).
            ▼  build_yolo_dataset()        parse+clip+convert bbox (src/utils.py),
                                            ghi label .txt chuẩn YOLO, copy/symlink
                                            ảnh, xuất data.yaml.

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

# Nhãn stratify riêng cho ảnh có thật trong thư mục ảnh nhưng KHÔNG có bbox nào
# trong train.csv (ảnh "nền" — background, không có wheat head). Bộ dữ liệu Global
# Wheat Detection có 49 ảnh như vậy (3422 ảnh trong thư mục, chỉ 3373 ảnh có annotation).
NO_ANNOTATION_LABEL = "__no_annotation__"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _list_image_ids(image_dir: str) -> List[str]:
    """Toàn bộ image_id (tên file, không đuôi) thực sự tồn tại trên đĩa trong `image_dir`."""
    return sorted(
        p.stem for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def build_image_metadata(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    Bảng metadata 1-dòng-1-ảnh (image_id, source) dùng để chia Stratified Split.

    Gộp cả các ảnh "nền" (tồn tại trong `image_dir` nhưng không có dòng nào trong
    `df`) vào nhóm `NO_ANNOTATION_LABEL`, để chúng cũng được phân bổ đồng đều giữa
    train/val thay vì bị bỏ sót hoàn toàn khỏi quá trình huấn luyện.
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
    Stratified Split (mặc định 80/20) theo cột 'source', chia Ở CẤP ĐỘ ẢNH DUY NHẤT
    (mỗi image_id chỉ rơi vào đúng 1 trong 2 tập) — tránh data leakage khi 1 ảnh có
    nhiều bbox bị xé lẻ vừa vào train vừa vào val.
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
    Ghi 1 file .txt nhãn YOLO: mỗi dòng "class_id x_center y_center width height"
    (normalized [0, 1]). Ảnh không có box hợp lệ nào -> ghi file RỖNG (KHÔNG bỏ qua
    việc tạo file) — đây là quy ước chuẩn của YOLO cho "ảnh nền" (background image),
    giúp model học cả trường hợp không có vật thể nào trong ảnh.
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
    Thực thi toàn bộ thành phần 1: parse+convert bbox, stratified split, tạo cấu
    trúc thư mục chuẩn YOLOv8, xuất `data.yaml`.

    Idempotent: nếu `<yolo_dataset_dir>/data.yaml` đã tồn tại, bỏ qua và tái sử
    dụng — cho phép `src/train.py` gọi lại hàm này ở đầu mỗi lần train mà không
    build lại từ đầu mỗi lần.

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
