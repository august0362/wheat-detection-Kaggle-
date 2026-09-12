"""
src/augmentations.py
---------------------
Thành phần 2 của pipeline: augmentation ON-THE-FLY (áp dụng ngay lúc nạp dữ liệu,
không augment offline ra file mới) bằng Albumentations, gắn thẳng vào vòng lặp
training của Ultralytics YOLOv8.

Vì sao cần "patch" thay vì chỉ truyền tham số cho `model.train()`:
Ultralytics đã có sẵn 1 bước Albumentations trong pipeline augment nội bộ
(`ultralytics.data.augment.Albumentations`), nhưng bộ transform MẶC ĐỊNH của nó chỉ
gồm vài phép biến đổi nhẹ (Blur/MedianBlur/ToGray/CLAHE, p=0.01) và quan trọng nhất:
KHÔNG cho tuỳ chỉnh `min_visibility` của `BboxParams` (competition yêu cầu 0.3) qua
bất kỳ tham số công khai nào của `model.train()` — `get_cfg()` của Ultralytics còn
chặn cứng mọi key lạ trong `train(**kwargs)` (xem `check_dict_alignment`), nên
không thể "luồn" transform tuỳ biến qua đường train() thông thường.

Giải pháp: monkey-patch class `ultralytics.data.augment.Albumentations` bằng một
class tự viết CÙNG INTERFACE (`__init__(self, p=1.0, *a, **kw)`, `__call__(self,
labels) -> labels`) TRƯỚC khi gọi `model.train()`. Vì `v8_transforms()` (nơi khởi
tạo `Albumentations(...)`) tra cứu tên `Albumentations` qua namespace module tại
THỜI ĐIỂM CHẠY (không cache import), việc gán lại `ultralytics.data.augment.
Albumentations = <class của mình>` có hiệu lực ngay cả khi việc gán diễn ra sau khi
`ultralytics.data.augment` đã được import.

Trick này phụ thuộc cấu trúc nội bộ (không phải API công khai chính thức) của
Ultralytics nên được bọc trong try/except: nếu bản Ultralytics tương lai đổi
interface, pipeline in cảnh báo và rơi về augmentation mặc định thay vì crash toàn
bộ quá trình train.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import albumentations as A
import numpy as np


def build_bbox_transform(
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    brightness_contrast_p: float = 0.3,
    hue_sat_value_p: float = 0.3,
    blur_p: float = 0.2,
    shift_scale_rotate_p: float = 0.3,
    min_visibility: float = 0.3,
) -> A.Compose:
    """
    Pipeline augmentation dùng chung cho cả bước patch-vào-Ultralytics lẫn script
    xem trước augmentation độc lập (`python -m src.augmentations`).

    `bbox_params.format="yolo"`: tại thời điểm Ultralytics gọi bước Albumentations,
    box đã ở dạng [x_center, y_center, w, h] normalized [0, 1] — đúng chuẩn YOLO.
    `min_visibility=0.3`: box còn lại <30% diện tích gốc sau augment (bị crop/xoay
    mất phần lớn) sẽ bị loại thẳng, tránh model học từ box gần như rỗng.
    `clip=True`: tự động clip toạ độ box về [0, 1] nếu ShiftScaleRotate làm box
    vượt nhẹ ra biên do sai số làm tròn, tránh Albumentations báo lỗi box không hợp lệ.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=hflip_p),
            A.VerticalFlip(p=vflip_p),
            A.RandomBrightnessContrast(p=brightness_contrast_p),
            A.HueSaturationValue(p=hue_sat_value_p),
            A.OneOf(
                [A.Blur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=(3, 7), p=1.0)],
                p=blur_p,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=10, p=shift_scale_rotate_p
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=min_visibility,
            clip=True,
        ),
    )


def build_preview_transform(image_size: int, **transform_kwargs: Any) -> A.Compose:
    """Bản có thêm `Resize` ở đầu, chỉ dùng để xem trước augmentation trên ảnh gốc (xem `main()` dưới)."""
    base = build_bbox_transform(**transform_kwargs)
    return A.Compose(
        [A.Resize(height=image_size, width=image_size), *base.transforms],
        bbox_params=base.processors["bboxes"].params,
    )


def patch_ultralytics_albumentations(**transform_kwargs: Any) -> bool:
    """
    Thay thế `ultralytics.data.augment.Albumentations` bằng bản tự viết dùng
    `build_bbox_transform(**transform_kwargs)`. PHẢI gọi hàm này TRƯỚC
    `YOLO(...).train(...)`.

    Args:
        **transform_kwargs: truyền thẳng vào `build_bbox_transform` (hflip_p,
            vflip_p, brightness_contrast_p, hue_sat_value_p, blur_p,
            shift_scale_rotate_p, min_visibility).

    Returns:
        True nếu patch thành công. False nếu thất bại (đã in cảnh báo) — training
        vẫn chạy tiếp được, chỉ là dùng augmentation mặc định của Ultralytics.
    """
    try:
        import ultralytics.data.augment as ultra_augment

        transform = build_bbox_transform(**transform_kwargs)

        class _CustomAlbumentations:
            """
            Drop-in thay thế cho `ultralytics.data.augment.Albumentations`. Global
            Wheat Detection là bài toán detection THUẦN BBOX (không segment/keypoint)
            nên class này chỉ cần xử lý box, đơn giản hơn nhiều so với bản gốc.
            """

            def __init__(self, p: float = 1.0, *_args: Any, **_kwargs: Any) -> None:
                # *_args/**_kwargs: nuốt mọi tham số Ultralytics truyền vào (vd `transforms`,
                # `flip_idx` ở các bản mới) để tương thích ngược/xuôi giữa các phiên bản.
                self.p = p
                self.transform = transform

            def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
                cls = labels["cls"]
                if len(cls) == 0 or random.random() >= self.p:
                    return labels

                instances = labels["instances"]
                instances.convert_bbox(format="xywh")
                instances.normalize(*labels["img"].shape[:2][::-1])
                bboxes = instances.bboxes

                new = self.transform(image=labels["img"], bboxes=bboxes, class_labels=cls.reshape(-1))
                if len(new["class_labels"]) == 0:
                    return labels  # augment loại hết box hợp lệ -> giữ nguyên ảnh gốc, không augment lần này

                labels["img"] = new["image"]
                labels["cls"] = np.array(new["class_labels"]).reshape(-1, 1)
                instances.update(bboxes=np.array(new["bboxes"], dtype=np.float32))
                return labels

        ultra_augment.Albumentations = _CustomAlbumentations
        print(
            "[INFO] Đã gắn augmentation pipeline Albumentations tuỳ biến "
            f"(min_visibility={transform_kwargs.get('min_visibility', 0.3)}) vào Ultralytics YOLOv8."
        )
        return True
    except Exception as e:  # pragma: no cover - phụ thuộc phiên bản Ultralytics đang cài
        print(f"[WARN] Không patch được Albumentations của Ultralytics ({e}); dùng augmentation mặc định thay thế.")
        return False


def main() -> None:
    """`python -m src.augmentations --config configs/local_config.yaml [--n 8]`
    Xem trước augmentation trên vài ảnh train thật, lưu thành 1 lưới ảnh — kiểm tra
    độc lập thành phần 2 mà không cần chạy training."""
    import argparse
    import glob
    import os
    import random as _random

    import cv2
    import matplotlib.pyplot as plt

    from src.utils import coco_to_yolo, load_config, parse_bbox_string, yolo_to_xyxy
    import pandas as pd

    parser = argparse.ArgumentParser(description="Xem trước augmentation pipeline (Albumentations)")
    parser.add_argument("--config", type=str, default="configs/local_config.yaml")
    parser.add_argument("--n", type=int, default=6, help="Số ảnh minh hoạ")
    parser.add_argument("--out", type=str, default="augmentation_preview.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    df = pd.read_csv(data_cfg["csv_path"])
    image_dir = data_cfg["train_image_dir"]
    transform = build_preview_transform(
        image_size=cfg.get("train", {}).get("imgsz", 640),
        hflip_p=aug_cfg.get("hflip_p", 0.5),
        vflip_p=aug_cfg.get("vflip_p", 0.5),
        brightness_contrast_p=aug_cfg.get("brightness_contrast_p", 0.3),
        hue_sat_value_p=aug_cfg.get("hue_sat_value_p", 0.3),
        blur_p=aug_cfg.get("blur_p", 0.2),
        shift_scale_rotate_p=aug_cfg.get("shift_scale_rotate_p", 0.3),
        min_visibility=aug_cfg.get("min_visibility", 0.3),
    )

    image_ids = df["image_id"].unique().tolist()
    _random.Random(cfg.get("seed", 42)).shuffle(image_ids)
    image_ids = image_ids[: args.n]

    fig, axes = plt.subplots(1, len(image_ids), figsize=(4 * len(image_ids), 4))
    axes = np.atleast_1d(axes)
    for ax, image_id in zip(axes, image_ids):
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        rows = df[df["image_id"] == image_id]
        yolo_boxes = [
            coco_to_yolo(parse_bbox_string(b), img_w, img_h) for b in rows["bbox"].tolist()
        ]
        yolo_boxes = [b for b in yolo_boxes if b is not None]
        class_labels = [0] * len(yolo_boxes)

        augmented = transform(image=image, bboxes=yolo_boxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_h, aug_w = aug_img.shape[:2]

        for box in augmented["bboxes"]:
            x1, y1, x2, y2 = yolo_to_xyxy(*box, aug_w, aug_h)
            cv2.rectangle(aug_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        ax.imshow(aug_img)
        ax.set_title(image_id, fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[DONE] Đã lưu ảnh xem trước augmentation: {args.out}")


if __name__ == "__main__":
    main()
