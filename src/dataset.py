"""
src/dataset.py
---------------
Dataset, augmentation pipeline và DataLoader factory cho bài toán Global Wheat
Detection (object detection, 1 class duy nhất: "wheat_head").

Luồng dữ liệu tổng quan:

    train.csv, cột "bbox" = "[x, y, w, h]" (string, format COCO)
            │
            ▼  parse_bbox_string()                  [src/utils.py]
    [x, y, w, h] (float)
            │
            ▼  _convert_and_clip_boxes()             (hàm dưới đây)
    [x_min, y_min, x_max, y_max] (Pascal VOC, đã clip về biên ảnh, đã loại box rỗng)
            │
            ▼  WheatDataset.__getitem__()
    ảnh RGB (cv2) + boxes + labels  --(Albumentations)-->  tensor đã augment/normalize
            │
            ▼  collate_fn()
    batch dạng tuple (vì số lượng box mỗi ảnh khác nhau, không stack được)
            │
            ▼  get_train_dataloader() / get_val_dataloader()
    torch.utils.data.DataLoader sẵn sàng đưa vào vòng lặp train/validate.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.utils import parse_bbox_string

# ─── Kiểu dữ liệu dùng lại nhiều nơi trong module ───────────────────────────
TargetDict = Dict[str, torch.Tensor]
Sample = Tuple[torch.Tensor, TargetDict]
Batch = Tuple[Tuple[torch.Tensor, ...], Tuple[TargetDict, ...]]

# Mean/std của ImageNet: backbone ResNet50-FPN được pretrain trên ImageNet, nên ảnh
# đầu vào phải được chuẩn hoá theo đúng thống kê này thì transfer learning mới hiệu quả.
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _convert_and_clip_boxes(
    bbox_strings: Sequence[str], img_width: int, img_height: int
) -> np.ndarray:
    """
    Chuyển danh sách chuỗi bbox COCO "[x, y, w, h]" của MỘT ảnh sang mảng Pascal VOC
    [x_min, y_min, x_max, y_max], đồng thời làm sạch dữ liệu:

        1. Clip toạ độ về đúng trong khoảng [0, img_width] x [0, img_height] — một số
           bbox gốc trong train.csv của cuộc thi này có toạ độ vượt nhẹ ra ngoài biên ảnh.
        2. Loại bỏ mọi box có diện tích <= 0 SAU KHI clip (box bị clip suy biến thành
           1 điểm hoặc 1 đường thẳng) — box như vậy sẽ khiến Faster R-CNN lỗi hoặc học sai.

    Args:
        bbox_strings: danh sách chuỗi bbox thô, mỗi phần tử dạng "[x, y, w, h]".
        img_width: chiều rộng ảnh gốc (pixel), dùng để clip.
        img_height: chiều cao ảnh gốc (pixel), dùng để clip.

    Returns:
        np.ndarray, shape (N, 4), dtype float32, N <= len(bbox_strings) (có thể ít hơn
        nếu có box bị loại bỏ). Trả về mảng rỗng shape (0, 4) nếu không còn box nào hợp lệ.
    """
    valid_boxes: List[List[float]] = []

    for bbox_str in bbox_strings:
        x, y, w, h = parse_bbox_string(bbox_str)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Clip cứng về biên ảnh [0, width] / [0, height].
        x_min = float(np.clip(x_min, 0, img_width))
        y_min = float(np.clip(y_min, 0, img_height))
        x_max = float(np.clip(x_max, 0, img_width))
        y_max = float(np.clip(y_max, 0, img_height))

        # Bỏ box suy biến (diện tích <= 0) sau khi clip.
        if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
            continue

        valid_boxes.append([x_min, y_min, x_max, y_max])

    if len(valid_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(valid_boxes, dtype=np.float32)


def get_train_transforms(image_size: int) -> A.Compose:
    """
    Augmentation pipeline cho tập TRAIN.

    Có random flip/brightness-contrast để tăng tính tổng quát hoá, giảm nguy cơ model
    học thuộc hướng chụp/điều kiện ánh sáng cụ thể của tập train (overfitting).

    Args:
        image_size: cạnh ảnh sau khi resize (ảnh vuông image_size x image_size).

    Returns:
        `albumentations.Compose` áp dụng được đồng thời lên cả ảnh và bbox.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),  # chuyển ảnh numpy (H, W, C) -> tensor PyTorch (C, H, W)
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],  # để Albumentations biết loại/giữ label đồng bộ với bbox bị loại/giữ
            min_area=0,
            min_visibility=0.1,  # bbox còn lại <10% diện tích gốc sau augment sẽ bị loại
        ),
    )


def get_valid_transforms(image_size: int) -> A.Compose:
    """
    Augmentation pipeline cho tập VALIDATION: KHÔNG random augment (không flip,
    không đổi độ sáng), chỉ resize + chuẩn hoá — để đánh giá model trên ảnh "sạch",
    nhất quán giữa các epoch và giống điều kiện lúc inference thật.

    Args:
        image_size: cạnh ảnh sau khi resize (ảnh vuông image_size x image_size).

    Returns:
        `albumentations.Compose` áp dụng được đồng thời lên cả ảnh và bbox.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class WheatDataset(Dataset):
    """
    PyTorch Dataset cho cuộc thi Global Wheat Detection.

    Mỗi sample tương ứng với MỘT ảnh và TOÀN BỘ bbox "wheat_head" trong ảnh đó.
    Đây là bài toán detection 1 class: label 0 dành cho background (ngầm định,
    không xuất hiện tường minh trong `labels`), label 1 là wheat_head.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
    ) -> None:
        """
        Args:
            df: DataFrame của train.csv (các cột image_id, width, height, bbox,
                source), đã được lọc sẵn theo fold train/val ở tầng train.py.
            image_dir: thư mục chứa các file ảnh "<image_id>.jpg".
            transforms: augmentation pipeline, dùng `get_train_transforms` hoặc
                `get_valid_transforms`. `None` nếu muốn lấy ảnh gốc không augment.
        """
        super().__init__()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        # 1 ảnh có thể ứng với NHIỀU dòng trong CSV (mỗi dòng = 1 bbox), nên độ dài
        # dataset phải tính theo số image_id DUY NHẤT, không phải số dòng CSV.
        self.image_ids: np.ndarray = df["image_id"].unique()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Sample:
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh tại: {image_path}")
        # OpenCV đọc ảnh theo thứ tự kênh màu BGR, nhưng model/Albumentations kỳ vọng
        # RGB -> thiếu bước đổi màu này ảnh sẽ bị lệch màu khi train.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        records = self.df[self.df["image_id"] == image_id]
        boxes = _convert_and_clip_boxes(records["bbox"].values, img_width, img_height)

        if len(boxes) == 0:
            # Negative sample: ảnh không có (hoặc không còn) bbox hợp lệ nào sau khi clip.
            labels = np.empty((0,), dtype=np.int64)
        else:
            # 1: wheat_head (0 dành cho background, không xuất hiện trong mảng labels).
            labels = np.ones((len(boxes),), dtype=np.int64)

        if self.transforms is not None:
            # Albumentations nhận numpy array, không nhận tensor -> phải augment
            # TRƯỚC khi ToTensorV2 chuyển sang tensor (đã nằm trong pipeline transforms).
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.array(transformed["labels"], dtype=np.int64)

        target: TargetDict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch: List[Sample]) -> Batch:
    """
    Gom một batch cho bài toán detection.

    DataLoader mặc định cố "stack" các sample thành 1 tensor lớn duy nhất, nhưng mỗi
    ảnh trong batch có SỐ LƯỢNG bbox khác nhau (target["boxes"] không đồng đều kích
    thước) nên không thể stack. Trả về tuple(images), tuple(targets) là pattern chuẩn
    dùng cho hầu hết các bài toán object detection trong PyTorch.

    Args:
        batch: list các sample (image, target) do `WheatDataset.__getitem__` trả về.

    Returns:
        Tuple gồm 2 phần tử: tuple các ảnh, và tuple các target dict tương ứng.
    """
    return tuple(zip(*batch))


def get_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Tạo DataLoader cho tập TRAIN.

    Args:
        dataset: một `WheatDataset` đã được khởi tạo với `get_train_transforms`.
        batch_size: số ảnh mỗi batch.
        num_workers: số tiến trình con dùng để load dữ liệu song song (0 = load ở
            main process, an toàn khi debug trên Windows/Jupyter).
        shuffle: có xáo trộn thứ tự ảnh mỗi epoch hay không (nên để True khi train,
            tránh model học theo thứ tự dữ liệu cố định).
        pin_memory: ghim bộ nhớ RAM để copy tensor sang GPU nhanh hơn — chỉ có tác
            dụng khi train trên GPU (CUDA).

    Returns:
        `torch.utils.data.DataLoader` đã gắn sẵn `collate_fn` phù hợp cho detection.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,  # tránh batch cuối chỉ có 1 ảnh gây lỗi ở 1 số layer BatchNorm
    )


def get_val_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Tạo DataLoader cho tập VALIDATION.

    Args:
        dataset: một `WheatDataset` đã được khởi tạo với `get_valid_transforms`.
        batch_size: số ảnh mỗi batch.
        num_workers: số tiến trình con dùng để load dữ liệu song song.
        shuffle: mặc định False vì thứ tự không quan trọng khi đánh giá, và cần
            nhất quán giữa các epoch để log/so sánh kết quả dễ dàng.
        pin_memory: ghim bộ nhớ RAM để copy tensor sang GPU nhanh hơn.

    Returns:
        `torch.utils.data.DataLoader` đã gắn sẵn `collate_fn` phù hợp cho detection.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,  # tập val cần giữ đủ toàn bộ ảnh để đánh giá chính xác
    )
