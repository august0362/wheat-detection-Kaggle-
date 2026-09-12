import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.utils import parse_bbox_string

class WheatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms=None):
        super().__init__()
        self.image_dir = image_dir
        self.df = df
        self.transforms = transforms
        # 1 ảnh có thể có NHIỀU dòng trong CSV (mỗi dòng = 1 bbox),
        # nên độ dài dataset phải tính theo số image_id duy nhất, không phải số dòng CSV.
        self.image_ids = self.df["image_id"].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        # OpenCV đọc ảnh theo thứ tự kênh màu BGR, nhưng model/Albumentations kỳ vọng RGB
        # -> thiếu bước đổi màu này ảnh sẽ bị "lệch màu" khi train.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lọc tất cả các dòng (bbox) thuộc về đúng ảnh này.
        records = self.df[self.df["image_id"] == image_id]
        boxes = []
        for bbox_str in records["bbox"].values:
            boxes.append(parse_bbox_string(bbox_str))

        boxes = np.array(boxes, dtype=np.float32)

        # Xử lý negative sample (ảnh không có bông lúa)
        # Lưu ý: train.csv gốc của cuộc thi thường không có ảnh negative,
        # nhánh này chỉ thực sự chạy nếu tự bổ sung thêm ảnh không có bbox.
        if len(boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        else:
            # 1: wheat head (0 được dành cho background)
            labels = np.ones((len(boxes),), dtype=np.int64)

        if self.transforms:
            # Albumentations nhận numpy array, không nhận tensor -> gọi transform TRƯỚC khi convert sang tensor.
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # torchvision detection model yêu cầu target là dict đúng các key này (boxes, labels, image_id).
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target

# DataLoader mặc định sẽ cố "stack" các sample thành 1 tensor lớn, nhưng ở đây
# mỗi ảnh có SỐ LƯỢNG bbox khác nhau (target["boxes"] không đồng đều) nên không stack được.
# collate_fn trả về tuple(images), tuple(targets) thay vì stack -> đây là cách làm chuẩn cho mọi bài toán detection.
def collate_fn(batch):
    return tuple(zip(*batch))