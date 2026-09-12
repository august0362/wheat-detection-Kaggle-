"""
src/utils.py
------------
Các hàm và class tiện ích dùng chung cho toàn bộ pipeline huấn luyện:
    - Đọc file cấu hình YAML.
    - Thiết lập seed cố định để đảm bảo khả năng tái lập kết quả (reproducibility).
    - Parse chuỗi bbox thô từ train.csv.
    - Theo dõi (log) các thành phần loss trong quá trình train/validate.
    - Lưu checkpoint model.

Module này KHÔNG phụ thuộc vào src/dataset.py hay src/model.py để tránh import vòng
(circular import) — mọi module khác trong dự án đều có thể import utils một cách an toàn.
"""
from __future__ import annotations

import ast
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Đọc file cấu hình YAML (vd: configs/kaggle_config.yaml) và trả về dict Python.

    Args:
        config_path: đường dẫn tới file .yaml.

    Returns:
        Dict chứa toàn bộ key-value khai báo trong file YAML.

    Raises:
        FileNotFoundError: nếu config_path không tồn tại.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """
    Cố định seed cho `random`, NumPy và PyTorch (cả CPU lẫn toàn bộ GPU) để hai lần
    chạy training với cùng config cho ra kết quả giống nhau — rất quan trọng khi so
    sánh hiệu quả giữa các thay đổi (đổi augmentation, đổi learning rate, ...).

    Lưu ý đánh đổi: `cudnn.deterministic = True` buộc cuDNN dùng thuật toán tất định
    thay vì thuật toán nhanh nhất được benchmark tự động -> training có thể chậm hơn
    một chút, nhưng đổi lại kết quả ổn định, tái lập được.

    Args:
        seed: giá trị seed, mặc định 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_bbox_string(bbox_str: str) -> List[float]:
    """
    Parse 1 chuỗi bbox thô trong cột "bbox" của train.csv.

    Kaggle lưu bbox dạng COCO dưới dạng TEXT, ví dụ: "[834.0, 222.0, 56.0, 36.0]"
    tương ứng (x_min, y_min, width, height).

    Hàm này CHỈ làm nhiệm vụ parse string -> list số thực [x, y, w, h], KHÔNG đổi
    sang Pascal VOC và KHÔNG clip toạ độ — hai bước đó cần biết kích thước ảnh thật
    nên được xử lý riêng trong `src/dataset.py` (xem `_convert_and_clip_boxes`).

    Args:
        bbox_str: chuỗi bbox, vd "[834.0, 222.0, 56.0, 36.0]".

    Returns:
        List 4 phần tử float: [x, y, width, height].
    """
    # ast.literal_eval an toàn hơn eval() vì chỉ parse literal Python (list/số/…),
    # không thực thi mã tuỳ ý -> tránh rủi ro command injection nếu dữ liệu bị can thiệp.
    x, y, w, h = ast.literal_eval(bbox_str)
    return [float(x), float(y), float(w), float(h)]


def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """
    Lưu checkpoint (state_dict của model/optimizer + metadata như epoch, loss) ra
    file .pth. Tự động tạo thư mục cha nếu chưa tồn tại.

    Args:
        state: dict chứa nội dung cần lưu, vd:
            {"epoch": 3, "model_state": model.state_dict(), "val_loss": 0.42}.
        filepath: đường dẫn file .pth đích.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(state, filepath)
    print(f"[INFO] Đã lưu checkpoint: {filepath}")


class AverageMeter:
    """
    Theo dõi giá trị hiện tại (`val`) và giá trị trung bình cộng dồn (`avg`) của
    MỘT đại lượng vô hướng duy nhất (vd: 1 loss, 1 metric) qua nhiều batch/bước.

    Dùng `n` (số sample trong batch) khi update để trung bình vẫn đúng ngay cả khi
    batch cuối cùng có kích thước nhỏ hơn các batch trước (batch lẻ).
    """

    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.reset()

    def reset(self) -> None:
        """Đặt lại toàn bộ số liệu về 0 — gọi vào đầu mỗi epoch mới."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Cập nhật với giá trị mới.

        Args:
            val: giá trị đo được ở bước hiện tại (vd: loss trung bình của 1 batch).
            n: số sample đóng góp vào giá trị `val` (thường = batch_size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class LossMeter:
    """
    Theo dõi ĐỒNG THỜI nhiều thành phần loss của Faster R-CNN trong 1 epoch.

    Khi ở train mode, Faster R-CNN trả về 1 dict gồm 4 loss thành phần:
        - loss_classifier:    loss phân loại object (wheat_head vs background)
        - loss_box_reg:       loss hồi quy toạ độ box (Smooth L1) ở ROI head
        - loss_objectness:    loss "có vật thể hay không" ở Region Proposal Network
        - loss_rpn_box_reg:   loss hồi quy anchor box ở RPN

    Class này tự tạo 1 `AverageMeter` riêng cho từng key xuất hiện trong loss_dict
    (không cần khai báo trước tên loss), đồng thời cộng dồn thêm 1 khoá "total_loss"
    là tổng của tất cả thành phần — đúng bằng giá trị dùng để `backward()`.
    """

    def __init__(self) -> None:
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, loss_dict: Dict[str, torch.Tensor], batch_size: int) -> None:
        """
        Cập nhật các meter từ loss_dict trả về bởi model ở 1 batch.

        Args:
            loss_dict: dict {tên_loss: tensor scalar}, lấy trực tiếp từ
                `model(images, targets)` khi model đang ở train mode.
            batch_size: số ảnh trong batch (dùng làm trọng số `n` khi trung bình).
        """
        total = 0.0
        for name, value in loss_dict.items():
            loss_value = float(value.detach().item())
            total += loss_value
            self._meters.setdefault(name, AverageMeter()).update(loss_value, n=batch_size)

        self._meters.setdefault("total_loss", AverageMeter()).update(total, n=batch_size)

    @property
    def averages(self) -> Dict[str, float]:
        """Trả về dict {tên_loss: giá trị trung bình cộng dồn tới thời điểm hiện tại}."""
        return {name: meter.avg for name, meter in self._meters.items()}

    def __str__(self) -> str:
        return " | ".join(f"{name}: {avg:.4f}" for name, avg in self.averages.items())
