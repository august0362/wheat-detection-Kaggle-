import os
import ast
import torch
import yaml
import numpy as np

# Đọc file config YAML (local_config.yaml / kaggle_config.yaml) thành dict Python
# -> train.py chỉ cần đổi --config để chạy local hay Kaggle mà không phải sửa code.
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_bbox_string(bbox_str: str) -> list:
    """
    Kaggle string: "[x, y, w, h]" -> Pascal VOC: [x_min, y_min, x_max, y_max]
    """
    # Cột "bbox" trong train.csv là chuỗi text, vd "[834.0, 222.0, 56.0, 36.0]"
    # ast.literal_eval an toàn hơn eval() vì chỉ parse literal Python, không chạy code tuỳ ý.
    x, y, w, h = ast.literal_eval(bbox_str)
    # Kaggle lưu bbox dạng COCO (x, y, width, height).
    # torchvision detection models cần dạng Pascal VOC (x_min, y_min, x_max, y_max)
    # nên phải cộng thêm w, h để ra toạ độ góc dưới-phải. Quên bước này là lỗi rất hay gặp.
    return [float(x), float(y), float(x + w), float(y + h)]

# Lưu checkpoint model (dùng khi train_loss cải thiện, xem train.py).
def save_checkpoint(state: dict, filepath: str):
    # exist_ok=True để không lỗi nếu thư mục outputs/ đã tồn tại; tự tạo nếu chưa có.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[INFO] Saved checkpoint to: {filepath}")

class AverageMeter:
    """Theo dõi giá trị trung bình của Loss trong từng epoch"""
    # Thay vì tự cộng dồn loss + đếm batch thủ công trong vòng lặp train,
    # gom logic đó vào 1 class dùng lại được -> code train_one_epoch() gọn hơn.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0    # giá trị của batch gần nhất
        self.avg = 0.0    # trung bình cộng dồn từ đầu tới giờ
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n   # n = số sample trong batch, để trung bình đúng khi batch cuối lẻ
        self.count += n
        self.avg = self.sum / self.count