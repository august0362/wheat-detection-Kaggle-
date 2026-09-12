import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import WheatDataset, collate_fn
from src.transforms import get_train_transforms
from src.model import create_model

# Dummy test để kiểm tra shape tensor
# Lưu ý: file này hiện MỚI chỉ test bước tạo model (tải weights + đổi box_predictor thành công hay không),
# chưa thực sự tạo WheatDataset/DataLoader và lấy 1 batch ảnh thật (cần có data/train + train.csv để test đầy đủ pipeline).
print("[CHECK] Testing pipeline setup...")
model = create_model(num_classes=2)
print("[PASS] Model created successfully!")