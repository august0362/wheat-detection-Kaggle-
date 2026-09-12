from src.model import create_model

# Dummy test để kiểm tra shape tensor
# Lưu ý: file này hiện MỚI chỉ test bước tạo model (tải weights + đổi box_predictor thành công hay không),
# chưa thực sự tạo WheatDataset/DataLoader và lấy 1 batch ảnh thật. Muốn test đầy đủ pipeline
# (cần có data/train/*.jpg + data/train.csv), có thể bổ sung:
#   import pandas as pd
#   from src.dataset import WheatDataset, get_train_dataloader, get_train_transforms
#   df = pd.read_csv("data/train.csv")
#   dataset = WheatDataset(df, "data/train", transforms=get_train_transforms(512))
#   loader = get_train_dataloader(dataset, batch_size=2)
#   images, targets = next(iter(loader))
print("[CHECK] Testing pipeline setup...")
model = create_model(num_classes=2)
print("[PASS] Model created successfully!")