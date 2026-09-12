# System Architecture — Global Wheat Detection

Tài liệu này mô tả kiến trúc hệ thống, luồng dữ liệu, và chức năng chi tiết của
từng module trong pipeline. Đọc kèm [README.md](README.md) để biết cách cài đặt/chạy.

## 1. Kiến trúc tổng thể

```
┌─────────────────────┐
│   RAW DATA           │  train.csv (image_id, width, height, bbox="[x,y,w,h]", source)
│   *.jpg + train.csv  │  + thư mục ảnh <image_id>.jpg
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE  (src/dataset.py)                                  │
│                                                                     │
│  create_fold_split()        chia Stratified K-Fold theo image_id  │
│         │                   (train.py, dùng cột "source")         │
│         ▼                                                          │
│  WheatDataset.__getitem__() đọc ảnh (cv2, BGR->RGB)                │
│         │                   parse + clip + lọc bbox                │
│         │                   (parse_bbox_string + _convert_and_     │
│         │                    clip_boxes)                           │
│         ▼                                                          │
│  Albumentations             get_train_transforms / get_valid_      │
│  (resize, flip, normalize)  transforms                             │
│         ▼                                                          │
│  collate_fn()                gom batch dạng tuple (bbox không đều) │
│         ▼                                                          │
│  get_train_dataloader() /   torch.utils.data.DataLoader            │
│  get_val_dataloader()                                              │
└──────────┬──────────────────────────────────────────────────────┘
           │  batch (images, targets)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODEL  (src/model.py)                                             │
│                                                                     │
│  create_model()                                                    │
│    torchvision.models.detection.fasterrcnn_resnet50_fpn(           │
│        weights=..., weights_backbone=...)                          │
│    -> thay box_predictor bằng FastRCNNPredictor(num_classes=2)     │
└──────────┬──────────────────────────────────────────────────────┘
           │  model(images, targets) -> loss_dict (train) / preds (eval)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING LOOP  (src/train.py)                                     │
│                                                                     │
│  main()                                                             │
│   ├─ load_config()               đọc YAML                          │
│   ├─ set_seed()                  cố định seed                      │
│   ├─ build_datasets()            gọi create_fold_split() + Dataset │
│   ├─ get_train/val_dataloader()                                    │
│   ├─ create_model()  + SGD optimizer + GradScaler (AMP)             │
│   └─ for epoch in range(num_epochs):                                │
│         train_one_epoch()   forward (autocast) + backward (scaler) │
│         validate_one_epoch()  forward (no_grad) -> val loss         │
│         save_checkpoint()   ghi last_model.pth mỗi epoch,           │
│                              ghi đè best_model.pth khi val loss ↓   │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   OUTPUT WEIGHTS      │  outputs/best_model.pth, outputs/last_model.pth
└─────────────────────┘
```

## 2. Bảng chức năng chi tiết

### `src/utils.py`

| Thành phần | Loại | Chức năng |
|---|---|---|
| `load_config(config_path)` | hàm | Đọc file YAML -> dict. |
| `set_seed(seed)` | hàm | Cố định seed cho `random`, NumPy, PyTorch (CPU + GPU) + cấu hình cuDNN tất định. |
| `parse_bbox_string(bbox_str)` | hàm | Parse chuỗi `"[x, y, w, h]"` -> `List[float]`. |
| `save_checkpoint(state, filepath)` | hàm | Lưu dict checkpoint ra `.pth`, tự tạo thư mục cha. |
| `AverageMeter` | class | Theo dõi giá trị trung bình cộng dồn của MỘT đại lượng vô hướng. |
| `LossMeter` | class | Theo dõi đồng thời NHIỀU thành phần loss (dùng `AverageMeter` cho từng key), cộng thêm `total_loss`. |

### `src/dataset.py`

| Thành phần | Loại | Chức năng |
|---|---|---|
| `_convert_and_clip_boxes(bbox_strings, w, h)` | hàm (private) | COCO -> Pascal VOC, clip về biên ảnh, loại box diện tích <= 0. |
| `get_train_transforms(image_size)` | hàm | Augmentation pipeline train: Resize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Normalize, ToTensorV2. |
| `get_valid_transforms(image_size)` | hàm | Augmentation pipeline val: Resize, Normalize, ToTensorV2 (không random augment). |
| `WheatDataset` | class (`torch.utils.data.Dataset`) | Đọc 1 ảnh + toàn bộ bbox của ảnh đó, áp transform, trả về `(image_tensor, target_dict)`. |
| `collate_fn(batch)` | hàm | Gom batch thành `tuple(images), tuple(targets)` vì số bbox mỗi ảnh khác nhau. |
| `get_train_dataloader(dataset, batch_size, num_workers, shuffle, pin_memory)` | hàm | Tạo `DataLoader` train (mặc định `shuffle=True`, `drop_last=True`). |
| `get_val_dataloader(dataset, batch_size, num_workers, shuffle, pin_memory)` | hàm | Tạo `DataLoader` validation (mặc định `shuffle=False`, `drop_last=False`). |

### `src/model.py`

| Thành phần | Loại | Chức năng |
|---|---|---|
| `create_model(num_classes, pretrained)` | hàm | Tạo Faster R-CNN ResNet50-FPN, thay `box_predictor` cho khớp `num_classes`. `pretrained=False` tắt tải cả weights COCO lẫn backbone ImageNet. |

### `src/train.py`

| Thành phần | Loại | Chức năng |
|---|---|---|
| `create_fold_split(df, n_splits, fold, seed)` | hàm | Stratified K-Fold theo `image_id`, stratify theo `source`. |
| `train_one_epoch(model, optimizer, scaler, loader, device, use_amp)` | hàm | 1 epoch train: forward (autocast) + backward (GradScaler) + cập nhật trọng số, trả về dict loss trung bình. |
| `validate_one_epoch(model, loader, device, use_amp)` | hàm (`@torch.no_grad()`) | 1 epoch validation: `model.train()` bên trong `no_grad()` để lấy loss mà không học. |
| `build_datasets(cfg)` | hàm | Đọc CSV, chia fold, tạo `WheatDataset` train/val. |
| `main()` | hàm | Entry point CLI: load config -> set seed -> build data/model/optimizer -> vòng lặp epoch -> lưu checkpoint. |

## 3. Vận hành 2 đầu: Local vs Kaggle

| | Local (VS Code) | Kaggle Notebook |
|---|---|---|
| Mục đích | Debug pipeline, sửa code, chạy thử nhanh | Train thật, tận dụng GPU |
| Config | `configs/local_config.yaml` | `configs/kaggle_config.yaml` |
| Ảnh/Batch | Nhỏ (512px, batch=2) | Full size (1024px, batch=8) |
| GPU | Thường không có (CPU) | P100 hoặc T4x2 |
| AMP | Tắt (`use_amp: false`) | Bật (`use_amp: true`) |
| Dữ liệu | Giải nén thủ công vào `data/` | Mount tự động qua `/kaggle/input/...` |
| Lệnh chạy | `python -m src.train --config configs/local_config.yaml` | `python -m src.train --config configs/kaggle_config.yaml` |

Vì mọi tham số môi trường-cụ-thể (đường dẫn, batch size, epoch, AMP...) đều nằm
trong file YAML, **code trong `src/` không cần và không nên** chứa bất kỳ điều
kiện rẽ nhánh "nếu đang chạy trên Kaggle thì..." nào — đây là nguyên tắc thiết kế
xuyên suốt dự án để tránh phân mảnh code giữa 2 môi trường.
