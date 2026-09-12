# Global Wheat Detection

Pipeline huấn luyện Faster R-CNN (ResNet50-FPN) cho cuộc thi Kaggle
[Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) —
phát hiện bounding box của các "bông lúa mì" (wheat head) trong ảnh ruộng.

Codebase được thiết kế để dùng chung được ở **2 môi trường**: phát triển/debug ở
local (VS Code) và huấn luyện thật trên Kaggle Notebook (GPU) — chỉ cần đổi
`--config`, không cần sửa code. Xem chi tiết kiến trúc & luồng dữ liệu trong
[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

## 1. Cấu trúc thư mục

```
wheat_detection/
├── .gitignore
├── README.md
├── SYSTEM_ARCHITECTURE.md
├── requirements.txt
├── configs/
│   ├── local_config.yaml       # chạy thử nhanh ở local (ảnh nhỏ, 1 epoch)
│   └── kaggle_config.yaml      # cấu hình train thật trên Kaggle GPU
├── src/
│   ├── __init__.py
│   ├── utils.py                 # config, seed, parse bbox, checkpoint, đo loss
│   ├── dataset.py                # Dataset, augmentation, DataLoader factory
│   ├── model.py                  # kiến trúc Faster R-CNN
│   └── train.py                  # script huấn luyện chính (CLI)
└── test_pipeline.py              # kiểm tra nhanh việc tạo model ở local
```

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

Trên Kaggle Notebook, hầu hết các thư viện (torch, torchvision, pandas, opencv,
scikit-learn, tqdm) đã có sẵn — thường chỉ cần cài thêm `albumentations` nếu bản
cài sẵn quá cũ.

## 3. Chuẩn bị dữ liệu

### Ở local

Giải nén `global-wheat-detection.zip` sao cho khớp với `configs/local_config.yaml`:

```
data/
├── train/
│   ├── <image_id>.jpg
│   └── ...
└── train.csv
```

### Trên Kaggle

Add cuộc thi "Global Wheat Detection" vào Notebook (qua tab **Competitions** hoặc
**Add Input**). Dữ liệu sẽ tự động mount vào `/kaggle/input/...` theo đường dẫn đã
khai báo sẵn trong `configs/kaggle_config.yaml`. Nếu Kaggle mount ở đường dẫn khác
(tuỳ cách bạn add dataset), chỉnh lại 2 khoá `data_dir` và `csv_path` trong file
config cho khớp.

Nhớ bật **Internet** trong Notebook Settings vì `src/model.py` cần tải pretrained
weights (ImageNet + COCO) ở lần chạy đầu tiên.

## 4. Cách chạy

### 4.1. Kiểm tra nhanh pipeline (local)

```bash
python test_pipeline.py
```

Chỉ kiểm tra bước tạo model (tải weights + thay box predictor) chạy được không —
xem comment trong file để biết cách mở rộng test cho cả Dataset/DataLoader thật.

### 4.2. Huấn luyện

Script chạy như một **module** trong package `src` (không chạy file rời), luôn
gọi từ thư mục gốc project:

```bash
# Chạy thử nhanh ở local (CPU, ảnh nhỏ, 1 epoch)
python -m src.train --config configs/local_config.yaml

# Train thật trên Kaggle Notebook (GPU)
python -m src.train --config configs/kaggle_config.yaml
```

Kết quả mỗi epoch in ra từng thành phần loss của Faster R-CNN
(`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`,
`total_loss`) cho cả tập train và validation.

## 5. Output

Sau khi train, thư mục `output_dir` (khai báo trong config) sẽ chứa:

| File | Ý nghĩa |
|---|---|
| `best_model.pth` | Checkpoint có **val loss thấp nhất** — dùng file này để inference/submit. |
| `last_model.pth` | Checkpoint của **epoch cuối cùng** — dùng để resume training nếu bị ngắt giữa chừng. |

Mỗi checkpoint là 1 dict gồm: `epoch`, `model_state`, `optimizer_state`,
`train_loss`, `val_loss`, `config` (toàn bộ config đã dùng để train ra checkpoint đó).

## 6. Cấu hình quan trọng (`configs/*.yaml`)

| Khoá | Ý nghĩa |
|---|---|
| `seed` | Seed cố định cho NumPy/PyTorch để tái lập kết quả. |
| `n_splits`, `fold` | Chia Stratified K-Fold theo cột `source`; `fold` là phần dùng làm validation. |
| `image_size` | Cạnh ảnh sau resize (ảnh vuông). |
| `use_amp` | Bật Automatic Mixed Precision (chỉ có tác dụng khi có GPU CUDA). |
| `pretrained` | `true` để tải pretrained weights (ImageNet + COCO), `false` để khởi tạo ngẫu nhiên (không cần internet). |

## 7. Vài điểm kỹ thuật đáng chú ý

- **Bbox được clip + lọc**: toạ độ bbox được clip về trong biên ảnh và loại bỏ box
  diện tích <= 0 trước khi đưa vào model (xem `_convert_and_clip_boxes` trong
  `src/dataset.py`).
- **Validation loss của Faster R-CNN**: torchvision detection model chỉ trả về
  loss khi ở `train()` mode; vòng lặp validate phải giữ `model.train()` nhưng bọc
  trong `torch.no_grad()` để không cập nhật trọng số (xem docstring của
  `validate_one_epoch` trong `src/train.py`).
- **`.gitignore`** đã chặn `data/`, `outputs/`, `*.pth`, `*.csv` để tránh đẩy dữ
  liệu/checkpoint nặng lên Git.
