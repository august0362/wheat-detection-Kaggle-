# Global Wheat Detection — YOLOv8

Pipeline end-to-end huấn luyện **YOLOv8 (Ultralytics)** cho cuộc thi Kaggle
[Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection) —
phát hiện bounding box của các "bông lúa mì" (wheat head) trong ảnh ruộng, chỉ
1 class duy nhất (`wheat`).

**Cách vận hành:** code chỉ được viết/lưu/commit ở VS Code (GitHub); toàn bộ xử
lý dữ liệu, huấn luyện và suy luận chạy trên **Kaggle Notebook (GPU)**. Chỉ cần
đổi `--config`, không cần sửa code giữa 2 môi trường. Xem chi tiết kiến trúc &
luồng dữ liệu trong [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

## 1. Cấu trúc thư mục

```
wheat_detection/
├── .gitignore
├── README.md
├── SYSTEM_ARCHITECTURE.md
├── requirements.txt
├── test_pipeline.py              # kiểm tra nhanh bbox convert + metric, không cần data/internet
├── configs/
│   ├── local_config.yaml         # chạy thử nhanh ở local (yolov8n, ảnh nhỏ, 1 epoch)
│   └── kaggle_config.yaml        # cấu hình train thật trên Kaggle GPU (yolov8m/x)
├── scripts/
│   └── deploy_kaggle.py          # đóng gói code+weights+wheel offline -> đẩy lên Kaggle Dataset (mục 9)
├── notebooks/
│   └── kaggle_submission.ipynb   # notebook nộp bài tối giản, chạy khi Internet: Off (mục 9)
└── src/
    ├── __init__.py
    ├── utils.py                  # config, seed, parse/convert bbox (COCO <-> xyxy <-> YOLO)
    ├── data_prep.py               # (1) Data Preparation & Splitting
    ├── augmentations.py           # (2) Augmentation Pipeline (Albumentations, on-the-fly)
    ├── metrics/
    │   └── evaluator.py           # (3) Evaluation Metric (Greedy Matching, IoU 0.50:0.75:0.05)
    ├── train.py                   # (4) Training & Tracking
    ├── visualize.py               # (4) biểu đồ + ảnh so sánh prediction vs GT
    └── infer.py                   # (5) Inference & Submission
```

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

Trên Kaggle Notebook, hầu hết thư viện (torch, pandas, opencv, scikit-learn,
matplotlib, tqdm) đã có sẵn — thường chỉ cần cài thêm `ultralytics` và
`albumentations` nếu bản cài sẵn quá cũ:

```bash
!pip install -q -U ultralytics albumentations
```

## 3. Chuẩn bị dữ liệu

### Ở local (VS Code)

Giải nén `global-wheat-detection.zip` khớp với `configs/local_config.yaml`:

```
data/
├── train/<image_id>.jpg
├── test/<image_id>.jpg
├── train.csv
└── sample_submission.csv
```

### Trên Kaggle

Add cuộc thi "Global Wheat Detection" vào Notebook (tab **Competitions** hoặc
**Add Input**) — dữ liệu tự mount vào `/kaggle/input/...` theo đường dẫn đã
khai báo sẵn trong `configs/kaggle_config.yaml`. Nếu Kaggle mount ở đường dẫn
khác, chỉnh lại các khoá trong mục `data:` của config cho khớp.

Nhớ bật **Internet** trong Notebook Settings (tải pretrained weights YOLOv8 ở
lần chạy đầu tiên).

## 4. Cách chạy

Script chạy như MODULE trong package `src`, luôn gọi từ thư mục gốc project:

```bash
# 1) Kiểm tra nhanh logic (không cần data/internet)
python test_pipeline.py

# 2) Chạy thử nhanh toàn bộ pipeline ở local (CPU, model nhỏ, 1 epoch)
python -m src.train --config configs/local_config.yaml

# 3) Train thật trên Kaggle Notebook (GPU) — cell của Kaggle:
!git pull
!python -m src.train --config configs/kaggle_config.yaml

# 4) Inference + sinh submission.csv (trên Kaggle, sau khi train xong):
!python -m src.infer --config configs/kaggle_config.yaml --output submission.csv
```

`src/train.py` tự gọi `src/data_prep.py` ở đầu mỗi lần chạy (bỏ qua nếu dataset
YOLO đã được build sẵn) — không cần chạy `data_prep.py` riêng, nhưng vẫn có thể:

```bash
python -m src.data_prep --config configs/kaggle_config.yaml
```

Xem trước augmentation pipeline (không cần train) — kiểm tra độc lập thành phần 2:

```bash
python -m src.augmentations --config configs/local_config.yaml --n 6
```

## 5. Output (sau khi train)

Trong `<train.output_dir>/<experiment_name>/` (Ultralytics tự tạo):

| File/thư mục | Ý nghĩa |
|---|---|
| `weights/best.pt` | Checkpoint tốt nhất theo fitness (mAP) trên tập val — dùng để inference/submit. |
| `weights/last.pt` | Checkpoint epoch cuối — dùng để resume. |
| `results.csv` | Loss (box/cls/dfl) + mAP mỗi epoch, Ultralytics tự sinh. |
| `custom_metric_history.csv` | Custom Competition Score (IoU 0.50:0.75:0.05) mỗi `metric_eval_interval` epoch — do `src/train.py` tự ghi. |
| `training_history.png` | Biểu đồ loss + custom metric (`src/visualize.py::plot_training_history`). |
| `val_predictions_vs_gt.png` | Ảnh so sánh prediction (đỏ) vs Ground Truth (xanh lá) trên 1 batch mẫu tập val. |

## 6. Cấu hình quan trọng (`configs/*.yaml`)

| Khoá | Ý nghĩa |
|---|---|
| `data.val_size` | Tỉ lệ Stratified Split val (mặc định 0.2 = 80/20), stratify theo cột `source`. |
| `data.copy_images` | `true`: copy ảnh vào `yolo_dataset_dir` (bắt buộc trên Kaggle vì `/kaggle/input` chỉ đọc). `false`: symlink (nhanh hơn, chỉ dùng khi có quyền ghi/symlink). |
| `augmentation.*` | Xác suất từng phép augment (HorizontalFlip, VerticalFlip, RandomBrightnessContrast, HueSaturationValue, Blur/GaussianBlur, ShiftScaleRotate) + `min_visibility`. |
| `model.variant` | `yolov8m.pt` (cân bằng tốc độ/độ chính xác) hoặc `yolov8x.pt` (chính xác hơn, chậm hơn). |
| `train.disable_native_augmentation` | `true` (khuyến nghị): tắt hsv/degrees/translate/scale/shear/flip mặc định của Ultralytics để không augment trùng với Albumentations tuỳ biến. |
| `train.metric_eval_interval` | Số epoch giữa 2 lần tính Custom Competition Metric (tốn thời gian vì phải predict lại toàn bộ tập val). |
| `inference.conf_threshold` / `iou_threshold` | Ngưỡng confidence/NMS dùng lúc sinh submission — nên tune trên `custom_metric_history.csv`/`val_predictions_vs_gt.png` trước khi submit. |

## 7. Vài điểm kỹ thuật đáng chú ý

- **Bbox được clip + lọc**: toạ độ bbox được clip về biên ảnh và loại bỏ box
  diện tích <= 0 trước khi ghi label YOLO (`src/utils.py::coco_to_xyxy_clipped`).
- **Ảnh "nền" (background)**: cuộc thi có 49 ảnh không có wheat head nào (có
  file ảnh nhưng không có dòng trong `train.csv`) — `src/data_prep.py` vẫn đưa
  các ảnh này vào dataset (label rỗng), giúp model học cả trường hợp âm tính,
  và vẫn được stratified split đồng đều giữa train/val.
- **Augmentation tuỳ biến gắn vào Ultralytics bằng monkey-patch** (không phải
  API công khai chính thức): `ultralytics.data.augment.Albumentations` mặc
  định không cho tuỳ chỉnh `min_visibility` — `src/augmentations.py` thay thế
  class này để dùng đúng bộ transform + `min_visibility=0.3` cuộc thi yêu cầu.
  Được bọc try/except: nếu 1 bản Ultralytics tương lai đổi cấu trúc nội bộ,
  pipeline in cảnh báo và tự rơi về augmentation mặc định thay vì crash.
- **Custom metric không phải mAP**: mAP mà Ultralytics tự tính (`results.csv`)
  KHÁC với metric thật của cuộc thi (Greedy Matching, không phải
  precision-recall theo chuẩn COCO) — `src/metrics/evaluator.py` triển khai
  đúng công thức cuộc thi, độc lập hoàn toàn với Ultralytics/PyTorch.
- **`.gitignore`** đã chặn `data/`, `outputs/`, `*.pt`, `*.csv` để tránh đẩy dữ
  liệu/checkpoint nặng lên Git — `runs/`/`yolo_dataset/` sinh ra trong các thư
  mục này nên cũng tự động được bỏ qua.

## 8. Quy trình làm việc 2 đầu (VS Code <-> Kaggle)

1. Mở VS Code, sửa logic trong `src/` hoặc đổi siêu tham số trong `configs/kaggle_config.yaml`.
2. `git add . && git commit -m "..." && git push`.
3. Trên Kaggle Notebook: `!git pull && !python -m src.train --config configs/kaggle_config.yaml`.
4. Sau khi train xong: `!python -m src.infer --config configs/kaggle_config.yaml --output submission.csv`, rồi Submit trực tiếp từ Notebook (Kaggle > Submit to Competition) hoặc tải `submission.csv` về.

## 9. Nộp bài Code Competition khi Internet Off

Global Wheat Detection là **Code Competition**: bài nộp thật phải sinh ra từ 1
Kaggle Notebook chạy ở chế độ **Internet: Off**, nên không thể `!git pull` hay
`!pip install` như mục 8. Quy trình:

1. Train trên Kaggle GPU như bình thường (mục 8), tải `best.pt` về máy từ tab
   **Output** của training notebook.
2. Đóng gói code (từ Git HEAD) + `best.pt` + các wheel pip còn thiếu thành 1
   Kaggle Dataset private, bằng `scripts/deploy_kaggle.py` (chạy ở local):
   ```bash
   python scripts/deploy_kaggle.py --weights <đường-dẫn-best.pt> --slug wheat-yolov8-offline-bundle --new
   # Các lần cập nhật sau, bỏ --new:
   python scripts/deploy_kaggle.py --weights <đường-dẫn-best.pt> --slug wheat-yolov8-offline-bundle -m "..."
   ```
3. Mở `notebooks/kaggle_submission.ipynb` trên Kaggle, Add Input dataset vừa
   tạo + cuộc thi (tab Competitions), bật **Internet: Off**, sửa `BUNDLE_DIR`
   khớp slug dataset, chạy toàn bộ -> Submit.

Setup `kaggle/kaggle.json` (đã có sẵn, đã gitignore) và các lưu ý về wheel
đúng nền tảng: xem comment đầu `scripts/deploy_kaggle.py`.
