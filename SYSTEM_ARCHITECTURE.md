# System Architecture — Global Wheat Detection (YOLOv8)

Tài liệu này mô tả kiến trúc hệ thống, luồng dữ liệu, và chức năng chi tiết của
từng module. Đọc kèm [README.md](README.md) để biết cách cài đặt/chạy.

## 1. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RAW DATA (Kaggle)                                                        │
│  train.csv (image_id, width, height, bbox="[x,y,w,h]" COCO, source)       │
│  + thư mục train/*.jpg (3422 ảnh, 3373 có annotation, 49 ảnh "nền")       │
└──────────────────────────────┬───────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  (1) DATA PREPARATION & SPLITTING  (src/data_prep.py)                     │
│                                                                             │
│  build_image_metadata()      1 dòng/ảnh, gộp cả ảnh "nền" (không có bbox) │
│         │                    vào nhóm __no_annotation__ để không bị bỏ sót │
│         ▼                                                                  │
│  stratified_image_split()    train_test_split 80/20, stratify theo        │
│         │                    'source', chia Ở CẤP ĐỘ ẢNH (không xé lẻ bbox)│
│         ▼                                                                  │
│  build_yolo_dataset()        coco_to_yolo() (src/utils.py) parse+clip+    │
│                               convert bbox -> ghi label .txt chuẩn YOLO,   │
│                               copy/symlink ảnh, xuất data.yaml             │
└──────────────────────────────┬───────────────────────────────────────────┘
                                │  <yolo_dataset_dir>/{images,labels}/{train,val}/, data.yaml
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  (2) AUGMENTATION PIPELINE  (src/augmentations.py)                        │
│                                                                             │
│  patch_ultralytics_albumentations()  monkey-patch                         │
│  ultralytics.data.augment.Albumentations bằng class tự viết, dùng          │
│  build_bbox_transform(): HorizontalFlip, VerticalFlip,                    │
│  RandomBrightnessContrast, HueSaturationValue, Blur/GaussianBlur,         │
│  ShiftScaleRotate — BboxParams(format="yolo", min_visibility=0.3, clip=True)│
│  -> áp dụng ON-THE-FLY ngay trong DataLoader nội bộ của Ultralytics.       │
└──────────────────────────────┬───────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  (4) TRAINING & TRACKING  (src/train.py)                                  │
│                                                                             │
│  main()                                                                    │
│   ├─ build_yolo_dataset(cfg)          Bước (1), idempotent                │
│   ├─ patch_ultralytics_albumentations() Bước (2)                          │
│   ├─ _load_val_ground_truth()         đọc label .txt tập val 1 lần,       │
│   │                                   denormalize -> xyxy pixel (Ground   │
│   │                                   Truth cố định cho callback metric)  │
│   ├─ YOLO(model.variant) + add_callback("on_model_save", metric_callback) │
│   └─ model.train(data=data.yaml, ..., **native_aug_overrides)             │
│         │         (tắt hsv/degrees/translate/scale/shear/flip mặc định   │
│         │          của Ultralytics — tránh augment trùng với Bước 2)      │
│         │                                                                  │
│         ▼  mỗi epoch, SAU KHI last.pt/best.pt đã ghi xong đĩa:             │
│      make_metric_callback()  load lại last.pt -> predict toàn bộ tập val  │
│         │                    -> competition_score() (src/metrics/         │
│         │                    evaluator.py) so với Ground Truth cố định    │
│         └─> ghi đè custom_metric_history.csv (epoch, custom_score)        │
│                                                                             │
│  Sau khi train xong:                                                       │
│   ├─ plot_training_history()          loss (results.csv) + custom score   │
│   └─ visualize_val_predictions()      ảnh so sánh pred (đỏ) vs GT (xanh)  │
└──────────────────────────────┬───────────────────────────────────────────┘
                                │  weights/best.pt
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  (5) INFERENCE & SUBMISSION  (src/infer.py)                               │
│                                                                             │
│  run_inference()                                                           │
│   ├─ YOLO(best.pt).predict(source=test_dir, conf=..., iou=...)            │
│   │  (NMS + lọc confidence đã áp dụng nội bộ qua tham số conf/iou)         │
│   ├─ format_prediction_string()  "conf x_min y_min width height", sắp xếp │
│   │                               giảm dần theo confidence mỗi ảnh         │
│   └─ ghi submission.csv (MỌI image_id trong sample_submission.csv đều có  │
│                           1 dòng, kể cả khi PredictionString rỗng)         │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Bảng chức năng chi tiết theo module

### `src/utils.py`

| Thành phần | Chức năng |
|---|---|
| `load_config(path)` | Đọc YAML -> dict. |
| `set_seed(seed)` | Cố định seed `random`/NumPy/PyTorch. |
| `parse_bbox_string(s)` | `"[x, y, w, h]"` (string) -> `[x, y, w, h]` (float). |
| `coco_to_xyxy_clipped(bbox, w, h)` | COCO -> xyxy pixel, clip biên ảnh, `None` nếu box suy biến. |
| `xyxy_to_yolo` / `yolo_to_xyxy` | Chuyển đổi 2 chiều xyxy (pixel) <-> YOLO (normalized). |
| `coco_to_yolo(bbox, w, h)` | COCO -> YOLO trực tiếp (compose 2 hàm trên). |

### `src/data_prep.py` — Thành phần 1

| Thành phần | Chức năng |
|---|---|
| `build_image_metadata(df, image_dir)` | 1 dòng/ảnh (image_id, source), gộp cả ảnh nền. |
| `stratified_image_split(meta, val_size, seed)` | `train_test_split` stratify theo 'source', ở cấp độ ảnh. |
| `build_yolo_dataset(cfg)` | Toàn bộ pipeline: convert bbox, ghi label, copy ảnh, xuất `data.yaml`. Idempotent. |

### `src/augmentations.py` — Thành phần 2

| Thành phần | Chức năng |
|---|---|
| `build_bbox_transform(...)` | `A.Compose` gồm 6 phép augment cuộc thi yêu cầu + `BboxParams(min_visibility=0.3)`. |
| `patch_ultralytics_albumentations(...)` | Monkey-patch `ultralytics.data.augment.Albumentations`, PHẢI gọi trước `model.train()`. |
| `main()` (CLI) | `python -m src.augmentations` — xem trước augmentation trên ảnh thật, không cần train. |

### `src/metrics/evaluator.py` — Thành phần 3

| Thành phần | Chức năng |
|---|---|
| `compute_iou_matrix(pred, gt)` | Ma trận IoU (P, G), vector hoá NumPy. |
| `_greedy_match_at_threshold(iou_matrix, t)` | Greedy Matching tại 1 ngưỡng IoU -> `Score(t) = TP/(TP+FP+FN)`. |
| `image_score(...)` | Trung bình `Score(t)` trên 6 ngưỡng [0.50..0.75] cho 1 ảnh. |
| `competition_score(...)` | Trung bình `image_score` trên toàn bộ ảnh — chính là metric cuộc thi. |

### `src/train.py` — Thành phần 4

| Thành phần | Chức năng |
|---|---|
| `_load_val_ground_truth(labels_dir, image_size)` | Đọc label .txt tập val 1 lần, denormalize -> xyxy pixel. |
| `make_metric_callback(...)` | Tạo callback `on_model_save`: load lại checkpoint, predict tập val, tính `competition_score`, ghi CSV. |
| `main()` | Entry point CLI: data prep -> patch augmentation -> `YOLO.train()` -> vẽ biểu đồ + visualize. |

### `src/visualize.py` — Thành phần 4 (tracking)

| Thành phần | Chức năng |
|---|---|
| `plot_training_history(...)` | Biểu đồ loss (từ `results.csv` Ultralytics) + custom score (từ `custom_metric_history.csv`). |
| `visualize_val_predictions(...)` | Lưới ảnh: box dự đoán (đỏ) chồng lên Ground Truth (xanh lá) trên N ảnh val ngẫu nhiên. |

### `src/infer.py` — Thành phần 5

| Thành phần | Chức năng |
|---|---|
| `_list_test_image_ids(...)` | Danh sách image_id cần dự đoán, ưu tiên lấy từ `sample_submission.csv`. |
| `format_prediction_string(boxes, scores)` | Format đúng chuẩn `"conf x_min y_min w h"`, sắp giảm dần theo confidence. |
| `run_inference(...)` | Predict (NMS+conf nội bộ) toàn bộ ảnh test, ghi `submission.csv` đủ dòng cho mọi ảnh. |

## 3. Vì sao "monkey-patch" Albumentations thay vì dùng tham số `model.train()`?

Ultralytics YOLOv8 có sẵn 1 bước Albumentations trong pipeline augment nội bộ,
nhưng: (a) bộ transform mặc định chỉ có Blur/MedianBlur/ToGray/CLAHE với
xác suất rất thấp (0.01), không đúng yêu cầu cuộc thi; (b) `get_cfg()` của
Ultralytics chặn cứng mọi key lạ truyền vào `model.train(**kwargs)`
(`check_dict_alignment`), nên không thể truyền thẳng 1 list transform tuỳ biến
qua tham số công khai; (c) `BboxParams.min_visibility` (0.3, yêu cầu bắt buộc
của cuộc thi) không được expose qua bất kỳ tham số nào của `Albumentations`.

Giải pháp: gán lại `ultralytics.data.augment.Albumentations` bằng 1 class tự
viết cùng interface (`__call__(self, labels) -> labels`) TRƯỚC khi gọi
`model.train()`. Vì hàm `v8_transforms()` (nơi khởi tạo `Albumentations(...)`)
tra cứu tên class qua namespace của MODULE tại thời điểm chạy (không phải
tham chiếu đã cache lúc import), việc gán lại có hiệu lực ngay cả khi thực
hiện sau khi `ultralytics.data.augment` đã được import ở nơi khác.

Đây là kỹ thuật dựa vào cấu trúc nội bộ (không phải API công khai chính thức)
của Ultralytics nên `patch_ultralytics_albumentations()` được bọc try/except:
nếu 1 bản Ultralytics tương lai đổi cấu trúc, pipeline in cảnh báo và tự rơi về
augmentation mặc định thay vì làm crash toàn bộ quá trình train.

## 4. Vận hành 2 đầu: Local (VS Code) vs Kaggle

| | Local (VS Code) | Kaggle Notebook |
|---|---|---|
| Mục đích | Debug pipeline, sửa code, chạy thử nhanh | Train thật, tận dụng GPU |
| Config | `configs/local_config.yaml` | `configs/kaggle_config.yaml` |
| Model | `yolov8n.pt` (nhỏ nhất, chỉ để test pipeline) | `yolov8m.pt`/`yolov8x.pt` |
| Ảnh/Batch/Epoch | 320px, batch=2, 1 epoch | 1024px, batch=16, 60 epoch |
| Device | `cpu` | GPU (`0`, hoặc `0,1` cho 2 GPU) |
| Dữ liệu | Giải nén thủ công vào `data/` | Mount tự động qua `/kaggle/input/...` |
| Lệnh chạy | `python -m src.train --config configs/local_config.yaml` | `python -m src.train --config configs/kaggle_config.yaml` |

Vì mọi tham số môi trường-cụ-thể (đường dẫn, batch size, epoch, device...) đều
nằm trong file YAML, code trong `src/` không chứa bất kỳ điều kiện rẽ nhánh
"nếu đang chạy trên Kaggle thì..." nào — nguyên tắc thiết kế xuyên suốt dự án
để tránh phân mảnh code giữa 2 môi trường.
