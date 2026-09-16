# Ghi chú Pipeline — Global Wheat Detection (YOLOv8)

File này để đọc lại sau, hiểu nhanh "đã làm gì" và "còn thiếu gì" mà không cần
đọc lại toàn bộ code. Tài liệu kỹ thuật đầy đủ hơn nằm ở
[README.md](README.md) (cách chạy) và [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)
(kiến trúc chi tiết từng module).

## 1. Bối cảnh

Ban đầu project dùng **Faster R-CNN** (torchvision). Đã **thiết kế lại toàn bộ**
sang **YOLOv8 (Ultralytics)**, theo đúng 5 thành phần: chuẩn bị dữ liệu →
augmentation → metric cuộc thi → train/tracking → inference/submission. Code chỉ
viết/lưu ở VS Code, chạy thật (data prep + train + infer) trên Kaggle GPU.

## 2. Đã làm được gì (theo từng file)

| File | Làm gì | Ghi chú |
|---|---|---|
| `src/utils.py` | Convert bbox qua lại COCO ↔ xyxy (pixel) ↔ YOLO (normalized), clip bbox về biên ảnh | Nền tảng dùng chung, các module khác đều gọi tới |
| `src/data_prep.py` | Đọc `train.csv`, chia Stratified 80/20 theo cột `source` (chia theo **ảnh**, không theo **dòng bbox**), ghi label `.txt` chuẩn YOLO, copy ảnh, xuất `data.yaml` | Có xử lý 49 ảnh "nền" (không có bbox nào) — coi là nhóm riêng khi chia, không bị bỏ sót |
| `src/augmentations.py` | HorizontalFlip, VerticalFlip, RandomBrightnessContrast, HueSaturationValue, Blur/GaussianBlur, ShiftScaleRotate, `min_visibility=0.3` | Gắn vào Ultralytics bằng **monkey-patch** (xem mục 4 — đây là chỗ "mong manh" nhất của cả pipeline) |
| `src/metrics/evaluator.py` | Tính đúng metric cuộc thi: Greedy Matching tại IoU 0.50→0.75 (bước 0.05), lấy trung bình | Độc lập hoàn toàn với PyTorch/Ultralytics, chỉ dùng NumPy |
| `src/train.py` | Train YOLOv8 qua Ultralytics + callback tự tính custom metric trên tập val sau mỗi N epoch | Dùng hook `on_model_save` (sau khi `last.pt` đã ghi xong đĩa) |
| `src/visualize.py` | Vẽ biểu đồ loss + custom score theo epoch, vẽ ảnh so sánh prediction (đỏ) vs GT (xanh) | Tự chạy cuối `train.py`, không cần gọi tay |
| `src/infer.py` | Load `best.pt`, predict tập test, xuất `submission.csv` đúng định dạng | NMS + lọc confidence dùng luôn tham số có sẵn của `model.predict()` |
| `configs/local_config.yaml` | Config chạy thử nhanh ở local (CPU, yolov8n, 1 epoch) | Chỉ để test pipeline không lỗi, KHÔNG dùng để train thật |
| `configs/kaggle_config.yaml` | Config train thật trên Kaggle GPU (yolov8m, 60 epoch) | |
| `push_to_github.bat` | Double-click để add + commit + pull + push lên GitHub | Có hỏi xác nhận trước khi làm |

**Đã test thật** (không chỉ viết xong là xong): chạy `src/train.py` end-to-end
trên tập con dữ liệu thật ở local — data prep ra đúng nhãn (đối chiếu tay với
`train.csv` gốc), augmentation patch không lỗi, custom metric callback chạy
đúng và in ra số hợp lý (~0.75 trên tập nhỏ mới train 1 epoch).

## 3. Vài quyết định kỹ thuật quan trọng (để nhớ vì sao làm vậy)

- **Chia train/val theo ảnh, không theo dòng CSV**: 1 ảnh có nhiều bbox, nếu
  chia theo dòng thì cùng 1 ảnh có thể vừa lọt train vừa lọt val → leak dữ liệu.
- **Custom metric ≠ mAP**: `results.csv` mà Ultralytics tự in ra là mAP kiểu
  COCO, KHÁC với cách chấm điểm thật của cuộc thi (Greedy Matching, không phải
  precision-recall chuẩn) → phải tự viết `evaluator.py` riêng, không dùng số
  mAP của Ultralytics để đánh giá.
- **Tắt augmentation gốc của Ultralytics** (`disable_native_augmentation: true`):
  vì augmentation tuỳ biến ở `src/augmentations.py` đã làm flip/xoay/màu sắc
  rồi, nếu không tắt sẽ bị augment 2 lần chồng nhau.
- **Augmentation phải "monkey-patch"** vào Ultralytics thay vì truyền tham số
  bình thường: vì Ultralytics chặn cứng mọi tham số lạ trong `model.train()`,
  và class `Albumentations` mặc định không cho chỉnh `min_visibility` (0.3 —
  yêu cầu bắt buộc của cuộc thi). Cách này **dựa vào cấu trúc nội bộ của
  Ultralytics**, không phải API chính thức → nếu sau này `pip install -U
  ultralytics` lên bản mới mà augmentation "im lặng không chạy nữa", đây là
  nghi phạm số 1 cần kiểm tra lại (đã có try/except nên sẽ không crash, chỉ
  âm thầm rơi về augmentation mặc định).

## 4. Rủi ro / hạn chế hiện tại

1. **Monkey-patch Albumentations** (đã nói ở trên) — mong manh theo version Ultralytics.
2. **`metric_eval_interval`**: mỗi lần tính custom metric phải load lại model +
   predict lại TOÀN BỘ tập val → khá tốn thời gian trên dataset lớn. Đang để
   mặc định 5 epoch/lần ở `kaggle_config.yaml` để đỡ tốn, không phải epoch nào
   cũng tính.
3. **Auto-resume (đã làm, `src/train.py::main`)**: nếu `weights/last.pt` đã có
   sẵn trong `<output_dir>/<experiment_name>/`, lần chạy sau tự `YOLO(last.pt)`
   + `model.train(resume=True)` thay vì train lại từ đầu. Chỉ có tác dụng nếu
   `/kaggle/working` (nơi chứa `output_dir`) còn nguyên trên đĩa — nếu Kaggle
   teardown hẳn session (không chỉ mất mạng tạm thời làm phải Restart Session)
   thì `/kaggle/working` mất theo, vẫn phải train lại từ đầu như cũ.
4. **Chỉ chia 1 lần 80/20**, chưa phải K-Fold thật (5 fold) → nếu val set "may
   mắn" dễ/khó bất thường, số liệu custom metric có thể lệch so với thực tế.
5. **Chưa có test-time augmentation (TTA)** và **chưa ensemble nhiều model**.
6. **Ngưỡng `conf`/`iou` lúc inference đang là số cố định** trong config, chưa
   có bước tự động dò ngưỡng tối ưu dựa trên custom metric trước khi submit.
7. **`push_to_github.bat` dùng `git add -A`** — tiện nhưng add tất cả, cần để ý
   không lỡ tay tạo file nhạy cảm (API key, v.v.) trong thư mục project.

## 5. Ý tưởng cải tiến tiếp theo (gợi ý theo độ ưu tiên)

1. **Dò ngưỡng conf/iou tối ưu** trên tập val bằng chính `evaluator.py` (sweep
   vài giá trị conf, chọn giá trị cho custom score cao nhất) trước khi predict
   trên test — cải thiện điểm submit mà không cần train lại.
2. **K-Fold thật (5 fold) + ensemble**: train 5 model trên 5 fold khác nhau,
   lúc infer trung bình cộng (hoặc WBF — Weighted Boxes Fusion) kết quả 5 model.
3. **Test-Time Augmentation**: `model.predict(..., augment=True)` của
   Ultralytics — dễ bật, thường tăng nhẹ độ chính xác, đổi lại infer chậm hơn.
4. ~~Resume training~~ — **đã làm**, xem mục 4.3.
5. **Pseudo-labeling trên test set** (kỹ thuật phổ biến trong các giải pháp
   thật của cuộc thi này): dùng model đã train dự đoán nhãn cho ảnh test tự
   tin cao, thêm vào tập train, train lại.
6. **Log ra W&B/TensorBoard** thay vì chỉ csv/png cây nhà lá vườn — dễ so sánh
   nhiều lần chạy hơn khi thử nhiều cấu hình.

## 6. Vấn đề đã gặp khi vận hành thật (để không lặp lại)

- **Kaggle chạy nhầm code cũ (Faster R-CNN)** dù đã sửa xong: nguyên nhân là
  `!git pull` trên Kaggle chạy TRƯỚC khi code mới được push từ máy local lên
  GitHub. → Luôn kiểm tra `!git log -1 --oneline` trên Kaggle trước khi bấm
  train, để chắc chắn đang ở đúng commit mới nhất.
- **GPU P100 vs T4 x2**: T4 x2 có Tensor Core nên AMP nhanh hơn nhiều, Ultralytics
  hỗ trợ multi-GPU sẵn (`device: "0,1"`), nhưng Kaggle tính quota GPU-hàng-tuần
  theo **số GPU × giờ** — dùng T4 x2 1 giờ tốn 2 giờ quota. P100 chậm hơn nhưng
  tiết kiệm quota hơn nếu cần chạy nhiều thí nghiệm trong tuần.
