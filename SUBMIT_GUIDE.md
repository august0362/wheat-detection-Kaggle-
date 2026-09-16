# Hướng dẫn Train + Nộp bài (Code Competition, Internet Off)

Tóm tắt ngắn gọn quy trình từ lúc train xong đến lúc nộp bài. Chi tiết/lý do kỹ
thuật xem [README.md](README.md) mục 8-9 và comment trong
`scripts/deploy_kaggle.py`.

## Tổng quan

- **Train**: Kaggle Notebook riêng, **Internet ON**, dùng GPU.
- **Nộp bài**: `notebooks/kaggle_submission.ipynb`, **Internet OFF**, code + model
  + wheel pip lấy từ 1 Kaggle Dataset offline (đóng gói sẵn bằng
  `scripts/deploy_kaggle.py`), không cần mạng lúc chấm.

## Bước 1 — Train trên Kaggle GPU

1. Push code mới nhất lên GitHub: double-click `push_to_github.bat`.
2. Notebook train (Internet ON, Accelerator = GPU):
   ```python
   !git clone https://github.com/august0362/wheat-detection-Kaggle-.git repo
   %cd repo
   !git log -1 --oneline          # kiểm tra đúng commit mới nhất trước khi train
   !pip install -q -U ultralytics albumentations
   !python -m src.train --config configs/kaggle_config.yaml
   ```
3. Train xong -> tab **Output** -> tải `weights/best.pt` về máy.

## Bước 2 — Đẩy model + code lên Kaggle Dataset (offline bundle)

Double-click `push_to_kaggle.bat` (hỏi đường dẫn `best.pt` + xác nhận trước khi
chạy) — hoặc chạy tay:
```powershell
python scripts/deploy_kaggle.py --weights <đường-dẫn-best.pt> --slug wheat-yolov8-offline-bundle -m "ghi chú"
```
- **Chỉ lần đầu tiên** tạo dataset: thêm `--new`.
- Cần đã đăng nhập Kaggle CLI (`kaggle config view` để kiểm tra; nếu chưa,
  `pip install kaggle` rồi chạy `kaggle` 1 lần để đăng nhập qua trình duyệt).
- Script tự chặn nếu repo còn thay đổi chưa `git commit`.

## Bước 3 — Nộp bài

1. Mở `notebooks/kaggle_submission.ipynb` trên Kaggle.
2. **Add Input**: dataset `wheat-yolov8-offline-bundle` (tìm bằng cách dán URL
   dataset vào ô search nếu gõ từ khoá không ra) + cuộc thi **Global Wheat
   Detection** (tab Competitions).
3. **Session options -> Internet: Off** (bắt buộc).
4. Restart session (để nạp đúng version mới nhất của dataset).
5. Run All — kiểm tra chạy hết không lỗi, `submission.csv` ra đúng số dòng.
6. **Save Version -> "Save & Run All (Commit)"** — bắt buộc, chạy tay từng cell
   KHÔNG tính là bài nộp hợp lệ.
7. Đợi commit chạy xong -> mục **"Submit to competition"** -> chọn
   `submission.csv` -> Submit.

Lần sau (train lại / tune threshold) chỉ lặp lại Bước 1-3, không cần sửa gì
khác trong code hay notebook.

## Sự cố đã gặp (tham khảo nếu lặp lại)

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Add Input search "0 Results" dù dataset đã "Ready" | Search index bị trễ so với dataset thật | Dán thẳng URL dataset vào ô search thay vì gõ từ khoá |
| `pip install --find-links=...` báo "Location ... is ignored: non-existing path" | Kaggle mount dataset dạng `/kaggle/input/datasets/<username>/<slug>/`, không phải `/kaggle/input/<slug>/` trực tiếp | Notebook đã tự dò bằng `glob` (cell đầu) — không cần sửa tay |
| Cài xong `ultralytics` báo thiếu `ultralytics-platform` | Bản `ultralytics` mới (python >= 3.11) bắt buộc thêm gói HUB/telemetry, kéo theo cả chuỗi httpx/anyio/polars | Đã thêm đủ vào `DEFAULT_PACKAGES` trong `scripts/deploy_kaggle.py` |
| `import ultralytics` treo lâu rồi mới báo `KeyboardInterrupt` | `ultralytics` tự gọi `is_online()` kiểm tra mạng; Kaggle Internet Off không reject ngay mà để treo | Notebook đã thêm `socket.setdefaulttimeout(2)` trước `import ultralytics` |
| `kaggle datasets create` chạy xong (exit code 0) nhưng dataset không được tạo | Kaggle CLI chỉ in dòng "... error ..." chứ không trả exit code khác 0 khi trùng tên/slug | `deploy_kaggle.py` đã tự soi chữ "error" trong output và báo lỗi rõ ràng |
| Deploy báo "còn thay đổi chưa commit" | `git archive` chỉ đóng gói từ commit HEAD, không thấy file mới chưa `git add` | `git add` + `git commit` trước khi deploy |
