"""
test_pipeline.py
------------------
Kiểm tra nhanh, KHÔNG cần internet/tải weight/tải data thật, các phần logic dễ sai
nhất của pipeline: convert bbox COCO<->YOLO<->xyxy (src/utils.py) và metric cuộc
thi (src/metrics/evaluator.py). Chạy được ở bất kỳ máy nào có cài requirements.txt.

Muốn kiểm tra CẢ pipeline thật (data prep + train + infer, cần data/train thật),
xem configs/local_config.yaml rồi chạy:
    python -m src.train --config configs/local_config.yaml

Chạy file này:
    python test_pipeline.py
"""
import sys

import numpy as np

from src.metrics.evaluator import competition_score, image_score
from src.utils import coco_to_yolo, yolo_to_xyxy

# Console mặc định trên Windows (vd cp1258, cp1252) không encode được tiếng Việt có
# dấu -> print() sẽ crash với UnicodeEncodeError nếu không ép UTF-8 trước.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def test_bbox_roundtrip() -> None:
    img_w, img_h = 1024, 1024
    original = (834.0, 222.0, 56.0, 36.0)  # x, y, w, h (COCO)

    yolo_box = coco_to_yolo(original, img_w, img_h)
    assert yolo_box is not None
    assert all(0.0 <= v <= 1.0 for v in yolo_box), f"Toạ độ YOLO phải nằm trong [0, 1]: {yolo_box}"

    x1, y1, x2, y2 = yolo_to_xyxy(*yolo_box, img_w, img_h)
    expected_x1, expected_y1 = original[0], original[1]
    expected_x2, expected_y2 = original[0] + original[2], original[1] + original[3]
    assert abs(x1 - expected_x1) < 1e-3 and abs(y1 - expected_y1) < 1e-3
    assert abs(x2 - expected_x2) < 1e-3 and abs(y2 - expected_y2) < 1e-3
    print("[PASS] coco_to_yolo <-> yolo_to_xyxy round-trip chính xác.")


def test_bbox_degenerate_dropped() -> None:
    # Box hoàn toàn nằm ngoài ảnh -> phải bị loại (trả về None).
    assert coco_to_yolo((2000.0, 2000.0, 10.0, 10.0), 1024, 1024) is None
    print("[PASS] Box suy biến/ngoài biên ảnh bị loại bỏ đúng.")


def test_evaluator_edge_cases() -> None:
    # Không GT, không pred -> 1.0
    assert image_score(np.empty((0, 4)), np.empty((0,)), np.empty((0, 4))) == 1.0
    # Không GT, có pred -> 0.0
    assert image_score(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.empty((0, 4))) == 0.0
    # Có GT, không pred -> 0.0
    assert image_score(np.empty((0, 4)), np.empty((0,)), np.array([[0, 0, 10, 10]])) == 0.0
    # Pred trùng khớp hoàn hảo với GT -> 1.0
    perfect = image_score(np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[0, 0, 10, 10]]))
    assert abs(perfect - 1.0) < 1e-6
    print("[PASS] evaluator xử lý đúng các trường hợp biên.")


def test_competition_score_average() -> None:
    score = competition_score(
        [np.array([[0, 0, 10, 10]]), np.empty((0, 4))],
        [np.array([0.9]), np.empty((0,))],
        [np.array([[0, 0, 10, 10]]), np.empty((0, 4))],
    )
    assert abs(score - 1.0) < 1e-6
    print("[PASS] competition_score = trung bình cộng Image_Score.")


if __name__ == "__main__":
    print("[CHECK] Kiểm tra nhanh pipeline (không cần internet/data thật)...")
    test_bbox_roundtrip()
    test_bbox_degenerate_dropped()
    test_evaluator_edge_cases()
    test_competition_score_average()
    print("[PASS] Toàn bộ kiểm tra nhanh đã PASS.")
