"""
src/model.py
------------
Định nghĩa kiến trúc model dùng cho bài toán Global Wheat Detection.

Sử dụng Faster R-CNN với backbone ResNet50-FPN (Feature Pyramid Network) có sẵn
trong torchvision, áp dụng kỹ thuật transfer learning: giữ nguyên phần backbone đã
pretrain trên COCO (80 class) để tận dụng đặc trưng ảnh tổng quát đã học được, chỉ
thay thế "đầu" phân loại cuối (box predictor) để khớp với bài toán 2 class của
mình (background / wheat_head).
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 0: background (ngầm định, không phải nhãn "wheat" nào), 1: wheat_head.
NUM_CLASSES_DEFAULT: int = 2


def create_model(num_classes: int = NUM_CLASSES_DEFAULT, pretrained: bool = True) -> torch.nn.Module:
    """
    Tạo model Faster R-CNN ResNet50-FPN, đã thay thế box predictor cho phù hợp với
    `num_classes` của bài toán hiện tại.

    Args:
        num_classes: tổng số class BAO GỒM CẢ background. Với bài toán này mặc
            định = 2 (0: background, 1: wheat_head).
        pretrained: True để tải trọng số đã pretrain trên COCO (`weights="DEFAULT"`,
            cần internet ở lần chạy đầu tiên — trên Kaggle nhớ bật "Internet" trong
            Notebook Settings). False để khởi tạo trọng số HOÀN TOÀN ngẫu nhiên
            (không tải bất kỳ file weight nào, kể cả backbone ImageNet), hữu ích khi
            chỉ muốn kiểm tra nhanh kiến trúc model mà không cần internet/tải weights.

    Returns:
        `torch.nn.Module` (cụ thể là `torchvision.models.detection.FasterRCNN`) sẵn
        sàng để `.to(device)` và train.
    """
    # Lưu ý: torchvision có 2 nguồn pretrained weight riêng biệt cho hàm này —
    # `weights` (toàn bộ Faster R-CNN, pretrain trên COCO) và `weights_backbone`
    # (riêng ResNet50, pretrain trên ImageNet, MẶC ĐỊNH vẫn được tải kể cả khi
    # weights=None). Phải set cả hai về None thì pretrained=False mới thực sự
    # không tải gì.
    weights = "DEFAULT" if pretrained else None
    weights_backbone = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=weights_backbone
    )

    # Lấy số feature đầu vào của đầu phân loại gốc (đang được train cho 80 class COCO)...
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # ...rồi thay bằng đầu mới chỉ có `num_classes`. Đây chính là bước "transfer
    # learning": giữ nguyên backbone (đã học sẵn đặc trưng ảnh tổng quát), chỉ train
    # lại phần phân loại + hồi quy box cuối cho bài toán riêng của mình.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
