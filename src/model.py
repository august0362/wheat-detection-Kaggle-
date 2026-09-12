import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes: int = 2):
    """
    Faster R-CNN ResNet50-FPN.
    num_classes = 2 (0: background, 1: wheat_head)
    """
    # weights="DEFAULT" = tải sẵn trọng số đã pretrain trên COCO (80 class) về máy.
    # -> cần internet lần đầu chạy; trên Kaggle nhớ bật "Internet" trong Notebook Settings.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Lấy số feature đầu vào của "đầu" phân loại gốc (đang train cho 80 class COCO)...
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # ...rồi thay bằng đầu mới chỉ có num_classes=2 (background + wheat_head).
    # Đây chính là bước "transfer learning": giữ nguyên backbone đã học sẵn đặc trưng ảnh,
    # chỉ train lại phần phân loại cuối cho bài toán riêng của mình.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model