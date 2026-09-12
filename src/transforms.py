import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation cho tập TRAIN: có random flip/brightness để model học tổng quát hơn,
# tránh học thuộc lòng (overfit) hướng ảnh hay độ sáng cụ thể.
def get_train_transforms(image_size: int = 1024):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),           # p = xác suất áp dụng phép biến đổi
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # mean/std của ImageNet vì backbone ResNet50 pretrained trên ImageNet
            # -> phải chuẩn hoá ảnh đầu vào giống lúc pretrain thì transfer learning mới hiệu quả.
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()   # chuyển ảnh numpy (H, W, C) -> tensor PyTorch (C, H, W)
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",       # phải khớp format bbox đã tạo ở utils.parse_bbox_string
            label_fields=["labels"],   # để Albumentations biết loại bbox nào thì loại label tương ứng theo
            min_area=0,
            min_visibility=0.1         # bbox bị resize/crop còn lại <10% diện tích gốc sẽ bị loại bỏ
        )
    )

# Augmentation cho tập VALIDATION: KHÔNG có random flip/brightness,
# vì cần đánh giá model trên ảnh "sạch" (giống lúc inference thật) để so sánh công bằng giữa các epoch.
def get_valid_transforms(image_size: int = 1024):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )