"""
src/train.py
-------------
Script huấn luyện chính (CLI-ready) cho bài toán Global Wheat Detection.

Chạy trên Kaggle Notebook (GPU P100/T4x2):
    python -m src.train --config configs/kaggle_config.yaml

Chạy thử nhanh ở local (CPU, vài batch, xem README.md để biết cách chuẩn bị data):
    python -m src.train --config configs/local_config.yaml

Vì script được chạy bằng `python -m src.train` (chạy như MỘT MODULE trong package
`src`, không phải chạy file rời), Python tự thêm thư mục gốc project vào sys.path
-> các import dạng `from src.xxx import ...` luôn hoạt động đúng mà KHÔNG cần
chỉnh sys.path thủ công, miễn là lệnh được gọi từ thư mục gốc project (nơi có
thư mục `src/`).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.dataset import (
    WheatDataset,
    get_train_dataloader,
    get_train_transforms,
    get_val_dataloader,
    get_valid_transforms,
)
from src.model import create_model
from src.utils import LossMeter, load_config, save_checkpoint, set_seed

# Console mặc định trên Windows (vd cp1258, cp1252) không encode được tiếng Việt có
# dấu -> print() sẽ crash với UnicodeEncodeError. Ép stdout/stderr sang UTF-8 ngay
# khi import module này để toàn bộ log tiếng Việt bên dưới hiển thị đúng trên mọi
# terminal (Windows lẫn Linux/Kaggle). `reconfigure` chỉ có ở Python 3.7+ và có thể
# không tồn tại khi stdout bị redirect sang một stream đặc biệt -> bọc try/except.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def create_fold_split(
    df: pd.DataFrame, n_splits: int, fold: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia train/val bằng Stratified K-Fold, chia theo TỪNG ẢNH DUY NHẤT (image_id)
    chứ không theo từng dòng CSV — tránh việc 1 ảnh có nhiều bbox bị xé lẻ, vừa
    lọt vào train vừa lọt vào val (data leakage).

    Stratify theo cột "source" (nguồn ảnh: arvalis_1, arvalis_2, ethz_1, inrae_1,
    rres_1, usask_1, ...) để tập train và tập val có tỉ lệ các nguồn ảnh tương tự
    nhau — mỗi nguồn có đặc điểm ánh sáng/mật độ bông lúa khác nhau, nếu chia ngẫu
    nhiên thuần tuý có thể khiến val set lệch phân phối so với train set.

    Args:
        df: toàn bộ DataFrame train.csv (nhiều dòng/ảnh).
        n_splits: tổng số fold.
        fold: chỉ số fold (0-indexed) được dùng làm tập VALIDATION, các fold còn
            lại dùng làm tập TRAIN.
        seed: random_state để việc chia fold tái lập được giữa các lần chạy.

    Returns:
        Tuple (train_df, val_df), mỗi DataFrame giữ nguyên toàn bộ cột gốc và đã
        `reset_index`.
    """
    # Mỗi ảnh chỉ có 1 "source" duy nhất -> lấy giá trị đầu tiên đại diện cho ảnh đó.
    image_meta = df.groupby("image_id")["source"].first().reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    image_meta["fold"] = -1
    for fold_idx, (_, val_indices) in enumerate(
        skf.split(image_meta["image_id"], image_meta["source"])
    ):
        image_meta.loc[val_indices, "fold"] = fold_idx

    val_ids = image_meta.loc[image_meta["fold"] == fold, "image_id"]
    train_ids = image_meta.loc[image_meta["fold"] != fold, "image_id"]

    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Chạy 1 epoch training đầy đủ (forward + backward + cập nhật trọng số).

    Args:
        model: Faster R-CNN model (xem `src/model.py`).
        optimizer: optimizer đã gắn `model.parameters()`.
        scaler: `torch.cuda.amp.GradScaler` dùng để scale loss khi train ở FP16,
            tránh underflow gradient. Nếu `use_amp=False`, scaler chạy ở chế độ
            "disabled" và hoạt động tương đương training FP32 bình thường.
        data_loader: DataLoader của tập train (xem `get_train_dataloader`).
        device: "cuda" hoặc "cpu".
        use_amp: bật/tắt Automatic Mixed Precision.

    Returns:
        Dict các loss trung bình trong epoch, gồm "total_loss" và từng thành phần
        (loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg).
    """
    model.train()  # bật train mode: BatchNorm/Dropout hoạt động khác lúc eval
    loss_meter = LossMeter()

    pbar = tqdm(data_loader, desc="Training", leave=False)
    for images, targets in pbar:
        # images/targets đang là tuple (do collate_fn) -> chuyển từng phần tử lên
        # device thủ công, không thể gọi .to(device) trên cả batch một lần như CNN
        # phân loại thông thường.
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # autocast tự động chọn phép tính nào chạy ở FP16 (nhanh hơn, ít VRAM hơn)
        # và phép tính nào cần giữ FP32 (để ổn định số học) — chỉ có tác dụng khi
        # chạy trên GPU hỗ trợ Tensor Core (P100 hỗ trợ hạn chế, T4 hỗ trợ tốt).
        with autocast(enabled=use_amp):
            # Điểm đặc biệt của torchvision detection model: khi ở train() mode và
            # có truyền targets, forward() KHÔNG trả về prediction mà trả về dict
            # các loss thành phần (phân loại, box regression, objectness, RPN box).
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # scaler.scale nhân loss với 1 hệ số lớn trước khi backward để gradient FP16
        # không bị underflow về 0, sau đó scaler.step tự "unscale" lại trước khi
        # optimizer.step(); khi use_amp=False, các lệnh này tương đương gọi thường.
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss_dict, batch_size=len(images))
        pbar.set_postfix({"loss": f"{loss_meter.averages['total_loss']:.4f}"})

    return loss_meter.averages


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """
    Chạy 1 epoch validation, CHỈ tính loss, không cập nhật trọng số.

    LƯU Ý QUAN TRỌNG (điểm rất dễ gây lỗi khi tự viết pipeline detection):
    Các model detection của torchvision (Faster R-CNN, RetinaNet, ...) CHỈ trả về
    dict loss khi đang ở `train()` mode VÀ có truyền `targets`. Ở `eval()` mode,
    model trả về danh sách prediction (boxes/scores/labels) thay vì loss, nên
    KHÔNG thể "tính val loss" bằng cách gọi `model.eval()` như các model phân loại
    thông thường.

    Giải pháp chuẩn, được dùng phổ biến trong các solution Kaggle cho bài toán này:
    vẫn giữ `model.train()` để lấy được loss dict, nhưng bọc toàn bộ trong
    `torch.no_grad()` để không tính gradient và không cập nhật trọng số -> vẫn đúng
    bản chất "chỉ đánh giá, không học". Điều này an toàn vì backbone ResNet-FPN của
    torchvision dùng `FrozenBatchNorm2d` (không cập nhật running stats theo mode
    train/eval), và đầu box (TwoMLPHead) không có Dropout.

    Args:
        model: Faster R-CNN model.
        data_loader: DataLoader của tập validation (xem `get_val_dataloader`).
        device: "cuda" hoặc "cpu".
        use_amp: bật/tắt autocast (không cần GradScaler vì không backward).

    Returns:
        Dict các loss trung bình trên toàn bộ tập validation.
    """
    model.train()
    loss_meter = LossMeter()

    pbar = tqdm(data_loader, desc="Validating", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)

        loss_meter.update(loss_dict, batch_size=len(images))
        pbar.set_postfix({"val_loss": f"{loss_meter.averages['total_loss']:.4f}"})

    return loss_meter.averages


def build_datasets(cfg: Dict[str, Any]) -> Tuple[WheatDataset, WheatDataset]:
    """
    Đọc train.csv, chia fold, và dựng 2 `WheatDataset` (train/val) với augmentation
    tương ứng cho từng tập.

    Args:
        cfg: dict config đã load từ YAML (xem configs/kaggle_config.yaml).

    Returns:
        Tuple (train_dataset, val_dataset).
    """
    df = pd.read_csv(cfg["csv_path"])

    train_df, val_df = create_fold_split(
        df, n_splits=cfg["n_splits"], fold=cfg["fold"], seed=cfg["seed"]
    )
    print(
        f"[INFO] Fold {cfg['fold']}/{cfg['n_splits']} "
        f"- Train: {train_df['image_id'].nunique()} ảnh "
        f"- Val: {val_df['image_id'].nunique()} ảnh"
    )

    # Train dùng augment mạnh (get_train_transforms), val dùng transform "sạch"
    # (get_valid_transforms) để đánh giá gần giống điều kiện inference thật.
    train_dataset = WheatDataset(
        train_df, cfg["data_dir"], transforms=get_train_transforms(cfg["image_size"])
    )
    val_dataset = WheatDataset(
        val_df, cfg["data_dir"], transforms=get_valid_transforms(cfg["image_size"])
    )
    return train_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Faster R-CNN cho Global Wheat Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_config.yaml",
        help="Đường dẫn tới file config YAML (vd: configs/kaggle_config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Đã load config: {args.config}")

    # Cố định seed TRƯỚC khi chia fold/tạo model để đảm bảo kết quả tái lập được.
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sử dụng device: {device}")

    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    if cfg.get("use_amp", False) and device.type != "cuda":
        print("[WARN] use_amp=True nhưng không có GPU -> tự động tắt AMP (AMP chỉ có tác dụng trên CUDA).")

    train_dataset, val_dataset = build_datasets(cfg)

    train_loader = get_train_dataloader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        pin_memory=cfg.get("pin_memory", True),
    )
    val_loader = get_val_dataloader(
        val_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=cfg.get("pin_memory", True),
    )

    model = create_model(
        num_classes=cfg.get("num_classes", 2), pretrained=cfg.get("pretrained", True)
    ).to(device)
    # requires_grad=True lọc ra các layer đang cho phép train (phòng trường hợp sau
    # này tự đóng băng/freeze một phần backbone).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["learning_rate"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 0.0005),
    )
    # enabled=use_amp: khi tắt AMP, GradScaler hoạt động như một no-op trong suốt
    # (scale factor = 1), nên code training không cần rẽ nhánh if/else riêng.
    scaler = GradScaler(enabled=use_amp)

    output_dir = cfg["output_dir"]
    best_model_path = os.path.join(output_dir, "best_model.pth")
    last_model_path = os.path.join(output_dir, "last_model.pth")

    best_val_loss = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{cfg['num_epochs']} ---")

        train_losses = train_one_epoch(model, optimizer, scaler, train_loader, device, use_amp)
        print("[TRAIN] " + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()))

        val_losses = validate_one_epoch(model, val_loader, device, use_amp)
        print("[VAL]   " + " | ".join(f"{k}: {v:.4f}" for k, v in val_losses.items()))

        val_loss = val_losses["total_loss"]
        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_losses["total_loss"],
            "val_loss": val_loss,
            "config": cfg,
        }

        # Luôn lưu "last_model.pth" để có thể resume/kiểm tra epoch gần nhất.
        save_checkpoint(checkpoint_state, last_model_path)

        # Chỉ ghi đè "best_model.pth" khi val_loss cải thiện — đây mới là model nên
        # dùng để inference/submit, vì được chọn dựa trên tập KHÔNG dùng để train.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_state, best_model_path)
            print(f"[INFO] Val loss cải thiện -> best_val_loss = {best_val_loss:.4f}")

    print(f"\n[DONE] Huấn luyện hoàn tất. Best val_loss = {best_val_loss:.4f}")
    print(f"[DONE] Model tốt nhất: {best_model_path}")
    print(f"[DONE] Model epoch cuối: {last_model_path}")


if __name__ == "__main__":
    main()
