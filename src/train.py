import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils import load_config, save_checkpoint, AverageMeter
from src.dataset import WheatDataset, collate_fn
from src.transforms import get_train_transforms, get_valid_transforms
from src.model import create_model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()  # bật train mode: BatchNorm/Dropout hoạt động khác lúc eval
    loss_meter = AverageMeter()

    pbar = tqdm(data_loader, desc="Training", leave=False)
    for images, targets in pbar:
        # images/targets đang là tuple (do collate_fn) -> chuyển từng phần tử lên GPU/CPU thủ công,
        # không thể gọi .to(device) trên cả batch như CNN phân loại thông thường.
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Điểm đặc biệt của torchvision detection model: khi ở train() mode và có truyền targets,
        # forward() KHÔNG trả về prediction mà trả về dict các loss thành phần
        # (loss phân loại, loss regression box, loss objectness, loss RPN box...).
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())  # cộng tất cả loss thành 1 số để backward

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_meter.update(losses.item(), len(images))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})  # hiện loss trung bình ngay trên thanh tqdm

    return loss_meter.avg

def main():
    parser = argparse.ArgumentParser()
    # --config quyết định chạy local (nhẹ, thử pipeline) hay kaggle (train thật) mà không sửa code.
    parser.add_argument("--config", type=str, default="configs/local_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Loaded config: {args.config}")

    # Tự động dùng GPU nếu có (Kaggle), fallback về CPU nếu chạy local không có GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    df = pd.read_csv(cfg["csv_path"])
    unique_ids = df["image_id"].unique()

    # Chia theo image_id (không chia theo dòng CSV) để tránh 1 ảnh vừa lọt vào train vừa lọt vào val
    # (data leakage) — random_state=42 để lần chạy nào cũng chia giống nhau, dễ so sánh kết quả.
    train_ids, val_ids = train_test_split(unique_ids, test_size=cfg["val_split"], random_state=42)
    train_df = df[df["image_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_ids)].reset_index(drop=True)

    # Train dùng augment mạnh (get_train_transforms), val dùng transform "sạch" (get_valid_transforms).
    train_dataset = WheatDataset(train_df, cfg["data_dir"], transforms=get_train_transforms(cfg["image_size"]))
    val_dataset = WheatDataset(val_df, cfg["data_dir"], transforms=get_valid_transforms(cfg["image_size"]))
    # Lưu ý: val_dataset hiện được tạo nhưng CHƯA có vòng lặp đánh giá (evaluate) nào dùng tới nó bên dưới
    # -> best_model đang chọn theo train_loss, không phải val_loss (train_loss thấp chưa chắc model tốt/không overfit).

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,             # xáo trộn thứ tự ảnh mỗi epoch để tránh model học theo thứ tự dữ liệu
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn     # bắt buộc phải truyền, nếu không DataLoader sẽ lỗi khi stack targets
    )

    model = create_model(num_classes=2).to(device)
    # requires_grad=True lọc ra các layer đang cho phép train (phòng trường hợp sau này tự đóng băng/freeze backbone).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["learning_rate"], momentum=0.9, weight_decay=0.0005)

    best_loss = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{cfg['num_epochs']} ---")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch} - Average Loss: {train_loss:.4f}")

        checkpoint_path = os.path.join(cfg["output_dir"], f"model_epoch_{epoch}.pth")  # (biến này hiện chưa được dùng để save)
        if train_loss < best_loss:
            best_loss = train_loss
            # Chỉ lưu duy nhất "best_model.pth" (ghi đè mỗi lần loss cải thiện), không lưu từng epoch riêng.
            save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "loss": train_loss},
                            os.path.join(cfg["output_dir"], "best_model.pth"))

if __name__ == "__main__":
    main()