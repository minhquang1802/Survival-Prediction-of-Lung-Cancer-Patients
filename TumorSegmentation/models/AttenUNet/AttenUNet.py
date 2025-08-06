import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose, LoadImaged, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, EnsureChannelFirstd, DivisiblePadd
)
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import time
from datetime import datetime

# === Setup path ===
root_dir = "/mrhung_nguyen_minh_quang_108/workspace/train/nnUNet_raw/Dataset015_lungTumor"
images = sorted(glob.glob(os.path.join(root_dir, "imagesTr", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(root_dir, "labelsTr", "*.nii.gz")))

model_dir = "./saved_models"
log_dir = "./runs/AttenUNet_lung"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "training_log.txt")
def log_to_file(message, file_path=log_file):
    with open(file_path, "a") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

# === Resume from checkpoint if exists ===
checkpoint_path = os.path.join(model_dir, "last_checkpoint.pth")
start_epoch = 0
best_metric = -1
best_metric_epoch = -1

writer = SummaryWriter(log_dir=log_dir)
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

val_frac = 0.2
val_split = int(len(data_dicts) * (1 - val_frac))
train_files = data_dicts[:val_split]
val_files = data_dicts[val_split:]

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    DivisiblePadd(keys=["image", "label"], k=16),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96),
        pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    DivisiblePadd(keys=["image", "label"], k=16),
    EnsureTyped(keys=["image", "label"]),
])

if __name__ == "__main__":
    start_time = time.time()

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, persistent_workers=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint.get("best_metric", -1)
        best_metric_epoch = checkpoint.get("best_metric_epoch", -1)
        print(f"Resumed training from epoch {start_epoch}")
        log_to_file(f"Resumed training from epoch {start_epoch}")

    max_epochs = 1000
    val_interval = 1
    patience = 300
    patience_counter = 0

    for epoch in range(start_epoch, max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        log_to_file(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            step += 1

        epoch_loss /= step
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        log_to_file(f"Train Loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                val_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    with torch.cuda.amp.autocast():
                        val_outputs = torch.sigmoid(model(val_inputs))
                        loss = loss_function(val_outputs, val_labels)
                    val_loss += loss.item()
                    val_outputs = (val_outputs > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_step += 1
                val_metric = dice_metric.aggregate().item()
                dice_metric.reset()

                writer.add_scalar("Loss/val", val_loss / val_step, epoch)
                writer.add_scalar("Dice/val", val_metric, epoch)
                log_to_file(f"Val Loss: {val_loss / val_step:.4f}, Val Dice: {val_metric:.4f}")

                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    print(f"Saved new best model at epoch {epoch+1} with Dice: {val_metric:.4f}")
                    log_to_file(f"Saved best model at epoch {epoch+1} with Dice: {val_metric:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                    log_to_file(f"No improvement. Patience: {patience_counter}/{patience}")

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch
                }, checkpoint_path)
                log_to_file(f"Checkpoint saved at epoch {epoch+1}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            log_to_file(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    log_to_file(f"Training complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    writer.close()