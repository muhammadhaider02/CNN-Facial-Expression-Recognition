import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from FER_transform import FERDataset
import numpy as np
from torch.amp import autocast, GradScaler

# 1. Config
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 2. Data
train_ds = FERDataset("metadata_train.parquet", train=True)
val_ds   = FERDataset("metadata_val.parquet", train=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Model (ResNet34)
class FERResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_cls = nn.Linear(in_features, 8)  # 8 emotions
        self.fc_reg = nn.Linear(in_features, 2)  # valence, arousal

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc_cls(feats), self.fc_reg(feats)

model = FERResNet34().to(DEVICE)

# 4. Loss & Optimizer
cls_counts = np.bincount(train_ds.df["expression"].astype(int), minlength=8)
cls_weights = torch.tensor(1.0 / (cls_counts + 1e-6), dtype=torch.float32)
cls_weights = (cls_weights / cls_weights.sum() * 8).to(DEVICE)

criterion_cls = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.05)
criterion_reg = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# 5. Training loop
best_val_acc = 0.0
bad_epochs = 0

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss, correct, total = 0, 0, 0

    for batch in train_dl:
        imgs, y_cls, y_reg = batch["image"].to(DEVICE), batch["y_cls"].to(DEVICE), batch["y_reg"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            out_cls, out_reg = model(imgs)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss = loss_cls + 0.5 * loss_reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, preds = out_cls.max(1)
        correct += preds.eq(y_cls).sum().item()
        total += y_cls.size(0)

    train_acc = 100. * correct / total
    scheduler.step()

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in val_dl:
            imgs, y_cls, y_reg = batch["image"].to(DEVICE), batch["y_cls"].to(DEVICE), batch["y_reg"].to(DEVICE)
            out_cls, out_reg = model(imgs)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss = loss_cls + 0.5 * loss_reg
            val_loss += loss.item()
            _, preds = out_cls.max(1)
            val_correct += preds.eq(y_cls).sum().item()
            val_total += y_cls.size(0)

    val_acc = 100. * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_dl):.4f} | Train Acc: {train_acc:.2f}% "
          f"|| Val Loss: {val_loss/len(val_dl):.4f} | Val Acc: {val_acc:.2f}%")

    # --- Early stopping & checkpoint ---
    if val_acc > best_val_acc:
        best_val_acc, bad_epochs = val_acc, 0
        torch.save(model.state_dict(), "FER_resnet34.pth")
        print(f"Saved new best model at epoch {epoch+1} with Val Acc {val_acc:.2f}%")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete. Best Val Acc:", best_val_acc)