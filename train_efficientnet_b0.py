import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from FER_transform import FERDataset, build_transforms
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt

# 1. Config
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 2. Data
train_ds = FERDataset("metadata_train.parquet", train=True)
val_ds   = FERDataset("metadata_val.parquet", train=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Model (EfficientNet-B0 with dropout)
class FEREfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.3)
        self.fc_cls = nn.Linear(in_features, 8)
        self.fc_reg = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.fc_cls(feats), self.fc_reg(feats)

model = FEREfficientNetB0().to(DEVICE)

# 4. Loss & Optimizer
cls_counts = np.bincount(train_ds.df["expression"].astype(int), minlength=8)
cls_weights = torch.tensor(1.0 / (cls_counts + 1e-6), dtype=torch.float32)
cls_weights = (cls_weights / cls_weights.sum() * 8).to(DEVICE)

criterion_cls = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.05)
criterion_reg = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = GradScaler()

# 5. EMA helper
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

ema = EMA(model)

# 6. Training loop
best_val_acc = 0.0
bad_epochs = 0

# For training graphs
history_train_loss = []
history_train_acc = []
history_val_loss = []
history_val_acc = []

start_time = time.time()

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    train_loss, correct, total = 0, 0, 0

    for batch in train_dl:
        imgs, y_cls, y_reg = batch["image"].to(DEVICE), batch["y_cls"].to(DEVICE), batch["y_reg"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(DEVICE.type == "cuda")):
            out_cls, out_reg = model(imgs)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss = loss_cls + 0.5 * loss_reg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        train_loss += loss.item()
        _, preds = out_cls.max(1)
        correct += preds.eq(y_cls).sum().item()
        total += y_cls.size(0)

    train_acc = 100. * correct / total
    scheduler.step()

    # Track train metrics per epoch
    history_train_loss.append(train_loss / max(len(train_dl), 1))
    history_train_acc.append(train_acc)

    # --- Validation ---
    ema.apply_shadow()
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

    # Track val metrics per epoch
    history_val_loss.append(val_loss / max(len(val_dl), 1))
    history_val_acc.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_dl):.4f} | Train Acc: {train_acc:.2f}% "
          f"|| Val Loss: {val_loss/len(val_dl):.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc, bad_epochs = val_acc, 0
        torch.save(model.state_dict(), "FER_efficientnet_b0.pth")
        print(f"Saved new best model at epoch {epoch+1} with Val Acc {val_acc:.2f}%")
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete. Best Val Acc:", best_val_acc)

# Total time tracking
elapsed = time.time() - start_time
hhmmss = time.strftime("%H:%M:%S", time.gmtime(elapsed))
print(f"Total training time: {hhmmss} ({elapsed:.1f}s)")

# Save training graphs
try:
    epochs_range = range(1, len(history_train_loss) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Losses
    axes[0].plot(epochs_range, history_train_loss, label="Train Loss")
    axes[0].plot(epochs_range, history_val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over epochs")
    axes[0].legend()

    # Accuracies
    axes[1].plot(epochs_range, history_train_acc, label="Train Acc (%)")
    axes[1].plot(epochs_range, history_val_acc, label="Val Acc (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy over epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("TrainingGraph_efficientnet_b0.png")
    plt.close(fig)
    print("Saved training graph to TrainingGraph_efficientnet_b0.png")
except Exception as e:
    print("Failed to save training graph:", e)