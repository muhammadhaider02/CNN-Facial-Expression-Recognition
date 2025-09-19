import argparse
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt

from FER_transform import FERDataset, build_transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FERResNet50(nn.Module):
    def __init__(self, vggface2_ckpt: str | None = None, device: torch.device | None = None):
        super().__init__()
        if vggface2_ckpt:
            backbone = models.resnet50(weights=None)
            self._load_vggface2_weights(backbone, vggface2_ckpt, device=device)
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # type: ignore[assignment]

        self.backbone = backbone
        self.dropout = nn.Dropout(0.7)  # increased dropout for stronger regularization
        self.fc_cls = nn.Linear(in_features, 8)
        self.fc_reg = nn.Linear(in_features, 2)

    @staticmethod
    def _load_vggface2_weights(backbone: nn.Module, path: str, device: torch.device | None = None):
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Convert numpy arrays to torch tensors if necessary
        for k, v in list(state.items()):
            if isinstance(v, np.ndarray):
                state[k] = torch.from_numpy(v)

        new_state = {}
        bb_state = backbone.state_dict()
        for k, v in state.items():
            if k in bb_state and not k.startswith("fc."):
                new_state[k] = v

        # Report coverage and detect potential shape mismatches
        bb_keys = [k for k in bb_state.keys() if not k.startswith("fc.")]
        loaded_keys = list(new_state.keys())
        coverage = (len(set(loaded_keys)) / max(len(bb_keys), 1)) * 100.0

        # Detect shape mismatches pre-load (defensive; strict load will also surface these)
        mismatched = [
            k for k in loaded_keys
            if k in bb_state and hasattr(new_state[k], 'shape') and hasattr(bb_state[k], 'shape')
            and tuple(new_state[k].shape) != tuple(bb_state[k].shape)
        ]

        print(f"[VGGFace2] Backbone param coverage: {len(set(loaded_keys))}/{len(bb_keys)} ({coverage:.1f}%)")
        # Load non-strict after removing any shape-mismatched keys
        for k in mismatched:
            new_state.pop(k, None)

        backbone.load_state_dict(new_state, strict=False)

        # Overlay ImageNet weights for remaining backbone params that were not covered
        try:
            imagenet_sd = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).state_dict()
        except Exception as e:
            print("[ImageNet] Failed to fetch ImageNet weights:", e)
            return

        vgg_loaded = set(new_state.keys()) - set(mismatched)
        remaining_keys = [k for k in bb_keys if k not in vgg_loaded]
        im_overlay = {}
        for k in remaining_keys:
            if k in imagenet_sd and hasattr(imagenet_sd[k], 'shape') and hasattr(bb_state[k], 'shape') \
               and tuple(imagenet_sd[k].shape) == tuple(bb_state[k].shape):
                im_overlay[k] = imagenet_sd[k]

        coverage_im = (len(im_overlay) / max(len(remaining_keys), 1)) * 100.0
        print(f"[ImageNet] Remaining Backbone params coverage: {len(im_overlay)}/{len(remaining_keys)} ({coverage_im:.1f}%)")

        if im_overlay:
            backbone.load_state_dict(im_overlay, strict=False)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.fc_cls(feats), self.fc_reg(feats)


def set_trainable(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def apply_freezing_policy(model: FERResNet50, epoch: int, freeze_epochs: int, unfreeze_all_at: int | None):
    set_trainable(model.fc_cls, True)
    set_trainable(model.fc_reg, True)

    if unfreeze_all_at is not None and (epoch + 1) >= unfreeze_all_at:
        set_trainable(model.backbone, True)
        return

    if epoch < freeze_epochs:
        set_trainable(model.backbone, False)
        set_trainable(model.backbone.layer4, True)
    else:
        set_trainable(model.backbone.conv1, False)
        set_trainable(model.backbone.bn1, False)
        set_trainable(model.backbone.layer1, False)
        set_trainable(model.backbone.layer2, False)
        set_trainable(model.backbone.layer3, True)
        set_trainable(model.backbone.layer4, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta", type=str, default="metadata_train.parquet")
    parser.add_argument("--val_meta", type=str, default="metadata_val.parquet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--vggface2_ckpt", type=str, default="weight/resnet50_ft_weight.pth")
    parser.add_argument("--unfreeze_all_at", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="FER_vggface2_resnet50.pth")
    args = parser.parse_args()

    set_seed(42)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # Data
    train_ds = FERDataset(args.train_meta, train=True)
    val_ds = FERDataset(args.val_meta, train=False)
    train_ds.transforms = build_transforms(train=True, size=args.img_size)
    val_ds.transforms = build_transforms(train=False, size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = FERResNet50(vggface2_ckpt=args.vggface2_ckpt, device=DEVICE).to(DEVICE)

    # Loss & Optimizer
    cls_counts = np.bincount(train_ds.df["expression"].astype(int), minlength=8)
    cls_weights = torch.tensor(1.0 / (cls_counts + 1e-6), dtype=torch.float32)
    cls_weights = (cls_weights / cls_weights.sum() * 8).to(DEVICE)

    criterion_cls = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.1)
    criterion_reg = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # Training loop
    EPOCHS = args.epochs
    FREEZE_EPOCHS = args.freeze_epochs
    PATIENCE = 15

    best_val_acc = 0.0
    best_val_rmse = float('inf')
    bad_epochs = 0

    # For training graphs
    history_train_loss: list[float] = []
    history_train_acc: list[float] = []
    history_val_loss: list[float] = []
    history_val_acc: list[float] = []
    history_val_rmse: list[float] = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        apply_freezing_policy(model, epoch, FREEZE_EPOCHS, args.unfreeze_all_at)

        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch in train_dl:
            imgs = batch["image"].to(DEVICE)
            y_cls = batch["y_cls"].to(DEVICE)
            y_reg = batch["y_reg"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(enabled=(DEVICE.type == "cuda")):
                    out_cls, out_reg = model(imgs)
                    loss_cls = criterion_cls(out_cls, y_cls)
                    loss_reg = criterion_reg(out_reg, y_reg)
                    loss = loss_cls + 0.5 * loss_reg  # Combined loss for both tasks

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA out of memory. Try reducing --batch_size or --img_size.")
                    sys.exit(1)
                else:
                    raise

            train_loss += loss.item()
            _, preds = out_cls.max(1)
            correct += preds.eq(y_cls).sum().item()
            total += y_cls.size(0)

        train_acc = 100.0 * correct / max(total, 1)
        scheduler.step()

        # Track train metrics per epoch
        history_train_loss.append(train_loss / max(len(train_dl), 1))
        history_train_acc.append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_rmse = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(DEVICE)
                y_cls = batch["y_cls"].to(DEVICE)
                y_reg = batch["y_reg"].to(DEVICE)
                out_cls, out_reg = model(imgs)
                loss_cls = criterion_cls(out_cls, y_cls)
                loss_reg = criterion_reg(out_reg, y_reg)
                loss = loss_cls + 0.5 * loss_reg

                val_loss += loss.item()
                _, preds = out_cls.max(1)
                val_correct += preds.eq(y_cls).sum().item()
                val_total += y_cls.size(0)

                # Calculate RMSE for regression
                val_rmse += np.sqrt(((y_reg - out_reg) ** 2).mean().item())

        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_rmse /= len(val_dl)

        # Track val metrics per epoch
        history_val_loss.append(val_loss / max(len(val_dl), 1))
        history_val_acc.append(val_acc)
        history_val_rmse.append(val_rmse)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss/len(train_dl):.4f} | Train Acc: {train_acc:.2f}% "
            f"|| Val Loss: {val_loss/len(val_dl):.4f} | Val Acc: {val_acc:.2f}% | RMSE: {val_rmse:.4f}"
        )

        # Save the best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model at epoch {epoch+1} with Val Acc {val_acc:.2f}%")

        # Early stopping based on both validation accuracy and regression RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete. Best Val Acc:", best_val_acc)
    print("Best RMSE:", best_val_rmse)

    # Total time tracking
    elapsed = time.time() - start_time
    hhmmss = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"Total training time: {hhmmss} ({elapsed:.1f}s)")

    # Save training graphs
    try:
        epochs_range = range(1, len(history_train_loss) + 1)
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

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

        # RMSE (validation)
        axes[2].plot(epochs_range, history_val_rmse, label="Val RMSE")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("RMSE")
        axes[2].set_title("Validation RMSE over epochs")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig("TrainingGraph_vggface2_resnet50.png")
        plt.close(fig)
        print("Saved training graph to TrainingGraph_vggface2_resnet50.png")
    except Exception as e:
        print("Failed to save training graph:", e)


if __name__ == "__main__":
    main()