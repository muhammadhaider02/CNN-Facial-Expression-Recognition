import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from FER_transform import FERDataset
from torchvision import models

# 1. Config
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]

# 2. Dataset
val_ds = FERDataset("metadata_val.parquet", train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Model (ResNet34)
class FERResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.fc_cls = nn.Linear(in_features, 8)
        self.fc_reg = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc_cls(feats), self.fc_reg(feats)

model = FERResNet34().to(DEVICE)
state_dict = torch.load("FER_resnet34.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# 4. Evaluation
y_true, y_pred = [], []
y_valence, y_arousal, y_valence_pred, y_arousal_pred = [], [], [], []

with torch.no_grad():
    for batch in val_dl:
        imgs, labels, reg_targets = batch["image"].to(DEVICE), batch["y_cls"].to(DEVICE), batch["y_reg"].to(DEVICE)
        out_cls, out_reg = model(imgs)

        # Classification
        preds = out_cls.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        # Regression (valence, arousal)
        y_valence.extend(reg_targets[:,0].cpu().numpy())
        y_arousal.extend(reg_targets[:,1].cpu().numpy())
        y_valence_pred.extend(out_reg[:,0].cpu().numpy())
        y_arousal_pred.extend(out_reg[:,1].cpu().numpy())

# 5. Metrics
print("Classification Report (Val set):")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
print(f"Overall Accuracy: {acc:.4f}")
print(f"Macro F1-score: {f1_macro:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ResNet34 Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig("ConfusionMatrix_resnet34.png")
plt.show()

# Regression metrics
rmse_val = root_mean_squared_error(y_valence, y_valence_pred)
rmse_aro = root_mean_squared_error(y_arousal, y_arousal_pred)
mae_val = mean_absolute_error(y_valence, y_valence_pred)
mae_aro = mean_absolute_error(y_arousal, y_arousal_pred)

print(f"Valence RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
print(f"Arousal RMSE: {rmse_aro:.4f}, MAE: {mae_aro:.4f}")

# 6. Additional required metrics

def _krippendorff_alpha_nominal(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    obs_disagree = np.mean(y_true_arr != y_pred_arr)
    pooled = np.concatenate([y_true_arr, y_pred_arr])
    values, counts = np.unique(pooled, return_counts=True)
    p = counts.astype(float) / pooled.size
    exp_disagree = 1.0 - np.sum(p ** 2)
    if exp_disagree == 0.0:
        return 1.0 if obs_disagree == 0.0 else float("nan")
    return 1.0 - (obs_disagree / exp_disagree)

def _pearson_corr(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
        return float("nan")
    return float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])

def _sagr(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr)))

def _ccc(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    x = y_true_arr.astype(float)
    y = y_pred_arr.astype(float)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0.0:
        return float("nan")
    return float((2.0 * cov) / denom)

alpha = _krippendorff_alpha_nominal(np.array(y_true), np.array(y_pred))
print(f"Krippendorff's Alpha (nominal): {alpha:.4f}")

# Pearson Correlation, SAGR, and CCC for valence and arousal
val_corr = _pearson_corr(np.array(y_valence), np.array(y_valence_pred))
aro_corr = _pearson_corr(np.array(y_arousal), np.array(y_arousal_pred))
val_sagr = _sagr(np.array(y_valence), np.array(y_valence_pred))
aro_sagr = _sagr(np.array(y_arousal), np.array(y_arousal_pred))
val_ccc = _ccc(np.array(y_valence), np.array(y_valence_pred))
aro_ccc = _ccc(np.array(y_arousal), np.array(y_arousal_pred))

print(f"Valence CORR (Pearson): {val_corr:.4f}")
print(f"Arousal CORR (Pearson): {aro_corr:.4f}")
print(f"Valence SAGR: {val_sagr:.4f}")
print(f"Arousal SAGR: {aro_sagr:.4f}")
print(f"Valence CCC: {val_ccc:.4f}")
print(f"Arousal CCC: {aro_ccc:.4f}")