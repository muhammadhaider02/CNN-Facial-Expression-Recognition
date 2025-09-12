import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from FER_transform import FERDataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# 1. Config
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]

# 2. Dataset
val_ds = FERDataset("metadata_val.parquet", train=False)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Model (EfficientNet-B0)
class FEREfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc_cls = nn.Linear(in_features, 8)   # 8 emotions
        self.fc_reg = nn.Linear(in_features, 2)   # valence, arousal

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc_cls(feats), self.fc_reg(feats)

model = FEREfficientNetB0().to(DEVICE)
state_dict = torch.load("FER_efficientnet_b0.pth", map_location=DEVICE, weights_only=True)
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
plt.title("EfficientNet-B0 Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig("ConfusionMatrix_efficientnet_b0.png")
plt.show()

# Regression metrics
rmse_val = root_mean_squared_error(y_valence, y_valence_pred)
rmse_aro = root_mean_squared_error(y_arousal, y_arousal_pred)
mae_val = mean_absolute_error(y_valence, y_valence_pred)
mae_aro = mean_absolute_error(y_arousal, y_arousal_pred)

print(f"Valence RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
print(f"Arousal RMSE: {rmse_aro:.4f}, MAE: {mae_aro:.4f}")