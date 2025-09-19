## FER Pipeline Report

### Data indexing: metadata_parallel.py
- Scans `images/*.jpg` and pairs each image with annotation `.npy` files from `annotations/`:
  - `<name>_exp.npy` (required, int class 0–7)
  - `<name>_val.npy`, `<name>_aro.npy` (optional floats)
  - `<name>_lnd.npy` (optional landmarks)
- Builds a row per image and writes:
  - `metadata_all.parquet` (full index)
  - `metadata_train.parquet` and `metadata_val.parquet` via stratified split on expression (val fraction 0.2, seed 42)
- Parallelized with ThreadPoolExecutor; prints progress every 1000 files.

### Dataset and transforms: FER_transform.py
- `FERDataset` loads a parquet split, filters rows with valid valence/arousal (not NaN, > -2), then returns:
  - image tensor (RGB, normalized ImageNet stats)
  - `y_cls` (long, 0–7)
  - `y_reg` (float32, [valence, arousal])
- Augmentations (train): RandomResizedCrop, HorizontalFlip, mild ColorJitter, Affine (translate/scale/rotate), CoarseDropout, ISONoise, Median/Gaussian blur, CLAHE, Normalize, ToTensorV2.
- Validation: Resize → Normalize → ToTensorV2.

## Training: train_resnet50.py

### Model architecture
- Backbone: `torchvision.models.resnet50` with `fc` replaced by `Identity` to expose a pooled feature vector.
- Regularization: `Dropout(p=0.7)` on features.
- Heads (multitask):
  - Classification head: `Linear(in_features, 8)` (8 expressions).
  - Regression head: `Linear(in_features, 2)` (valence, arousal).

### Initialization
- If `--vggface2_ckpt` provided (default `weight/resnet50_ft_weight.pth`): loads compatible weights into the backbone (skips `fc.*`).
- Otherwise uses ImageNet weights `ResNet50_Weights.IMAGENET1K_V1`.

### Freezing policy (staged fine-tuning)
- Epoch < `--freeze_epochs` (default 3): freeze backbone except `layer4`; heads always trainable.
- After that: `conv1`, `bn1`, `layer1`, `layer2` remain frozen; `layer3` and `layer4` train.
- `--unfreeze_all_at` (optional int epoch index, 1-based) fully unfreezes backbone at/after that epoch.

### Losses
- Classification: CrossEntropyLoss with label smoothing 0.1 and class weights computed as inverse frequency from training split.
- Regression: MSELoss.
- Total loss: `loss = loss_cls + 0.5 * loss_reg`.

### Optimization & schedule
- Optimizer: AdamW (`--lr` default 1e-4, `--weight_decay` default 1e-3).
- LR scheduler: CosineAnnealingLR with `T_max = --epochs`.
- Mixed precision: CUDA AMP + GradScaler.

### Data loading
- Transforms from `FER_transform.py` (size via `--img_size`, default 224).
- DataLoader: `batch_size=--batch_size` (default 8), `shuffle=True/False` for train/val, `num_workers=0` (Windows‑friendly).

### Metrics, checkpointing, early stopping
- Training prints per-epoch train loss/acc and val loss/acc/RMSE.
- Best model is selected by highest validation accuracy and saved to `--save_path` (default `FER_resnet50.pth`).
- Early stopping with patience 15 epochs: resets on improvement of either validation accuracy or regression RMSE; stops when neither improves for `patience` epochs.

### Key CLI arguments (defaults)
- `--train_meta metadata_train.parquet`
- `--val_meta metadata_val.parquet`
- `--epochs 100`
- `--freeze_epochs 3`
- `--batch_size 8`
- `--lr 1e-4`
- `--weight_decay 1e-3`
- `--img_size 224`
- `--vggface2_ckpt weight/resnet50_ft_weight.pth`
- `--unfreeze_all_at None`
- `--save_path FER_resnet50.pth`

### Output artifacts
- Checkpoint: `FER_resnet50.pth` (best by validation accuracy).
- Evaluation: `eval_resnet50.py` reports classification (Accuracy, F1, Kappa, AUC, PR‑AUC, Krippendorff’s Alpha) and regression (RMSE/MAE, Pearson CORR, SAGR, CCC) and saves `ConfusionMatrix_resnet50.png`.

## Training: train_efficientnet_b0.py

### Model architecture
- Backbone: `torchvision.models.efficientnet_b0` with `classifier` replaced by `Identity` to expose features.
- Regularization: `Dropout(p=0.3)` on features.
- Heads (multitask):
  - Classification head: `Linear(in_features, 8)` (8 expressions).
  - Regression head: `Linear(in_features, 2)` (valence, arousal).

### Initialization
- ImageNet weights `EfficientNet_B0_Weights.IMAGENET1K_V1`.

### Exponential Moving Average (EMA)
- Maintains a shadow copy of trainable parameters updated each step (`decay≈0.999`).
- Applies shadow weights for validation to stabilize metrics.

### Losses
- Classification: CrossEntropyLoss with label smoothing 0.05 and class weights (inverse frequency).
- Regression: MSELoss.
- Total loss: `loss = loss_cls + 0.5 * loss_reg`.

### Optimization & schedule
- Optimizer: AdamW (`LR=3e-4`, `weight_decay=1e-4`).
- LR scheduler: CosineAnnealingLR with `T_max = EPOCHS`.
- Mixed precision: CUDA AMP + GradScaler.

### Data loading
- Uses `FERDataset` transforms (default size 224 via file, flag not exposed here).
- DataLoader: `batch_size=16`, `num_workers=0`.

### Metrics, checkpointing, early stopping
- Prints per-epoch train loss/acc and val loss/acc (with EMA applied).
- Best checkpoint saved by highest validation accuracy to `FER_efficientnet_b0.pth`.
- Early stopping with patience 10 epochs.

### Output artifacts
- Checkpoint: `FER_efficientnet_b0.pth` (best by validation accuracy).
- Evaluation: `eval_efficientnet_b0.py` reports classification (Accuracy, F1, Kappa, AUC, PR‑AUC, Krippendorff’s Alpha) and regression (RMSE/MAE, Pearson CORR, SAGR, CCC) and saves `ConfusionMatrix_efficientnet_b0.png`.