## Facial Emotion Recognition (FER)

This repository trains and evaluates a multitask ResNet‑50 model for 8‑class emotion classification and 2‑target regression (valence, arousal).

### Prerequisites
- Python 3.9+
- Recommended packages:
  - torch, torchvision
  - pandas, numpy, scikit-learn
  - albumentations, opencv-python, matplotlib, seaborn
  - pyarrow (or fastparquet) for Parquet I/O

Example install:
```bash
pip install torch torchvision pandas numpy scikit-learn albumentations opencv-python matplotlib seaborn pyarrow
```

### Data layout
- Place images in `images/` as `.jpg`.
- Place annotations in `annotations/` with the same basename as images and suffixes:
  - `_exp.npy` (int class 0..7, required)
  - `_val.npy` (float, optional)
  - `_aro.npy` (float, optional)
  - `_lnd.npy` (optional facial landmarks)

### 1) Build metadata (required)
Creates `metadata_all.parquet`, `metadata_train.parquet`, `metadata_val.parquet`.
```bash
python metadata_parallel.py
```

### 2) Prepare dataset/transforms (no action required)
`FER_transform.py` defines the dataset and augmentations used by training/eval. You can optionally run it to sanity‑check a sample batch:
```bash
python FER_transform.py
```

### 3) Optional checks
- Check GPU availability:
```bash
python test_gpu.py
```
- Inspect label distribution / sentinel values in train split:
```bash
python test_labels.py
```

### 4) Train (ResNet‑50, VGGFace2‑initialized)
By default, the script looks for `weight/resnet50_vggface2.pth`. If missing, it falls back to ImageNet weights.
```bash
python train_resnet50.py \
  --train_meta metadata_train.parquet \
  --val_meta metadata_val.parquet \
  --epochs 100 \
  --batch_size 8 \
  --img_size 224 \
  --save_path FER_resnet50.pth
```
Useful flags (see script for all): `--lr`, `--weight_decay`, `--freeze_epochs`, `--unfreeze_all_at`, `--vggface2_ckpt`.

Outputs: best checkpoint saved to `FER_resnet50.pth` (by validation accuracy).

### 5) Evaluate
Runs on the validation split and saves a confusion matrix.
```bash
python eval_resnet50.py
```
Outputs:
- Classification: Accuracy, Macro F1, Cohen's Kappa, ROC-AUC (macro OvR), PR-AUC (macro), Krippendorff's Alpha
- Regression: RMSE/MAE (valence, arousal), Pearson CORR, SAGR, CCC
- Saves `ConfusionMatrix_resnet50.png`

### Tips
- If you hit CUDA OOM, reduce `--batch_size` or `--img_size`.
- On Windows, using `num_workers=0` is already set in training/eval to avoid DataLoader hangs.

### Further reading
- See [REPORT.md](REPORT.md) for a concise description of the pipeline:
  - What `metadata_parallel.py` builds, how `FER_transform.py` prepares data
  - Model architecture, initialization, training policy and settings in `train_resnet50.py`
- See [RESULTS.md](RESULTS.md) for full validation metrics and confusion matrices across models.