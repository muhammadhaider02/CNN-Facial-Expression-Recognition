import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

def build_transforms(train=True, size=224):
    if train:
        return A.Compose([
            # Cropping & flips
            A.RandomResizedCrop(height=size, width=size, scale=(0.9, 1.0), ratio=(0.97, 1.03)),
            A.HorizontalFlip(p=0.5),

            # Color & lighting variation (mild)
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.5),

            # Geometric transform (with expanded Affine args)
            A.Affine(
                translate_percent=(0.0, 0.05),
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                cval=0,
                mode=cv2.BORDER_REFLECT_101,
                p=0.4
            ),

            A.CoarseDropout(
                max_holes=2,
                max_height=int(0.2 * size),
                max_width=int(0.2 * size),
                fill_value=0,
                p=0.4
            ),

            # Noise & denoising / lighting normalization
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.4),
            A.MedianBlur(blur_limit=3, p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # Normalize and to tensor (ImageNet stats)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

class FERDataset(Dataset):
    def __init__(self, metadata_path, train=True):
        self.df = pd.read_parquet(metadata_path)
        print(f"Total samples in {metadata_path}: {len(self.df)}")
        # match filtering used elsewhere
        self.df = self.df[
            (self.df["valence"].notna()) &
            (self.df["arousal"].notna()) &
            (self.df["valence"] > -2) &
            (self.df["arousal"] > -2)
        ].reset_index(drop=True)
        print(f"Samples after filtering: {len(self.df)}")
        self.transforms = build_transforms(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filename"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        y_cls = torch.tensor(int(row["expression"]), dtype=torch.long)
        y_reg = torch.tensor([float(row["valence"]), float(row["arousal"])], dtype=torch.float32)

        return {"image": img, "y_cls": y_cls, "y_reg": y_reg}

if __name__ == "__main__":
    BATCH_SIZE = 32
    train_ds = FERDataset("metadata_train.parquet", train=True)
    val_ds   = FERDataset("metadata_val.parquet", train=False)

    # On Windows, bumping workers can hang; if so, set to 0
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    batch = next(iter(train_dl))
    print("Batch images shape:", batch["image"].shape)
    print("Batch y_cls shape:", batch["y_cls"].shape)
    print("Batch y_reg shape:", batch["y_reg"].shape)