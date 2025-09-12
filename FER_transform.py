import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]

def build_transforms(train=True, size=224):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.7),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

class FERDataset(Dataset):
    def __init__(self, metadata_path, train=True):
        self.df = pd.read_parquet(metadata_path)
        print(f"Total samples in {metadata_path}: {len(self.df)}")
        
        # handle NaN values and remove uncertain/no-face (-2)
        self.df = self.df[(self.df["valence"].notna()) & (self.df["arousal"].notna()) & 
                        (self.df["valence"] > -2) & (self.df["arousal"] > -2)].reset_index(drop=True)
        print(f"Samples after filtering: {len(self.df)}")
        self.transforms = build_transforms(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filename"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transforms(image=img)["image"]
        y_cls = torch.tensor(row["expression"], dtype=torch.long)
        y_reg = torch.tensor([row["valence"], row["arousal"]], dtype=torch.float32)

        return {"image": img, "y_cls": y_cls, "y_reg": y_reg}