from FER_transform import FERDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_ds = FERDataset("metadata_train.parquet", train=True)
    val_ds = FERDataset("metadata_val.parquet", train=False)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    batch = next(iter(train_dl))
    print("Images:", batch["image"].shape)
    print("Class labels:", batch["y_cls"].shape)
    print("Valence/Arousal:", batch["y_reg"].shape)
