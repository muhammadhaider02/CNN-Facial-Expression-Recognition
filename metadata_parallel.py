import os, numpy as np, pandas as pd
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedShuffleSplit

IMG_DIR, ANNOT_DIR = "images", "annotations"
OUT_ALL, OUT_TRAIN, OUT_VAL = "metadata_all.parquet","metadata_train.parquet","metadata_val.parquet"
VAL_FRACTION, RANDOM_SEED = 0.2, 42
WORKERS = min(32, (os.cpu_count() or 4) * 2)

def _row_for_image(img_path: str):
    base = os.path.splitext(os.path.basename(img_path))[0]
    exp_p = os.path.join(ANNOT_DIR, f"{base}_exp.npy")
    val_p = os.path.join(ANNOT_DIR, f"{base}_val.npy")
    aro_p = os.path.join(ANNOT_DIR, f"{base}_aro.npy")
    lnd_p = os.path.join(ANNOT_DIR, f"{base}_lnd.npy")

    if not os.path.exists(exp_p):
        return None

    exp = int(np.load(exp_p, mmap_mode="r", allow_pickle=False))
    val = float(np.load(val_p, mmap_mode="r", allow_pickle=False)) if os.path.exists(val_p) else np.nan
    aro = float(np.load(aro_p, mmap_mode="r", allow_pickle=False)) if os.path.exists(aro_p) else np.nan

    lnd = np.load(lnd_p, allow_pickle=False) if os.path.exists(lnd_p) else None
    if lnd is not None:
        lnd = lnd.tolist()

    return {
        "filename": img_path.replace("\\", "/"),
        "expression": exp,
        "valence": val,
        "arousal": aro,
        "landmarks": lnd,
    }

def build():
    imgs = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
    rows = []

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_row_for_image, p) for p in imgs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if i % 1000 == 0 or i == len(imgs):
                print(f"[{i}/{len(imgs)}] images processed...")

    df = pd.DataFrame(rows)
    df = df.sort_values("filename", kind="stable").reset_index(drop=True)
    print(f"Total indexed: {len(df)}")
    df.to_parquet(OUT_ALL, index=False)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    idx_tr, idx_va = next(sss.split(df, df["expression"]))
    df.iloc[idx_tr].to_parquet(OUT_TRAIN, index=False)
    df.iloc[idx_va].to_parquet(OUT_VAL, index=False)
    print(f"Done. Train: {len(idx_tr)} | Val: {len(idx_va)}")

if __name__ == "__main__":
    build()