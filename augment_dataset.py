# python augment_dataset.py --balance-to 1200 --size 224 --workers 8
# or
# python augment_dataset.py --multiplier 2.0 --size 224 --workers 8
import os, argparse, json, uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, cv2
from tqdm import tqdm
import albumentations as A

IMG_DIR_OUT, ANNOT_DIR_OUT = "images_aug", "annotations_aug"
META_TRAIN_IN, META_TRAIN_OUT = "metadata_train.parquet", "metadata_train_aug.parquet"
CLASS_NAMES = ["Neutral","Happy","Sad","Surprise","Fear","Disgust","Anger","Contempt"]

def _build_aug_local(size):
    # Milder, face-preserving offline aug
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={"x":(-0.04,0.04),"y":(-0.04,0.04)}, scale=(0.92,1.08), rotate=(-8,8), shear={"x":0,"y":0}, p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.7),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.6),
        A.Resize(height=size, width=size, p=1.0),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def _worker_init():
    try: cv2.setNumThreads(0)
    except: pass

def _to_keypoints_safe(lnd):
    if lnd is None or isinstance(lnd, float):  # NaN/float => missing
        return None
    try: arr = np.asarray(lnd, dtype=np.float32)
    except: return None
    arr = arr.reshape(-1)
    if arr.size == 0 or arr.size % 2 != 0: return None
    arr = arr.reshape(-1, 2)
    if not np.isfinite(arr).all(): return None
    return [tuple(map(float, pt)) for pt in arr]

def _norm_landmarks_for_meta(lnd):
    kps = _to_keypoints_safe(lnd)
    return None if not kps else [[float(x), float(y)] for (x,y) in kps]

def _augment_once(task):
    size = task["size"]; aug = _build_aug_local(size)
    img = cv2.imread(task["filename"])
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kps = _to_keypoints_safe(task.get("landmarks", None))
    if kps is None:
        t = aug(image=img); img_aug, lnd_aug = t["image"], None
    else:
        t = aug(image=img, keypoints=kps); img_aug, lnd_aug = t["image"], t["keypoints"]

    out_base = uuid.uuid4().hex
    out_img_path = os.path.join(IMG_DIR_OUT, f"{out_base}.jpg")
    cv2.imwrite(out_img_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY),95])

    np.save(os.path.join(ANNOT_DIR_OUT, f"{out_base}_exp.npy"), np.int64(task["expression"]))
    np.save(os.path.join(ANNOT_DIR_OUT, f"{out_base}_val.npy"), np.float32(task["valence"]))
    np.save(os.path.join(ANNOT_DIR_OUT, f"{out_base}_aro.npy"), np.float32(task["arousal"]))
    if lnd_aug is not None:
        np.save(os.path.join(ANNOT_DIR_OUT, f"{out_base}_lnd.npy"), np.asarray(lnd_aug, dtype=np.float32))

    return {
        "filename": out_img_path.replace("\\","/"),
        "expression": int(task["expression"]),
        "valence": float(task["valence"]),
        "arousal": float(task["arousal"]),
        "landmarks": None if lnd_aug is None else [[float(x),float(y)] for (x,y) in lnd_aug],
    }

def ensure_dirs():
    os.makedirs(IMG_DIR_OUT, exist_ok=True); os.makedirs(ANNOT_DIR_OUT, exist_ok=True)

def compute_balance_plan(df, balance_to):
    counts = df["expression"].value_counts().to_dict()
    plan = defaultdict(list)
    for cls, cnt in sorted(counts.items()):
        need = max(0, balance_to - cnt)
        if need == 0: continue
        idxs = df.index[df["expression"]==cls].tolist()
        q, r = divmod(need, len(idxs))
        for i, idx in enumerate(idxs):
            rep = q + (1 if i < r else 0)
            if rep>0: plan[cls].append((idx, rep))
    return plan, counts

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--multiplier", type=float)
    g.add_argument("--balance-to", type=int)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(META_TRAIN_IN):
        raise FileNotFoundError("Missing metadata_train.parquet")

    df = pd.read_parquet(META_TRAIN_IN)
    df = df[(df["valence"].notna()) & (df["arousal"].notna()) &
            (df["valence"]>-2) & (df["arousal"]>-2)].reset_index(drop=True)
    df["landmarks"] = df["landmarks"].apply(_norm_landmarks_for_meta) if "landmarks" in df.columns else None
    print(f"Train rows (filtered): {len(df)}"); ensure_dirs()

    tasks = []
    if args.multiplier is not None:
        tgt = int(round((args.multiplier-1.0)*len(df)))
        if tgt>0:
            idxs = np.random.randint(0, len(df), size=tgt)
            for i in idxs:
                row = df.iloc[i]
                tasks.append({"filename":row["filename"], "expression":int(row["expression"]),
                              "valence":float(row["valence"]), "arousal":float(row["arousal"]),
                              "landmarks":row["landmarks"], "size":args.size})
    else:
        plan, counts = compute_balance_plan(df, args.balance_to)
        print("Current class counts:", json.dumps({int(k):int(v) for k,v in counts.items()}, indent=2))
        print("Balancing plan:"); 
        for cls in sorted(plan): print(f"  class {cls}: +{sum(r for _,r in plan[cls])}")
        for cls in sorted(plan):
            for idx, rep in plan[cls]:
                row = df.iloc[idx]
                for _ in range(rep):
                    tasks.append({"filename":row["filename"], "expression":int(row["expression"]),
                                  "valence":float(row["valence"]), "arousal":float(row["arousal"]),
                                  "landmarks":row["landmarks"], "size":args.size})

    print(f"Planning {len(tasks)} augmented samples...")
    out_rows=[]
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1,args.workers), initializer=_worker_init) as ex:
            for r in tqdm(ex.map(_augment_once, tasks, chunksize=16), total=len(tasks)):
                if r is not None: out_rows.append(r)

    print(f"Created {len(out_rows)} new augmented samples.")
    if out_rows:
        df_out = pd.concat([df, pd.DataFrame(out_rows)], axis=0, ignore_index=True)
        df_out = df_out.sample(frac=1.0, random_state=123).reset_index(drop=True)
        df_out.to_parquet(META_TRAIN_OUT, index=False)
        print(f"Wrote: {META_TRAIN_OUT}  (rows: {len(df_out)})")
        print(f"Augmented images into: {IMG_DIR_OUT} | annotations into: {ANNOT_DIR_OUT}")
    else:
        print("No new samples created; nothing written.")

if __name__ == "__main__":
    main()
