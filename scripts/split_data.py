"""Split HandPD into train/test BY PERSON so nobody appears in both sets."""
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW = Path("data/raw/handpd/images")
OUT = Path("data")

# 1. Build a table: one row per image, with its label, drawing type and owner
rows = []
for drawing in ["Spiral", "Meander"]:
    for folder, label in [("Control", "healthy"), ("Patients", "parkinson")]:
        for img in (RAW / f"{drawing}_HandPD" / f"{drawing}{folder}").glob("*.jpg"):
            person_id = img.stem.split("-")[0]
            rows.append({
                "path": img,
                "label": label,
                "drawing": drawing.lower(),
                "person": f"{label}_{person_id}",
            })
df = pd.DataFrame(rows)
print("People per class:\n", df.groupby("label")["person"].nunique())

# 2. Split: ~20% of PEOPLE go to test, keeping the healthy/parkinson ratio
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["person"]))
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

# Check that the derived participant groups are disjoint.
overlap = set(df[df.split == "train"].person) & set(df[df.split == "test"].person)
assert not overlap, f"Leakage! {overlap}"

# 4. Copy into data/<split>/<label>/<drawing>_<file>.jpg
for r in df.itertuples():
    dest = OUT / r.split / r.label / f"{r.drawing}_{r.path.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(r.path, dest)

# 5. Save a record of the split so it can be reproduced
df.assign(path=df.path.astype(str)).to_csv(OUT / "split_manifest.csv", index=False)
print(df.groupby(["split", "label"]).size())