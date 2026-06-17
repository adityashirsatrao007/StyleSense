import pandas as pd
import shutil
import random
from pathlib import Path

random.seed(42)
RAW_DIR = Path("/home/aditya/Desktop/Projects/StyleSense/data/raw")
EXT_DIR = Path("/home/aditya/Desktop/Projects/StyleSense/data/external")
CSV_PATH = EXT_DIR / "styles.csv"
IMG_DIR = EXT_DIR / "images"

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

def map_class(row):
    usage = str(row.get("usage", "")).strip()
    article = str(row.get("articleType", "")).strip()
    master = str(row.get("masterCategory", "")).strip()
    if usage == "Sports":
        return "Sports"
    if usage == "Ethnic":
        return "Wedding"
    if usage == "Party":
        return "Night Party"
    if article in ("Dresses", "Nightdress", "Bodycon", "Jumpsuit"):
        return "Night Party"
    if article == "Heels":
        return "Night Party"
    if usage in ("Formal", "Smart Casual"):
        return "Business"
    if usage == "Casual" and master == "Apparel":
        return "Casual"
    return None

df["target_class"] = df.apply(map_class, axis=1)
mapped = df[df["target_class"].notna()].copy()

# Cap heels to 500 (avoid overloading Night Party with shoe-only images)
heels_mask = mapped["articleType"] == "Heels"
heels_ids = mapped[heels_mask].index
if len(heels_ids) > 500:
    drop_ids = set(random.sample(list(heels_ids), len(heels_ids) - 500))
    mapped = mapped[~mapped.index.isin(drop_ids)]

# Cap Casual to 1500 (sample from ~15K available)
casual_mask = mapped["target_class"] == "Casual"
casual_df = mapped[casual_mask]
non_casual = mapped[~casual_mask]
if len(casual_df) > 1500:
    casual_df = casual_df.sample(n=1500, random_state=42)
mapped = pd.concat([non_casual, casual_df])

print(f"Mapped {len(mapped)} images")
print(mapped["target_class"].value_counts())
print()

for target_class in ["Casual", "Sports", "Night Party", "Wedding", "Business"]:
    existing = len(list((RAW_DIR / target_class).iterdir()))
    # Skip Business (already largest)
    if target_class == "Business":
        print(f"{target_class}: SKIP (already {existing})")
        continue
    target = max(2000, existing)
    need = target - existing
    if need <= 0:
        print(f"{target_class}: already {existing}, skip")
        continue

    subset = mapped[mapped["target_class"] == target_class]
    out_dir = RAW_DIR / target_class
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in subset.iterrows():
        if copied >= need:
            break
        src = IMG_DIR / f"{row['id']}.jpg"
        dst = out_dir / f"ext_{row['id']}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    new_total = len(list(out_dir.iterdir()))
    print(f"{target_class}: added {copied}/{need} needed (total: {new_total})")

print("\n=== FINAL COUNTS ===")
for d in sorted(RAW_DIR.iterdir()):
    if d.is_dir():
        print(f"  {d.name}: {len(list(d.iterdir()))}")
