"""Map Kaggle Fashion Product Images dataset to StyleSense 5 classes.

Usage:
    python data/prepare_dataset.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import pandas as pd
from config import RAW_DIR

KAGGLE_PATH = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "paramaggarwal"
    / "fashion-product-images-small"
    / "versions"
    / "1"
)

MAX_PER_CLASS = 800
RANDOM_SEED = 42


def prepare():
    df = pd.read_csv(KAGGLE_PATH / "styles.csv", on_bad_lines="skip")
    img_dir = KAGGLE_PATH / "images"

    mapping = {
        "Business": df[df["usage"] == "Formal"],
        "Casual": df[df["usage"] == "Casual"].sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED),
        "Night Party": df[df["articleType"].isin(["Dresses", "Night suits", "Skirts"])],
        "Sports": df[df["usage"] == "Sports"].sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED),
        "Wedding": df[df["usage"] == "Ethnic"].sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED),
    }

    total = 0
    for cls_name, cls_df in mapping.items():
        dst_dir = RAW_DIR / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for _, row in cls_df.iterrows():
            img_id = row["id"]
            src = img_dir / f"{img_id}.jpg"
            dst = dst_dir / f"{cls_name.lower().replace(' ', '_')}_{img_id}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1

        total += copied
        print(f"  {cls_name}: {copied} images")

    print(f"\nTotal: {total} images in {RAW_DIR}/")
    print("Ready to train!")


if __name__ == "__main__":
    prepare()
