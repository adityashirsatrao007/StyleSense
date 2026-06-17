"""
Create synthetic demo dataset for quick testing of StyleSense.
Generates simulated fashion images (colored rectangles + noise per class)
so the pipeline can be verified end-to-end without real data.

Usage:
    python data/create_demo_data.py          # 50 images per class
    python data/create_demo_data.py --n 100  # 100 images per class
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
import argparse
import config

# Class-specific color signatures (RGB)
CLASS_SIGNATURES = {
    "Business": (40, 60, 120),
    "Casual": (80, 160, 200),
    "Night Party": (180, 40, 120),
    "Sports": (60, 180, 80),
    "Wedding": (220, 180, 200),
}

SEED = config.RANDOM_SEED
rng = np.random.RandomState(SEED)


def generate_class_image(class_name, img_size=(224, 224)):
    base_color = CLASS_SIGNATURES[class_name]
    img = np.zeros((*img_size, 3), dtype=np.uint8)

    base_variation = rng.randint(-20, 21, 3)
    color = np.clip(np.array(base_color) + base_variation, 0, 255)

    if class_name == "Business":
        img[:, :, :] = color
        stripe = rng.randint(0, 3)
        img[:, stripe::4, :] = np.clip(color + 30, 0, 255)
    elif class_name == "Casual":
        img[:, :, :] = color
        for _ in range(rng.randint(3, 8)):
            x, y = rng.randint(0, img_size[0]), rng.randint(0, img_size[1])
            s = rng.randint(20, 60)
            c = rng.randint(0, 255, 3).tolist()
            img[y:min(y+s, img_size[0]), x:min(x+s, img_size[1])] = c
    elif class_name == "Night Party":
        img[:, :, :] = color
        for _ in range(rng.randint(10, 25)):
            x, y = rng.randint(0, img_size[0]), rng.randint(0, img_size[1])
            r = rng.randint(5, 15)
            yy, xx = np.ogrid[:img_size[0], :img_size[1]]
            mask = (xx - x)**2 + (yy - y)**2 <= r**2
            sparkle = rng.randint(200, 256, 3).tolist()
            img[mask] = sparkle
    elif class_name == "Sports":
        img[:, :, :] = color
        for i in range(0, img_size[0], 20):
            img[i:i+4, :, :] = np.clip(color + 40, 0, 255)
    elif class_name == "Wedding":
        img[:, :, :] = color
        pattern = np.zeros((8, 8, 3), dtype=np.uint8)
        pattern[::2, ::2] = np.clip(color + 50, 0, 255)
        pattern[1::2, 1::2] = np.clip(color + 50, 0, 255)
        for i in range(0, img_size[0], 8):
            for j in range(0, img_size[1], 8):
                img[i:i+8, j:j+8] = pattern

    noise = rng.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser(description="Create demo dataset")
    parser.add_argument("--n", type=int, default=50, help="Images per class")
    parser.add_argument("--output_dir", type=str, default=str(config.RAW_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng.seed(SEED)

    for class_name in config.CLASS_NAMES:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        for i in range(args.n):
            img = generate_class_image(class_name)
            img.save(str(class_dir / f"{class_name.lower()}_{i:04d}.png"))
        print(f"  {class_name}: {args.n} images")

    total = args.n * len(config.CLASS_NAMES)
    print(f"\n✅ Demo dataset created: {total} images in {output_dir}/")
    print(f"   ~{args.n} per class across {len(config.CLASS_NAMES)} categories")
    print(f"\n   Train now: python train.py --data_dir {output_dir}")
    print(f"   Generate all paper figures (synthetic): python src/evaluate.py")


if __name__ == "__main__":
    main()
