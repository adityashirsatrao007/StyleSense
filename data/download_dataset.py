"""
Download and prepare the DeepFashion2 dataset for StyleSense training.

DeepFashion2 is a comprehensive fashion dataset with 491K images across
13 clothing categories. This script downloads the validation set (~30K images)
which can be filtered into the 5 StyleSense categories.

Alternatively, use Fashion-MNIST (simpler, grayscale) as a quick demo.

Usage:
    python data/download_dataset.py --source deepfashion2
    python data/download_dataset.py --source fashion_mnist
    python data/download_dataset.py --source kaggle  # requires kaggle API
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import config


def setup_fashion_mnist():
    """Download Fashion-MNIST via TensorFlow and save as PNGs."""
    print("[INFO] Setting up Fashion-MNIST demo dataset...")
    import tensorflow as tf
    import numpy as np
    from PIL import Image

    fashion_labels = {
        0: "Casual",    # T-shirt/top
        1: "Casual",    # Trouser
        2: "Night Party",  # Pullover
        3: "Night Party",  # Dress
        4: "Casual",    # Coat
        5: "Sports",    # Sandal
        6: "Sports",    # Shirt
        7: "Sports",    # Sneaker
        8: "Business",  # Bag
        9: "Business",  # Ankle boot
    }

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    mapped_count = {cls: 0 for cls in config.CLASS_NAMES}
    output_dir = config.RAW_DIR

    for i in range(len(x_all)):
        label = int(y_all[i])
        mapped = fashion_labels.get(label)
        if mapped is None:
            continue
        if mapped_count[mapped] >= 200:
            continue

        class_dir = output_dir / mapped
        class_dir.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(x_all[i]).convert("RGB").resize((224, 224))
        img.save(str(class_dir / f"fmnist_{mapped_count[mapped]:04d}.png"))
        mapped_count[mapped] += 1

    print(f"\n✅ Fashion-MNIST dataset saved to {output_dir}/")
    for cls, count in mapped_count.items():
        print(f"  {cls}: {count} images")
    print(f"\n  Total: {sum(mapped_count.values())} images")
    print(f"  Now run: python train.py --data_dir {output_dir}")


def setup_deepfashion2():
    """
    Instructions for DeepFashion2 setup.
    Download from: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html
    """
    print("=" * 60)
    print("  DeepFashion2 Dataset Setup Instructions")
    print("=" * 60)
    print("""
  DeepFashion2 is available for academic use at:
    http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html

  1. Register and download the dataset
  2. Extract to data/raw/ with the following structure:

     data/raw/
     ├── Business/     (formal suits, blazers, ties)
     ├── Casual/       (t-shirts, jeans, sweaters)
     ├── Night Party/  (dresses, party wear)
     ├── Sports/       (athletic wear, sneakers)
     └── Wedding/      (wedding gowns, sherwanis)

  3. For automatic downloading, use:
     python -c "
     import kagglehub
     path = kagglehub.dataset_download('paramaggarwal/fashion-product-images-small')
     print('Downloaded to:', path)
     "

  4. Preprocess and organize into class folders:
     python data/download_dataset.py --source kaggle
    """)


def setup_kaggle():
    """Download from Kaggle (requires kagglehub or kaggle API)."""
    try:
        import kagglehub
        print("[INFO] Downloading fashion-product-images-small from Kaggle...")
        path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
        print(f"  Downloaded to: {path}")
        print(f"  Now organize images into {config.RAW_DIR}/{{classes}}/ and run train.py")
    except ImportError:
        print("[INFO] Installing kagglehub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        setup_kaggle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download StyleSense dataset")
    parser.add_argument(
        "--source", type=str, default="fashion_mnist",
        choices=["fashion_mnist", "deepfashion2", "kaggle"],
        help="Dataset source",
    )
    args = parser.parse_args()

    if args.source == "fashion_mnist":
        setup_fashion_mnist()
    elif args.source == "deepfashion2":
        setup_deepfashion2()
    elif args.source == "kaggle":
        setup_kaggle()
