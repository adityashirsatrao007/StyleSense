"""
StyleSense - Fashion Dataset Setup Script

Downloads a sample fashion dataset or prepares your own.
For demonstration purposes, this creates a small mock dataset
directory structure. Replace with your actual dataset.

Expected structure:
    data/raw/
        Business/
        Casual/
        Night Party/
        Sports/
        Wedding/

For real training, use the DeepFashion2 dataset or your curated collection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import shutil
import requests
import random
import config


def create_directory_structure():
    classes = config.CLASS_NAMES
    for cls in classes:
        cls_dir = config.RAW_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {cls_dir}/")
    print(f"\nDataset structure ready at: {config.RAW_DIR}")
    print(f"Place your images in the corresponding class folders.")
    print(f"\nMinimum recommended: ~100-150 images per class (500-750 total)")
    print(f"For paper-level accuracy (96%): ~600 images per class (3000+ total)")


def download_sample_dataset(url=None, extract=True):
    """
    Download a sample fashion dataset.
    By default provides instructions; uncomment and configure for your source.

    Options:
    - DeepFashion2: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html
    - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
    - Kaggle Fashion Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
    """
    print("Automatic download not configured.")
    print("\nRecommended datasets:")
    print("  1. DeepFashion2 - http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html")
    print(
        "  2. Fashion Product Images (Kaggle) - https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small"
    )
    print("  3. Use your own curated dataset of 3000+ images")
    print("\nPlace downloaded images into:")
    for cls in config.CLASS_NAMES:
        print(f"    {config.RAW_DIR / cls}/")
    create_directory_structure()

    print(f"\nThen run: python train.py --data_dir {config.RAW_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup StyleSense dataset")
    parser.add_argument(
        "--action",
        type=str,
        default="structure",
        choices=["structure", "download"],
        help="Create structure or download dataset",
    )
    args = parser.parse_args()

    if args.action == "structure":
        create_directory_structure()
    elif args.action == "download":
        download_sample_dataset()
