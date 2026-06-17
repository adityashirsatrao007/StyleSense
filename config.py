import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
TFLITE_DIR = BASE_DIR / "tflite"

for d in [RAW_DIR, PROCESSED_DIR, SAVED_MODELS_DIR, TFLITE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Business", "Casual", "Night Party", "Sports", "Wedding"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
FC_LAYER_UNITS = 256

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

LABEL_SMOOTHING = 0.1
L2_REG = 1e-4
FINE_TUNE_AT_LAYER = 100
FREEZE_BN = True
