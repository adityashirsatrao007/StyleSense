import tensorflow as tf
import numpy as np
from pathlib import Path
import config

AUTOTUNE = tf.data.AUTOTUNE


def create_datasets(
    data_dir=None,
    img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size=config.BATCH_SIZE,
    validation_split=config.VAL_SPLIT,
    augment=True,
):
    if data_dir is None:
        data_dir = str(config.RAW_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=config.RANDOM_SEED,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=config.RANDOM_SEED,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    class_indices = {name: i for i, name in enumerate(class_names)}

    raw_path = Path(data_dir)
    class_counts = {}
    for cls in class_names:
        cls_dir = raw_path / cls
        class_counts[cls] = len(list(cls_dir.glob("*")))

    total = sum(class_counts.values())
    train_samples = int(total * (1 - validation_split))
    val_samples = total - train_samples

    class_info = {
        "class_names": class_names,
        "class_indices": class_indices,
        "class_counts": class_counts,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }

    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)

    to_onehot = lambda x, y: (x, tf.one_hot(y, config.NUM_CLASSES))
    train_ds = train_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(to_onehot, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    if augment:
        aug_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.35),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ])
        train_ds = train_ds.map(
            lambda x, y: (aug_layers(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    train_ds = train_ds.shuffle(500, reshuffle_each_iteration=True)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_info


def compute_class_weight_from_dir(data_dir=None, class_info=None):
    if class_info:
        class_counts_dict = class_info["class_counts"]
        class_names = class_info["class_names"]
    else:
        if data_dir is None:
            data_dir = config.RAW_DIR
        raw_path = Path(data_dir)
        class_names = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
        class_counts_dict = {}
        for cls in class_names:
            cls_dir = raw_path / cls
            class_counts_dict[cls] = len(list(cls_dir.glob("*")))

    total = sum(class_counts_dict.values())
    n_classes = len(class_names)
    weights = {}
    for i, cls in enumerate(class_names):
        count = class_counts_dict.get(cls, 1)
        weights[i] = total / (n_classes * count)
    return weights


def prepare_single_image(image_path, img_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def prepare_numpy_array(img_array, img_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    import cv2
    import numpy as np
    img = cv2.resize(img_array, img_size)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img
