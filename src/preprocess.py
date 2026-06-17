import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config


def create_data_generators(
    data_dir=None,
    img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    batch_size=config.BATCH_SIZE,
    validation_split=config.VAL_SPLIT,
):
    if data_dir is None:
        data_dir = config.RAW_DIR

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        validation_split=validation_split,
    )

    val_datagen = ImageDataGenerator(
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=config.RANDOM_SEED,
    )

    return train_generator, validation_generator


def compute_class_weight(train_generator):
    total = train_generator.samples
    n_classes = len(train_generator.class_indices)
    class_counts = np.bincount(train_generator.classes)
    weights = {i: total / (n_classes * count) for i, count in enumerate(class_counts)}
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
