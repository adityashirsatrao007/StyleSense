import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2
import config


def build_stylesense_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes=config.NUM_CLASSES,
    dropout_rate=config.DROPOUT_RATE,
    fc_units=config.FC_LAYER_UNITS,
    trainable_base=False,
    l2_reg=config.L2_REG,
    fine_tune_at=None,
):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = trainable_base

    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    regularizer = regularizers.l2(l2_reg)

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=trainable_base or fine_tune_at is not None)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(fc_units, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizer)(x)

    model = Model(inputs=inputs, outputs=outputs, name="StyleSense")
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE, label_smoothing=0):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


def unfreeze_model(model, fine_tune_at=config.FINE_TUNE_AT_LAYER):
    base_model = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)),
        None,
    )
    if base_model is None:
        raise ValueError("No sub-model found to unfreeze")
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True
    # Keep BatchNorm frozen during fine-tune to avoid internal covariate shift on small data
    if config.FREEZE_BN:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model


def get_callbacks(phase="phase1"):
    is_phase2 = phase in ("phase2", "ft")
    prefix = "ft" if is_phase2 else "pt"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(config.SAVED_MODELS_DIR / f"stylesense_best_{prefix}.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    period_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(config.SAVED_MODELS_DIR / f"stylesense_{prefix}_epoch_{{epoch:02d}}.keras"),
        save_freq="epoch",
        save_best_only=False,
        verbose=0,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        str(config.SAVED_MODELS_DIR / f"training_log_{prefix}.csv"),
        append=True,
    )
    return [early_stopping, reduce_lr, best_checkpoint, period_checkpoint, csv_logger]
