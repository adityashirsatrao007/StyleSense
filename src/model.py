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
):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(fc_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="StyleSense")
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks():
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
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(config.SAVED_MODELS_DIR / "stylesense_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, reduce_lr, model_checkpoint]
