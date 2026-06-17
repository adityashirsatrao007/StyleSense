import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import numpy as np
import argparse
from src.model import build_stylesense_model, compile_model, get_callbacks
from src.preprocess import create_data_generators
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_auc,
    plot_training_history,
    plot_prediction_confidence,
)
import config


def main():
    parser = argparse.ArgumentParser(description="Train StyleSense Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(config.RAW_DIR),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Unlock base model layers for fine-tuning",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("StyleSense: Real-Time AI Fashion Recommendations using MobileNet")
    print("=" * 60)

    print("\n[INFO] Loading data generators...")
    train_gen, val_gen = create_data_generators(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes: {train_gen.class_indices}")

    print("\n[INFO] Building StyleSense MobileNetV2 model...")
    model = build_stylesense_model(trainable_base=args.fine_tune)
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    print("\n[INFO] Starting training...")
    callbacks = get_callbacks()
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[INFO] Evaluating model on validation set...")
    results = evaluate_model(model, val_gen)

    print("\n[INFO] Generating plots...")
    plot_training_history(history, save_path_prefix="stylesense")
    plot_confusion_matrix(
        results["confusion_matrix"],
        results["class_labels"],
        save_path="stylesense_confusion_matrix.png",
    )
    plot_roc_auc(
        results["y_true"],
        results["predictions"],
        results["class_labels"],
        save_path="stylesense_roc_auc.png",
    )
    plot_prediction_confidence(
        results["predictions"],
        results["y_true"],
        results["class_labels"],
        save_path="stylesense_confidence_dist.png",
    )

    print("\n[INFO] Saving final model...")
    final_path = config.SAVED_MODELS_DIR / "stylesense_final.keras"
    model.save(final_path)
    print(f"  Model saved to {final_path}")

    print("\n[INFO] Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = config.TFLITE_DIR / "stylesense_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite model saved to {tflite_path}")

    print("\n" + "=" * 60)
    print("StyleSense training complete!")
    print(f"  Validation Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Weighted F1-Score:   {results['f1_score']:.4f}")
    print(f"  Test Loss:           {history.history['val_loss'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
