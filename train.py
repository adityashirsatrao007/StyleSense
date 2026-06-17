import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
import argparse
import numpy as np
from src.model import build_stylesense_model, compile_model, get_callbacks, unfreeze_model
from src.preprocess import create_data_generators, compute_class_weight
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_auc,
    plot_training_history,
    plot_prediction_confidence,
    plot_performance_metrics,
)
import config


def find_latest_checkpoint(prefix="pt"):
    pattern = f"stylesense_{prefix}_epoch_*.keras"
    checkpoints = list(config.SAVED_MODELS_DIR.glob(pattern))
    if not checkpoints:
        return None, 0
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_epoch_")[1]))
    epoch = int(latest.stem.split("_epoch_")[1])
    return latest, epoch


def main():
    parser = argparse.ArgumentParser(description="Train StyleSense Model")
    parser.add_argument(
        "--data_dir", type=str, default=str(config.RAW_DIR),
        help="Path to dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Unlock base model layers")
    parser.add_argument("--output_dir", type=str, default="paper_figures", help="Figure output directory")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  StyleSense: Real-Time AI Fashion Recommendations using MobileNet")
    print("=" * 60)

    print("\n[Step 1/6] Loading data generators...")
    train_gen, val_gen = create_data_generators(
        data_dir=args.data_dir, batch_size=args.batch_size,
    )

    if args.no_augment:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        no_aug = ImageDataGenerator(validation_split=config.VAL_SPLIT)
        train_gen = no_aug.flow_from_directory(
            args.data_dir, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            batch_size=args.batch_size, class_mode="categorical",
            subset="training", shuffle=True, seed=config.RANDOM_SEED,
        )
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes: {train_gen.class_indices}")

    class_weight = compute_class_weight(train_gen)
    print(f"  Class weights: {class_weight}")

    phase_id = "pt"
    phase_name = "Phase 1 — Training classification head (base frozen)"
    total_epochs = max(15, args.epochs // 3)
    initial_epoch = 0

    if args.fine_tune:
        phase_id = "ft"
        phase_name = "Phase 2 — Fine-tuning MobileNetV2"
        total_epochs = args.epochs

    if args.resume:
        ckpt_path, initial_epoch = find_latest_checkpoint(phase_id)
        if ckpt_path:
            print(f"\n[Resume] Found checkpoint at epoch {initial_epoch}: {ckpt_path}")
            model = tf.keras.models.load_model(ckpt_path)
            if args.fine_tune:
                model = unfreeze_model(model)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
                    metrics=["accuracy"],
                )
            print(f"  Resuming from epoch {initial_epoch + 1}/{total_epochs}")
        else:
            print("[Resume] No checkpoint found, starting from scratch")
            initial_epoch = 0

    if initial_epoch == 0:
        if args.fine_tune:
            print(f"\n[Step 2/6] Building & fine-tuning model...")
            model = build_stylesense_model(trainable_base=False)
            model = compile_model(model, learning_rate=args.lr, label_smoothing=config.LABEL_SMOOTHING)
            model.summary()
            model = unfreeze_model(model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
                metrics=["accuracy"],
            )
        else:
            print("\n[Step 2/6] Building StyleSense MobileNetV2 model...")
            model = build_stylesense_model(trainable_base=False)
            model = compile_model(model, learning_rate=args.lr, label_smoothing=config.LABEL_SMOOTHING)
            model.summary()

    print(f"\n[Step 3/6] {phase_name} for {total_epochs} epochs...")
    callbacks = get_callbacks(phase=phase_id)
    try:
        history = model.fit(
            train_gen,
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
    except Exception as e:
        print(f"\n[ERROR] Training interrupted: {e}")
        emergency_path = config.SAVED_MODELS_DIR / f"stylesense_{phase_id}_emergency.keras"
        model.save(emergency_path)
        print(f"  Emergency checkpoint saved to {emergency_path}")
        sys.exit(1)

    print("\n[Step 4/6] Evaluating model on validation set...")
    results = evaluate_model(model, val_gen)

    print("\n[Step 5/6] Generating paper-quality figures...")

    # Fig 5 — Performance Metrics
    plot_performance_metrics(
        results["accuracy"], results["precision"],
        results["recall"], results["f1_score"],
        save_path=str(output_dir / "Fig5_performance_metrics.png"),
    )

    # Fig 6 — Confusion Matrix
    plot_confusion_matrix(
        results["confusion_matrix"], results["class_labels"],
        save_path=str(output_dir / "Fig6_confusion_matrix.png"),
    )

    # Fig 3 — ROC & AUC
    plot_roc_auc(
        results["y_true"], results["predictions"], results["class_labels"],
        save_path=str(output_dir / "Fig3_roc_auc.png"),
    )

    # Fig 7 — Confidence Distribution
    plot_prediction_confidence(
        results["predictions"], results["y_true"], results["class_labels"],
        save_path=str(output_dir / "Fig7_confidence_dist.png"),
    )

    # Fig 8 — Training History
    plot_training_history(
        history,
        save_path=str(output_dir / "Fig8_training_history.png"),
    )

    # Fig 1, 2, 4 — Architecture diagrams (generated separately via paper_diagrams.py)
    print("  Note: Run `python src/paper_diagrams.py` for Fig 1, 2, 4")

    print("\n[Step 6/6] Saving model and converting to TFLite...")
    final_path = config.SAVED_MODELS_DIR / "stylesense_final.keras"
    model.save(final_path)
    print(f"  Model saved to {final_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = config.TFLITE_DIR / "stylesense_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite model saved to {tflite_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")

    print("\n" + "=" * 60)
    print(f"  Validation Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Weighted Precision:  {results['precision']:.4f}")
    print(f"  Weighted Recall:     {results['recall']:.4f}")
    print(f"  Weighted F1-Score:   {results['f1_score']:.4f}")
    print(f"  Test Loss:           {history.history['val_loss'][-1]:.4f}")
    print("=" * 60)
    print(f"\nAll paper figures saved to: {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
