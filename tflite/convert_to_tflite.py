import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
import config


def convert_to_tflite(
    keras_model_path,
    output_path=None,
    quantize=False,
    representative_dataset=None,
):
    if output_path is None:
        output_path = config.TFLITE_DIR / "stylesense_model.tflite"

    print(f"[INFO] Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path)

    print("[INFO] Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset:
            converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.float16]
        print("[INFO] Applying FP16 quantization for mobile optimization")
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"[INFO] TFLite model saved to {output_path}")
    print(f"[INFO] Model size: {file_size_mb:.2f} MB")

    return output_path


def convert_quantized(keras_model_path):
    output_path = config.TFLITE_DIR / "stylesense_model_quantized.tflite"
    return convert_to_tflite(keras_model_path, output_path, quantize=True)


def convert_edge_optimized(keras_model_path):
    output_path = config.TFLITE_DIR / "stylesense_model_edge.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(keras_model_path)
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Edge-optimized TFLite model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert StyleSense to TFLite")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.SAVED_MODELS_DIR / "stylesense_final.keras"),
        help="Path to trained Keras model",
    )
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument(
        "--edge", action="store_true", help="Edge-optimized conversion"
    )
    args = parser.parse_args()

    if args.edge:
        convert_edge_optimized(args.model_path)
    elif args.quantize:
        convert_quantized(args.model_path)
    else:
        convert_to_tflite(args.model_path)
