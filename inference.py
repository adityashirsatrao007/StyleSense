import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import argparse
import config
from src.predict import StyleSensePredictor


def main():
    parser = argparse.ArgumentParser(description="Run StyleSense Inference")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.SAVED_MODELS_DIR / "stylesense_best.keras"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--tflite",
        action="store_true",
        help="Use TFLite model for inference",
    )
    args = parser.parse_args()

    if args.tflite:
        model_path = config.TFLITE_DIR / "stylesense_model.tflite"
        predictor = StyleSensePredictor(tflite_path=str(model_path))
    else:
        predictor = StyleSensePredictor(model_path=args.model_path)

    result = predictor.predict(args.image_path)

    print("\n" + "=" * 50)
    print("StyleSense Fashion Recommendation")
    print("=" * 50)
    print(f"  Image:           {args.image_path}")
    print(f"  Predicted Style: {result['predicted_class']}")
    print(f"  Confidence:      {result['confidence']*100:.2f}%")
    print("-" * 50)
    print("  Class Probabilities:")
    for cls, prob in sorted(
        result["class_probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(prob * 30)
        print(f"    {cls:15s}: {prob*100:5.2f}% {bar}")
    print("=" * 50)


if __name__ == "__main__":
    main()
