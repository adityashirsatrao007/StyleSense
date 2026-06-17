import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import config
from src.predict import StyleSensePredictor


def main():
    parser = argparse.ArgumentParser(description="Run StyleSense Inference")
    parser.add_argument("image_path", type=str, nargs="?", help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.SAVED_MODELS_DIR / "stylesense_best.keras"),
        help="Path to trained model",
    )
    parser.add_argument("--tflite", action="store_true", help="Use TFLite model")
    args = parser.parse_args()

    if args.tflite:
        model_path = config.TFLITE_DIR / "stylesense_model.tflite"
        predictor = StyleSensePredictor(tflite_path=str(model_path))
    else:
        if not Path(args.model_path).exists():
            print(f"ERROR: Model not found at {args.model_path}")
            print("Train a model first:  python train.py --data_dir data/raw")
            print("Or generate demo data: python data/create_demo_data.py && python train.py")
            sys.exit(1)
        predictor = StyleSensePredictor(model_path=args.model_path)

    if args.image_path:
        if not Path(args.image_path).exists():
            print(f"ERROR: Image not found: {args.image_path}")
            print("Provide a valid image path.")
            sys.exit(1)
        result = predictor.predict(args.image_path)
        print(f"\n{'='*50}")
        print(f"  Predicted Style: {result['predicted_class']}")
        print(f"  Confidence:      {result['confidence']*100:.2f}%")
        print(f"{'='*50}")
        print(f"  Class Probabilities:")
        for cls, prob in sorted(result["class_probabilities"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"    {cls:15s}: {prob*100:5.2f}% {bar}")
        print(f"{'='*50}")
    else:
        print("StyleSense Inference Engine")
        print(f"  Model: {args.model_path}")
        print(f"\nUsage: python inference.py path/to/image.jpg")
        print(f"       python inference.py path/to/image.jpg --tflite")


if __name__ == "__main__":
    main()
