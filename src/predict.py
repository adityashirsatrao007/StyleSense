import tensorflow as tf
import numpy as np
from pathlib import Path
import config
from src.preprocess import prepare_single_image, prepare_numpy_array


class StyleSensePredictor:
    def __init__(self, model_path=None, tflite_path=None):
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = config.CLASS_NAMES

        if model_path and Path(model_path).exists():
            self.load_keras_model(model_path)
        elif tflite_path and Path(tflite_path).exists():
            self.load_tflite_model(tflite_path)

    def load_keras_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Keras model loaded from {model_path}")

    def load_tflite_model(self, tflite_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"TFLite model loaded from {tflite_path}")

    def predict_keras(self, image_input):
        if isinstance(image_input, (str, Path)):
            img_array = prepare_single_image(str(image_input))
        else:
            img_array = prepare_numpy_array(image_input)

        predictions = self.model.predict(img_array, verbose=0)
        return self._process_predictions(predictions[0])

    def predict_tflite(self, image_input):
        if isinstance(image_input, (str, Path)):
            img_array = prepare_single_image(str(image_input))
        else:
            img_array = prepare_numpy_array(image_input)

        img_array = img_array.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]["index"])
        return self._process_predictions(predictions[0])

    def _process_predictions(self, preds):
        pred_class_idx = int(np.argmax(preds))
        confidence = float(preds[pred_class_idx])
        pred_class = self.class_names[pred_class_idx]

        all_probs = {
            self.class_names[i]: float(preds[i]) for i in range(len(self.class_names))
        }

        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "class_probabilities": all_probs,
        }

    def predict(self, image_input):
        if self.model:
            return self.predict_keras(image_input)
        elif self.interpreter:
            return self.predict_tflite(image_input)
        else:
            raise ValueError("No model loaded. Provide model_path or tflite_path.")
