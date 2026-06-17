# StyleSense — Real-Time AI Fashion Recommendations using MobileNet

**ISPCC 2025** — Prof. Ashlesha S. Adhatrao, Ms. Vaibhavi Zadbuke, Ms. Shruti G. Waghamare  
N K Orchid College of Engg & Tech, Solapur, MH, India

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityashirsatrao007/StyleSense/blob/main/notebooks/StyleSense_Colab_Training.ipynb)

---

## Overview

StyleSense is a MobileNetV2-based fashion classification system that analyzes clothing in real-time and categorizes styles into 5 social contexts:

| Category | Description |
|----------|-------------|
| **Business** | Formal corporate attire |
| **Casual** | Everyday relaxed wear |
| **Night Party** | Evening/party ensembles |
| **Sports** | Athletic and sportswear |
| **Wedding** | Ceremonial wedding attire |

### Paper Results
- **Validation Accuracy:** 96.00%
- **Weighted F1-Score:** 0.95
- **Precision/Recall:** 0.95 / 0.95
- **Test Loss:** 0.18

---

## Project Structure

```
StyleSense/
├── config.py                  # Configuration & hyperparameters
├── train.py                   # Training pipeline CLI
├── inference.py               # Single-image inference CLI
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── src/
│   ├── model.py               # MobileNetV2 architecture
│   ├── preprocess.py          # Data loading & augmentation
│   ├── evaluate.py            # Metrics, confusion matrix, ROC curves
│   └── predict.py             # Inference engine (Keras + TFLite)
├── app/
│   └── app.py                 # Flask web application
├── tflite/
│   └── convert_to_tflite.py   # TFLite conversion utilities
├── notebooks/
│   └── stylesense_training.ipynb  # Jupyter training notebook
├── data/
│   └── dataset_setup.py       # Dataset directory setup
└── saved_models/              # Trained model output
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Expects images organized by class in `data/raw/`:

```
data/raw/
├── Business/
├── Casual/
├── Night Party/
├── Sports/
└── Wedding/
```

Create the structure:
```bash
python data/dataset_setup.py --action structure
```

Place your images (~600 per class for 96% accuracy) into the respective folders.

**Recommended Datasets:**
- [DeepFashion2](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion2.html)
- [Fashion Product Images (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- Curated collection of 3000+ images

### 3. Train Model

```bash
python train.py --data_dir data/raw --epochs 50 --batch_size 32
```

**Arguments:**
| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/raw` | Dataset directory |
| `--epochs` | 50 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--fine_tune` | False | Unlock MobileNetV2 base layers |

Training automatically:
- Splits data 80/20 train/val
- Uses EarlyStopping (patience=7) + ReduceLROnPlateau
- Generates confusion matrix, ROC curves, confidence distribution
- Converts model to TFLite for mobile deployment

### 4. Run Inference

```bash
python inference.py path/to/image.jpg
```

With TFLite model:
```bash
python inference.py path/to/image.jpg --tflite
```

### 5. Launch Web App

```bash
python app/app.py
```

Open `http://localhost:5000` in your browser. Upload a fashion image for real-time classification.

---

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen base)
    ↓
GlobalAveragePooling2D + Flatten
    ↓
Dense(128, ReLU) + Dropout(0.3)
    ↓
Dense(5, Softmax)
    ↓
Prediction: [Business, Casual, Night Party, Sports, Wedding]
```

---

## TFLite Mobile Deployment

```bash
python tflite/convert_to_tflite.py --model_path saved_models/stylesense_final.keras
python tflite/convert_to_tflite.py --model_path saved_models/stylesense_final.keras --quantize
python tflite/convert_to_tflite.py --model_path saved_models/stylesense_final.keras --edge
```

Output models: `tflite/stylesense_model.tflite`, `tflite/stylesense_model_quantized.tflite`, `tflite/stylesense_model_edge.tflite`

### Flutter / Android Integration

```kotlin
// Load TFLite model in Android
val interpreter = Interpreter(loadModelFile(context, "stylesense_model.tflite"))

// Preprocess image (224×224, normalized)
// Run inference
interpreter.run(inputImage, outputLabels)
```

---

## Citation

```bibtex
@inproceedings{adhatrao2025stylesense,
  title={StyleSense: Real-Time AI Fashion Recommendations using MobileNet},
  author={Adhatrao, Ashlesha S. and Zadbuke, Vaibhavi and Waghamare, Shruti G.},
  booktitle={2025 7th International Conference on Signal Processing, Computing and Control (ISPCC)},
  year={2025}
}
```

## License

MIT License — see `LICENSE` for details.
