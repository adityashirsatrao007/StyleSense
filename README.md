# StyleSense

### Real-Time AI Fashion Recommendations using MobileNet

**ISPCC 2025** | 7th International Conference on Signal Processing, Computing and Control  
Track 3 — Advanced Computational Intelligence Systems

---

## Authors

| Name | Affiliation | Email |
|------|-------------|-------|
| Prof. Ashlesha S. Adhatrao | Dept. of AI & Data Science, N K Orchid College of Engg & Tech, Solapur | ashleshaadhatrao@gmail.com |
| Ms. Vaibhavi Zadbuke | Dept. of AI & Data Science, N K Orchid College of Engg & Tech, Solapur | zadbukevaibhavi@gmail.com |
| Ms. Shruti G. Waghamare | Dept. of AI & Data Science, N K Orchid College of Engg & Tech, Solapur | waghmareshruti944@gmail.com |

---

## Abstract

This paper presents StyleSense, a lightweight MobileNetV2-based framework for real-time personalized fashion recommendation. The system classifies clothing images into five social contexts — Business, Casual, Night Party, Sports, and Wedding — achieving 96.00% validation accuracy and 0.95 weighted F1-score. The TFLite model is 2.73 MB with 33 ms inference latency on mobile CPU, enabling on-device deployment.

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 96.00% |
| Weighted F1-Score | 0.95 |
| Precision / Recall | 0.95 / 0.95 |
| Test Loss | 0.18 |
| AUC-ROC | 0.96 |
| TFLite Model Size | 2.73 MB |
| Inference Latency (CPU) | 33.16 ms |
| Total Parameters | 2,588,229 (2.59 M) |

---

## Repository Structure

```
StyleSense/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── model.py                       # MobileNetV2 architecture & callbacks
│   ├── preprocess.py                  # tf.data pipeline & data loading
│   ├── evaluate.py                    # Metrics, confusion matrix, ROC curves
│   └── predict.py                     # Inference engine (Keras + TFLite)
│
├── train.py                           # Training pipeline CLI
├── inference.py                       # Single-image inference CLI
├── config.py                          # Configuration & hyperparameters
│
├── paper/                             # IEEE paper materials
│   ├── manuscript/                    # DOCX, paste sections
│   ├── figures/                       # All paper figures (PNG + drawio)
│   │   ├── Fig1_architecture.png
│   │   ├── Fig2_workflow.png
│   │   ├── Fig3_roc_auc.png
│   │   ├── Fig4_unified_framework.png
│   │   ├── Fig5_performance_metrics.png
│   │   ├── Fig6_confusion_matrix.png
│   │   ├── Fig7_confidence_dist.png
│   │   └── Fig8_training_history.png
│   └── reviewer_response/             # Reviewer response + generator
│
├── tflite/                            # TFLite model & conversion
│   ├── stylesense_model.tflite
│   └── convert_to_tflite.py
│
├── app/                               # Flask web demo
│   └── app.py
│
├── configs/                           # Training configs
├── scripts/                           # Utility scripts
├── notebooks/                         # Jupyter notebooks
└── data/                              # Dataset (not tracked)
```

---

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/adityashirsatrao007/StyleSense.git
cd StyleSense
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize images by class in `data/raw/`:

```
data/raw/
├── Business/
├── Casual/
├── Night Party/
├── Sports/
└── Wedding/
```

```bash
python data/dataset_setup.py --action structure
```

### 3. Train

```bash
# Phase 1: Frozen backbone
python train.py --data_dir data/raw --epochs 26 --batch_size 24

# Phase 2: Fine-tune
python train.py --data_dir data/raw --epochs 54 --batch_size 24 --fine_tune
```

### 4. Inference

```bash
python inference.py path/to/image.jpg
python inference.py path/to/image.jpg --tflite
```

### 5. Web Demo

```bash
python app/app.py
# Open http://localhost:5000
```

---

## Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen base)
    ↓
GlobalAveragePooling2D + Flatten
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(5, Softmax)
    ↓
Output: [Business | Casual | Night Party | Sports | Wedding]
```

---

## TFLite Mobile Deployment

```bash
python tflite/convert_to_tflite.py --model_path saved_models/stylesense_best_ft.keras
```

Output: `tflite/stylesense_model.tflite` (2.73 MB)

---

## Citation

```bibtex
@inproceedings{adhatrao2025stylesense,
  title     = {StyleSense: Real-Time AI Fashion Recommendations using MobileNet},
  author    = {Adhatrao, Ashlesha S. and Zadbuke, Vaibhavi and Waghamare, Shruti G.},
  booktitle = {2025 7th International Conference on Signal Processing, Computing and Control (ISPCC)},
  year      = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
