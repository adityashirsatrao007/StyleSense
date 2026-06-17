import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, render_template_string
import os
import config

from src.predict import StyleSensePredictor

app = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

predictor = None


def load_model():
    global predictor
    model_path = config.SAVED_MODELS_DIR / "stylesense_best.keras"
    tflite_path = config.TFLITE_DIR / "stylesense_model.tflite"

    if model_path.exists():
        predictor = StyleSensePredictor(model_path=str(model_path))
        print(f"Loaded Keras model: {model_path}")
    elif tflite_path.exists():
        predictor = StyleSensePredictor(tflite_path=str(tflite_path))
        print(f"Loaded TFLite model: {tflite_path}")
    else:
        print("WARNING: No trained model found. Run train.py first.")
        print(f"  Expected at: {model_path}")
        print(f"  Or TFLite at: {tflite_path}")
        predictor = None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StyleSense - AI Fashion Recommendation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            padding: 20px;
        }
        .container {
            background: white; border-radius: 20px; padding: 40px; max-width: 600px; width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { font-size: 28px; margin-bottom: 8px; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
        .upload-box {
            border: 2px dashed #ddd; border-radius: 12px; padding: 40px; text-align: center;
            transition: all 0.3s; cursor: pointer; margin-bottom: 20px;
        }
        .upload-box:hover { border-color: #667eea; background: #f8f9ff; }
        .upload-box input { display: none; }
        .upload-label { color: #667eea; font-weight: 600; cursor: pointer; }
        #preview { max-width: 100%; max-height: 300px; margin-top: 16px; border-radius: 8px; display: none; }
        .btn {
            width: 100%; padding: 14px; border: none; border-radius: 10px; font-size: 16px;
            font-weight: 600; cursor: pointer; transition: all 0.3s;
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(102,126,234,0.4); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result {
            margin-top: 24px; padding: 20px; border-radius: 12px; display: none;
        }
        .result.show { display: block; }
        .result.success { background: #f0fdf4; border: 1px solid #86efac; }
        .result.error { background: #fef2f2; border: 1px solid #fca5a5; }
        .prediction-class { font-size: 24px; font-weight: 700; color: #111; margin-bottom: 4px; }
        .prediction-conf { font-size: 16px; color: #666; margin-bottom: 16px; }
        .prob-bar { display: flex; align-items: center; margin-bottom: 6px; }
        .prob-label { width: 100px; font-size: 13px; color: #444; }
        .prob-fill-bg { flex: 1; height: 18px; background: #e5e7eb; border-radius: 9px; overflow: hidden; margin: 0 8px; }
        .prob-fill { height: 100%; border-radius: 9px; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.5s; }
        .prob-val { width: 50px; font-size: 12px; color: #666; text-align: right; }
        .spinner { display: none; width: 20px; height: 20px; border: 3px solid #e5e7eb; border-top-color: #667eea; border-radius: 50%; animation: spin 0.6s linear infinite; margin: 0 auto 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error-msg { color: #dc2626; font-size: 14px; }
    </style>
</head>
<body>
<div class="container">
    <h1>StyleSense</h1>
    <p class="subtitle">Real-Time AI Fashion Recommendations using MobileNet</p>

    <div class="upload-box" id="uploadBox">
        <input type="file" id="imageInput" accept="image/*">
        <p class="upload-label">Click to upload a fashion image</p>
        <p style="color:#999;font-size:13px;margin-top:4px;">or drag & drop</p>
        <img id="preview" alt="Preview">
    </div>

    <button class="btn" id="predictBtn" disabled>
        <div class="spinner" id="spinner"></div>
        <span id="btnText">Analyze Style</span>
    </button>

    <div class="result" id="result">
        <div class="prediction-class" id="predClass"></div>
        <div class="prediction-conf" id="predConf"></div>
        <div id="probBars"></div>
    </div>

    <div class="result error" id="errorResult">
        <div class="error-msg" id="errorMsg"></div>
    </div>
</div>

<script>
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    const result = document.getElementById('result');
    const errorResult = document.getElementById('errorResult');
    const predClass = document.getElementById('predClass');
    const predConf = document.getElementById('predConf');
    const probBars = document.getElementById('probBars');
    const errorMsg = document.getElementById('errorMsg');

    let selectedFile = null;

    uploadBox.addEventListener('click', () => imageInput.click());
    uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.style.borderColor = '#667eea'; });
    uploadBox.addEventListener('dragleave', () => { uploadBox.style.borderColor = '#ddd'; });
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault(); uploadBox.style.borderColor = '#ddd';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            uploadBox.querySelector('p:first-of-type').textContent = file.name;
        };
        reader.readAsDataURL(file);
        predictBtn.disabled = false;
        result.classList.remove('show');
        errorResult.classList.remove('show');
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        predictBtn.disabled = true;
        spinner.style.display = 'block';
        btnText.textContent = 'Analyzing...';
        result.classList.remove('show');
        errorResult.classList.remove('show');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) {
                errorMsg.textContent = data.error;
                errorResult.classList.add('show');
                return;
            }

            predClass.textContent = data.predicted_class;
            predConf.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            probBars.innerHTML = '';

            const classes = Object.entries(data.class_probabilities)
                .sort((a, b) => b[1] - a[1]);
            const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'];
            classes.forEach(([cls, prob], i) => {
                const bar = document.createElement('div');
                bar.className = 'prob-bar';
                bar.innerHTML = `
                    <span class="prob-label">${cls}</span>
                    <div class="prob-fill-bg">
                        <div class="prob-fill" style="width:${prob * 100}%;background:${colors[i % colors.length]}"></div>
                    </div>
                    <span class="prob-val">${(prob * 100).toFixed(1)}%</span>
                `;
                probBars.appendChild(bar);
            });

            result.classList.add('show');
            result.className = 'result success show';
        } catch (err) {
            errorMsg.textContent = 'Failed to analyze image. Is the model loaded?';
            errorResult.classList.add('show');
        } finally {
            predictBtn.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Analyze Style';
        }
    });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded. Train a model first (run train.py)"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = UPLOAD_FOLDER / file.filename
    file.save(filepath)

    try:
        result = predictor.predict(str(filepath))
        os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        if filepath.exists():
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": predictor is not None,
            "classes": config.CLASS_NAMES,
        }
    )


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
