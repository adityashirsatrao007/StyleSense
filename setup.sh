#!/usr/bin/env bash
# StyleSense — IEEE-ready project setup script
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  StyleSense — Project Setup"
echo "========================================"

# 1. Check Python 3.10+
PYTHON=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        if [ "$(echo "$VER" | cut -d. -f1)" -ge 3 ] && [ "$(echo "$VER" | cut -d. -f2)" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Found: $("$PYTHON" --version 2>/dev/null || echo 'none')"
    exit 1
fi
echo "[1/4] Using Python: $("$PYTHON" --version)"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow matplotlib seaborn scikit-learn numpy opencv-python flask Pillow
echo "  Dependencies installed successfully."

# 4. Create directories
echo "[4/4] Creating project directories..."
mkdir -p data/raw data/processed saved_models tflite paper_figures uploads
echo "  Project directories ready."

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  Next steps:"
echo "  1. Place dataset in data/raw/{Business,Casual,NightParty,Sports,Wedding}/"
echo "     OR run synthetic demo:  python data/create_demo_data.py"
echo ""
echo "  2. Train model:            python train.py --data_dir data/raw"
echo "     (omit --data_dir if you placed data in data/raw/)"
echo ""
echo "  3. Generate all figures:   python train.py"
echo "     Architecture diagrams:  python src/paper_diagrams.py"
echo ""
echo "  4. Run web app:            python app/app.py"
echo "     → http://localhost:5000"
echo ""
echo "  5. Activate venv later:    source .venv/bin/activate"
echo "========================================"
