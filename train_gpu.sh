#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.venv/bin/activate"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$(find "$DIR/.venv" -path '*/nvidia/cuda_nvcc' -type d | head -1)"
export LD_LIBRARY_PATH=$(find "$DIR/.venv" -type d -name "lib" -path "*/nvidia/*" | tr '\n' ':'):$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=1

exec python3 "$DIR/train.py" --data_dir "$DIR/data/raw" --epochs 80 --batch_size 32 --fine_tune
