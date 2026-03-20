#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-14B-Instruct}"

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Download a smaller Qwen model first, or override MODEL_DIR."
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-14B-Instruct bash run_sample5_and_sample100_qwen14b.sh"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-1.5B-Instruct bash run_sample5_and_sample100_qwen14b.sh"
    exit 1
fi

export TOKENIZER_PATH="$MODEL_DIR"

cd "$ROOT_DIR"

echo "[1/2] Running sample5 with model: $MODEL_DIR"
bash "$ROOT_DIR/run_inference_qwen14b.sh" sample5

echo ""
echo "[2/2] Running sample100 with model: $MODEL_DIR"
bash "$ROOT_DIR/run_inference_qwen14b.sh" sample100

echo ""
echo "All runs completed."
