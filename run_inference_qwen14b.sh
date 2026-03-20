#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-14B-Instruct}"

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Download a smaller Qwen model first, or override MODEL_DIR."
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-14B-Instruct bash run_inference_qwen14b.sh sample5"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-1.5B-Instruct bash run_inference_qwen14b.sh sample5"
    exit 1
fi

export TOKENIZER_PATH="$MODEL_DIR"

cd "$ROOT_DIR"
bash "$ROOT_DIR/run_inference.sh" "$@"
