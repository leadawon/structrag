#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage:
  bash scripts/7b/run_inference.sh
  bash scripts/7b/run_inference.sh sample5
  bash scripts/7b/run_inference.sh sample100
  bash scripts/7b/run_inference.sh sample9999
  bash scripts/7b/run_inference.sh single <dataset_id>

Behavior:
  - No args: runs sample5 then sample100
  - With args: forwards to run_inference.sh using the 7B model defaults

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct
  MAX_MODEL_LEN=32768
  STRUCTRAG_MAX_INPUT_TOKENS=32768
EOF
    exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct bash scripts/7b/run_inference.sh sample5"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct bash scripts/7b/run_inference.sh"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export STRUCTRAG_MAX_INPUT_TOKENS

cd "$ROOT_DIR"

if [[ $# -eq 0 ]]; then
    echo "[1/2] Running sample5 with model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample5

    echo ""
    echo "[2/2] Running sample100 with model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample100

    echo ""
    echo "All runs completed."
else
    bash "$ROOT_DIR/run_inference.sh" "$@"
fi
