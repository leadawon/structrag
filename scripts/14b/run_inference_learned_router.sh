#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-14B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
ROUTER_URL="${ROUTER_URL:-127.0.0.1:1226}"
ROUTER_TOKENIZER_PATH="${ROUTER_TOKENIZER_PATH:-$MODEL_DIR}"
ROUTER_API_MODEL_NAME="${ROUTER_API_MODEL_NAME:-Qwen}"
ROUTER_LABEL="${ROUTER_LABEL:-learned-router}"
ROUTER_DISABLE_GUIDED_DECODING="${ROUTER_DISABLE_GUIDED_DECODING:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage:
  bash scripts/14b/run_inference_learned_router.sh
  bash scripts/14b/run_inference_learned_router.sh sample5
  bash scripts/14b/run_inference_learned_router.sh sample100
  bash scripts/14b/run_inference_learned_router.sh sample9999
  bash scripts/14b/run_inference_learned_router.sh single <dataset_id>

Behavior:
  - No args: runs sample5 then sample100
  - Uses the main 14B model plus an external learned router at \$ROUTER_URL

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-14B-Instruct
  ROUTER_URL=127.0.0.1:1226
  ROUTER_TOKENIZER_PATH=$ROOT_DIR/model/Qwen2.5-14B-Instruct
  ROUTER_LABEL=learned-router
EOF
    exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    exit 1
fi

if [[ ! -e "$ROUTER_TOKENIZER_PATH" ]]; then
    echo "Router tokenizer path not found: $ROUTER_TOKENIZER_PATH"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export STRUCTRAG_MAX_INPUT_TOKENS
export ROUTER_URL
export ROUTER_TOKENIZER_PATH
export ROUTER_API_MODEL_NAME
export ROUTER_LABEL
export ROUTER_DISABLE_GUIDED_DECODING

cd "$ROOT_DIR"

if [[ $# -eq 0 ]]; then
    echo "[1/2] Running sample5 with learned router and model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample5

    echo ""
    echo "[2/2] Running sample100 with learned router and model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample100

    echo ""
    echo "All learned-router runs completed."
else
    bash "$ROOT_DIR/run_inference.sh" "$@"
fi
