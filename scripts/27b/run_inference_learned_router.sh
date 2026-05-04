#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${MODEL_DIR:-}" ]]; then
    MODEL_CANDIDATES=(
        "$ROOT_DIR/model/Qwen3.5-27B"
        "$ROOT_DIR/model/Qwen3.5-27B-Instruct"
        "/workspace/lambo/models/Qwen3.5-27B"
        "/workspace/LAMBO/models/Qwen3.5-27B"
        "/workspace/qwen/Qwen3.5-27B"
        "/workspace/qwen/Qwen3.5-27B-Instruct"
    )
    MODEL_DIR="${MODEL_CANDIDATES[0]}"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        if [[ -e "$candidate" ]]; then
            MODEL_DIR="$candidate"
            break
        fi
    done
else
    MODEL_CANDIDATES=("$MODEL_DIR")
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen3.5-27B}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/27b/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"
ROUTER_URL="${ROUTER_URL:-127.0.0.1:1226}"
ROUTER_TOKENIZER_PATH="${ROUTER_TOKENIZER_PATH:-$MODEL_DIR}"
ROUTER_API_MODEL_NAME="${ROUTER_API_MODEL_NAME:-$API_MODEL_NAME}"
ROUTER_LABEL="${ROUTER_LABEL:-learned-router}"
ROUTER_DISABLE_GUIDED_DECODING="${ROUTER_DISABLE_GUIDED_DECODING:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage:
  bash scripts/27b/run_inference_learned_router.sh
  bash scripts/27b/run_inference_learned_router.sh sample5
  bash scripts/27b/run_inference_learned_router.sh sample100
  bash scripts/27b/run_inference_learned_router.sh sample9999
  bash scripts/27b/run_inference_learned_router.sh single <dataset_id>

Behavior:
  - No args: runs sample5 then sample100
  - Uses the main Qwen3.5 27B model plus an external learned router at \$ROUTER_URL

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen3.5-27B
  API_MODEL_NAME=Qwen3.5-27B
  STRUCTRAG_ENABLE_THINKING=0
  SERVER_SCRIPT_PATH=$ROOT_DIR/scripts/27b/run_server.sh
  ROUTER_URL=127.0.0.1:1226
  ROUTER_TOKENIZER_PATH=$ROOT_DIR/model/Qwen3.5-27B
  ROUTER_API_MODEL_NAME=Qwen3.5-27B
  ROUTER_LABEL=learned-router
EOF
    exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Checked candidates:"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        echo "  - $candidate"
    done
    echo "Prior LAMBO v2 logs used: /workspace/lambo/models/Qwen3.5-27B"
    exit 1
fi

if [[ ! -e "$ROUTER_TOKENIZER_PATH" ]]; then
    echo "Router tokenizer path not found: $ROUTER_TOKENIZER_PATH"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
export STRUCTRAG_ENABLE_THINKING
export MAX_MODEL_LEN
export STRUCTRAG_MAX_INPUT_TOKENS
export SERVER_SCRIPT_PATH
export SERVER_LOG_PATH
export SERVER_PID_FILE
export SERVER_PGID_FILE
export LOG_PATH
export PID_FILE
export PGID_FILE
export CLEAN_STALE_VLLM
export DISABLE_CUSTOM_ALL_REDUCE
export RESTART_WAIT_TIMEOUT
export RESTART_WAIT_INTERVAL
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
