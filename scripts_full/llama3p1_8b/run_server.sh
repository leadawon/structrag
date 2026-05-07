#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
CUDA_DEVICES="${CUDA_DEVICES:-$CUDA_VISIBLE_DEVICES}"
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
    TENSOR_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
fi
export CUDA_DEVICES
export TENSOR_PARALLEL_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Select a Python that actually has vllm installed.
# structragvenv is used for the inference client (no vllm needed there);
# the server needs vllm, so we search explicitly.
vllm_capable_python() {
    local py="$1"
    [[ -x "$py" ]] && "$py" -c "import vllm" >/dev/null 2>&1
}

if [[ -z "${SERVER_PYTHON_BIN:-}" ]]; then
    for _candidate in \
        "${PYTHON_BIN:-}" \
        /workspace/venvs/structragvenv/bin/python \
        /workspace/venvs/ragteamvenv/bin/python \
        /opt/conda/bin/python3 \
        python3
    do
        [[ -z "$_candidate" ]] && continue
        if vllm_capable_python "$_candidate"; then
            SERVER_PYTHON_BIN="$_candidate"
            break
        fi
    done
    if [[ -z "${SERVER_PYTHON_BIN:-}" ]]; then
        echo "No Python with vllm found. Install vllm first:"
        echo "  pip install vllm"
        exit 1
    fi
fi

if [[ -z "${MODEL_DIR:-}" ]]; then
    MODEL_CANDIDATES=(
        "$ROOT_DIR/model/Llama-3.1-8B-Instruct"
        "$ROOT_DIR/model/Meta-Llama-3.1-8B-Instruct"
        "/workspace/models/Llama-3.1-8B-Instruct"
        "/workspace/models/Meta-Llama-3.1-8B-Instruct"
        "/workspace/llama/Llama-3.1-8B-Instruct"
    )
    MODEL_DIR="${MODEL_CANDIDATES[0]}"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        if [[ -e "$candidate" ]]; then
            MODEL_DIR="$candidate"
            break
        fi
    done
fi

DEFAULT_LOG_PATH="$ROOT_DIR/logs/llama31_8b_vllm.log"
DEFAULT_PID_FILE="$ROOT_DIR/logs/llama31_8b_vllm.pid"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN:-0}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"
PID_FILE="${PID_FILE:-$DEFAULT_PID_FILE}"
PGID_FILE="${PGID_FILE:-${PID_FILE}.pgid}"

usage() {
    cat <<EOF
Usage:
  bash scripts_full/llama3p1_8b/run_server.sh
  bash scripts_full/llama3p1_8b/run_server.sh --detach
  bash scripts_full/llama3p1_8b/run_server.sh --stop

Defaults:
  MODEL_DIR=$MODEL_DIR
  SERVER_PYTHON_BIN=$SERVER_PYTHON_BIN
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
  MAX_MODEL_LEN=$MAX_MODEL_LEN
  SERVED_MODEL_NAME=$SERVED_MODEL_NAME
  DTYPE=$DTYPE
  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
  MAX_NUM_SEQS=$MAX_NUM_SEQS
  ENFORCE_EAGER=$ENFORCE_EAGER

Examples:
  bash scripts_full/llama3p1_8b/run_server.sh --detach
  MODEL_DIR=/path/to/Llama-3.1-8B-Instruct bash scripts_full/llama3p1_8b/run_server.sh --detach
  CUDA_VISIBLE_DEVICES=1 bash scripts_full/llama3p1_8b/run_server.sh --detach
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Download first:"
    echo "  bash scripts_full/llama3p1_8b/download_model.sh"
    echo "Or specify:"
    echo "  MODEL_DIR=/path/to/Llama-3.1-8B-Instruct bash scripts_full/llama3p1_8b/run_server.sh $*"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export SERVED_MODEL_NAME
export MAX_MODEL_LEN
export DTYPE
export GPU_MEMORY_UTILIZATION
export MAX_NUM_SEQS
export ENFORCE_EAGER
export ALLOW_LONG_MAX_MODEL_LEN
export CLEAN_STALE_VLLM
export DISABLE_CUSTOM_ALL_REDUCE
export LOG_PATH
export PID_FILE
export PGID_FILE
export PYTHON_BIN="$SERVER_PYTHON_BIN"

cd "$ROOT_DIR"
bash "$ROOT_DIR/run_server.sh" "$@"
