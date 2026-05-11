#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES
CUDA_DEVICES="${CUDA_DEVICES:-$CUDA_VISIBLE_DEVICES}"
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
    TENSOR_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
fi
export CUDA_DEVICES
export TENSOR_PARALLEL_SIZE

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
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$SCRIPT_DIR/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm_gpu4567.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm_gpu4567.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

usage() {
    cat <<EOF
Usage:
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh sample5
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh sample100
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh sample9999
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh single <dataset_id>
  bash scripts_full/qwen3p5_27b_vllm/run_inference.sh --logging sample5

Behavior:
  - No args: runs sample5 then sample100
  - With args: forwards to run_inference.sh using the Qwen3.5 27B defaults
  - With --logging: writes detailed traces under $ROOT_DIR/27b_vllm_full_logging
  - Re-running the same command auto-resumes the latest incomplete matching run
    unless FORCE_NEW_RUN=1 is set

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen3.5-27B
  CUDA_VISIBLE_DEVICES=4,5,6,7
  CUDA_DEVICES=4,5,6,7
  TENSOR_PARALLEL_SIZE=4
  API_MODEL_NAME=Qwen3.5-27B
  STRUCTRAG_ENABLE_THINKING=0
  MAX_MODEL_LEN=32768
  STRUCTRAG_MAX_INPUT_TOKENS=32768
  SERVER_SCRIPT_PATH=$SCRIPT_DIR/run_server.sh
  SERVER_LOG_PATH=$ROOT_DIR/logs/qwen35_27b_vllm_gpu4567.log
  RESTART_WAIT_TIMEOUT=1800
  RESTART_WAIT_INTERVAL=15
EOF
}

LOGGING_MODE=0
FORWARDED_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        --logging)
            LOGGING_MODE=1
            ;;
        *)
            FORWARDED_ARGS+=("$arg")
            ;;
    esac
done

setup_logging_mode() {
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/27b_vllm_full_logging}"
    local active_run_env="$logging_root/active_run.env"

    mkdir -p "$logging_root"

    if [[ ( -z "${STRUCTRAG_LOGGING_RUN_ID:-}" || -z "${STRUCTRAG_LOGGING_DIR:-}" ) && -f "$active_run_env" ]]; then
        # shellcheck disable=SC1090
        source "$active_run_env"
    fi

    if [[ -z "${STRUCTRAG_LOGGING_RUN_ID:-}" ]]; then
        STRUCTRAG_LOGGING_RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    if [[ -z "${STRUCTRAG_LOGGING_DIR:-}" ]]; then
        STRUCTRAG_LOGGING_DIR="$logging_root/runs/$STRUCTRAG_LOGGING_RUN_ID"
    fi

    mkdir -p "$STRUCTRAG_LOGGING_DIR/server" "$STRUCTRAG_LOGGING_DIR/inference" "$STRUCTRAG_LOGGING_DIR/samples"
    ln -sfn "runs/$STRUCTRAG_LOGGING_RUN_ID" "$logging_root/latest"

    if [[ ! -f "$active_run_env" ]]; then
        cat > "$active_run_env" <<EOF
export STRUCTRAG_LOGGING=1
export STRUCTRAG_LOGGING_ROOT="$logging_root"
export STRUCTRAG_LOGGING_RUN_ID="$STRUCTRAG_LOGGING_RUN_ID"
export STRUCTRAG_LOGGING_DIR="$STRUCTRAG_LOGGING_DIR"
EOF
    fi

    cat > "$STRUCTRAG_LOGGING_DIR/inference/launch.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
API_MODEL_NAME=$API_MODEL_NAME
STRUCTRAG_ENABLE_THINKING=$STRUCTRAG_ENABLE_THINKING
SERVER_LOG_PATH=$SERVER_LOG_PATH
SERVER_PID_FILE=$SERVER_PID_FILE
ARGV=${FORWARDED_ARGS[*]:-default}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Checked candidates:"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        echo "  - $candidate"
    done
    echo "Prior LAMBO v2 logs used: /workspace/lambo/models/Qwen3.5-27B"
    echo "Example:"
    echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts_full/qwen3p5_27b_vllm/run_inference.sh sample5"
    echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts_full/qwen3p5_27b_vllm/run_inference.sh"
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

if [[ "$LOGGING_MODE" -eq 1 ]]; then
    setup_logging_mode
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi

cd "$ROOT_DIR"

if [[ ${#FORWARDED_ARGS[@]} -eq 0 ]]; then
    echo "[1/2] Running sample5 with model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample5

    echo ""
    echo "[2/2] Running sample100 with model: $MODEL_DIR"
    bash "$ROOT_DIR/run_inference.sh" sample100

    echo ""
    echo "All runs completed."
else
    bash "$ROOT_DIR/run_inference.sh" "${FORWARDED_ARGS[@]}"
fi
