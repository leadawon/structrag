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
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
DEFAULT_LOG_PATH="$ROOT_DIR/logs/qwen32b_vllm.log"
DEFAULT_PID_FILE="$ROOT_DIR/logs/qwen32b_vllm.pid"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN:-0}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"
PID_FILE="${PID_FILE:-$DEFAULT_PID_FILE}"

usage() {
    cat <<EOF
Usage:
  bash scripts/32b/run_server.sh
  bash scripts/32b/run_server.sh --detach
  bash scripts/32b/run_server.sh --stop
  bash scripts/32b/run_server.sh --logging
  bash scripts/32b/run_server.sh --logging --detach
  bash scripts/32b/run_server.sh --logging --stop

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  CUDA_VISIBLE_DEVICES=0,1,2,3
  CUDA_DEVICES=0,1,2,3
  TENSOR_PARALLEL_SIZE=4
  MAX_MODEL_LEN=32768
  GPU_MEMORY_UTILIZATION=0.88
  MAX_NUM_SEQS=1
  ENFORCE_EAGER=1
  ALLOW_LONG_MAX_MODEL_LEN=0

Logging mode:
  STRUCTRAG_LOGGING_ROOT=$ROOT_DIR/32b_logging
EOF
}

DETACH_MODE=0
STOP_MODE=0
LOGGING_MODE=0
FORWARDED_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        --detach)
            DETACH_MODE=1
            FORWARDED_ARGS+=("$arg")
            ;;
        --stop)
            STOP_MODE=1
            FORWARDED_ARGS+=("$arg")
            ;;
        --logging)
            LOGGING_MODE=1
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

if [[ "$DETACH_MODE" -eq 1 && "$STOP_MODE" -eq 1 ]]; then
    echo "--detach and --stop cannot be used together."
    exit 1
fi

setup_logging_mode() {
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/32b_logging}"
    local active_run_env="$logging_root/active_run.env"

    mkdir -p "$logging_root"

    if [[ "$STOP_MODE" -eq 1 && -z "${STRUCTRAG_LOGGING_DIR:-}" && -f "$active_run_env" ]]; then
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

    if [[ "$LOG_PATH" == "$DEFAULT_LOG_PATH" ]]; then
        LOG_PATH="$STRUCTRAG_LOGGING_DIR/server/vllm.log"
    fi
    if [[ "$PID_FILE" == "$DEFAULT_PID_FILE" ]]; then
        PID_FILE="$STRUCTRAG_LOGGING_DIR/server/vllm.pid"
    fi

    cat > "$active_run_env" <<EOF
export STRUCTRAG_LOGGING=1
export STRUCTRAG_LOGGING_ROOT="$logging_root"
export STRUCTRAG_LOGGING_RUN_ID="$STRUCTRAG_LOGGING_RUN_ID"
export STRUCTRAG_LOGGING_DIR="$STRUCTRAG_LOGGING_DIR"
export LOG_PATH="$LOG_PATH"
export PID_FILE="$PID_FILE"
export MODEL_DIR="$MODEL_DIR"
EOF

    cat > "$STRUCTRAG_LOGGING_DIR/server/run_server.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
LOG_PATH=$LOG_PATH
PID_FILE=$PID_FILE
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

if [[ "$LOGGING_MODE" -eq 1 ]]; then
    setup_logging_mode
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash scripts/32b/run_server.sh"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash scripts/32b/run_server.sh --detach"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION
export MAX_NUM_SEQS
export ENFORCE_EAGER
export ALLOW_LONG_MAX_MODEL_LEN
export LOG_PATH
export PID_FILE

cd "$ROOT_DIR"
if [[ "$LOGGING_MODE" -eq 1 ]]; then
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi
bash "$ROOT_DIR/run_server.sh" "${FORWARDED_ARGS[@]}"
