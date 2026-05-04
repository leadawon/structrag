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
if [[ -z "${SERVER_PYTHON_BIN:-}" ]]; then
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        SERVER_PYTHON_BIN="$PYTHON_BIN"
    elif [[ -x /workspace/venvs/ragteamvenv/bin/python ]]; then
        SERVER_PYTHON_BIN="/workspace/venvs/ragteamvenv/bin/python"
    else
        SERVER_PYTHON_BIN="/workspace/venvs/structrag/bin/python"
    fi
fi
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
DEFAULT_LOG_PATH="$ROOT_DIR/logs/qwen35_27b_vllm.log"
DEFAULT_PID_FILE="$ROOT_DIR/logs/qwen35_27b_vllm.pid"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-27B}"
DTYPE="${DTYPE:-bfloat16}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
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
  bash scripts/27b/run_server.sh
  bash scripts/27b/run_server.sh --detach
  bash scripts/27b/run_server.sh --stop
  bash scripts/27b/run_server.sh --logging
  bash scripts/27b/run_server.sh --logging --detach
  bash scripts/27b/run_server.sh --logging --stop

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen3.5-27B
  SERVER_PYTHON_BIN=/workspace/venvs/ragteamvenv/bin/python
  CUDA_VISIBLE_DEVICES=0,1,2,3
  CUDA_DEVICES=0,1,2,3
  TENSOR_PARALLEL_SIZE=4
  MAX_MODEL_LEN=32768
  SERVED_MODEL_NAME=Qwen3.5-27B
  DTYPE=bfloat16
  REASONING_PARSER=qwen3
  GPU_MEMORY_UTILIZATION=<unset; use vLLM default unless overridden>
  MAX_NUM_SEQS=<unset; use vLLM default unless overridden>
  DISABLE_CUSTOM_ALL_REDUCE=0
  ENFORCE_EAGER=1
  ALLOW_LONG_MAX_MODEL_LEN=0
  CLEAN_STALE_VLLM=1

Logging mode:
  STRUCTRAG_LOGGING_ROOT=$ROOT_DIR/27b_logging
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
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/27b_logging}"
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
export PGID_FILE="$PGID_FILE"
export MODEL_DIR="$MODEL_DIR"
EOF

    cat > "$STRUCTRAG_LOGGING_DIR/server/run_server.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
SERVER_PYTHON_BIN=$SERVER_PYTHON_BIN
LOG_PATH=$LOG_PATH
PID_FILE=$PID_FILE
PGID_FILE=$PGID_FILE
DISABLE_CUSTOM_ALL_REDUCE=$DISABLE_CUSTOM_ALL_REDUCE
CLEAN_STALE_VLLM=$CLEAN_STALE_VLLM
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
    echo "Checked candidates:"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        echo "  - $candidate"
    done
    echo "Prior LAMBO v2 logs used: /workspace/lambo/models/Qwen3.5-27B"
    echo "Example:"
    echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts/27b/run_server.sh"
    echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts/27b/run_server.sh --detach"
    exit 1
fi

if [[ ! -x "$SERVER_PYTHON_BIN" ]]; then
    echo "Server Python binary not found: $SERVER_PYTHON_BIN"
    echo "Set SERVER_PYTHON_BIN=/path/to/python."
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export PYTHON_BIN="$SERVER_PYTHON_BIN"
export MAX_MODEL_LEN
export SERVED_MODEL_NAME
export DTYPE
export REASONING_PARSER
export GPU_MEMORY_UTILIZATION
export MAX_NUM_SEQS
export DISABLE_CUSTOM_ALL_REDUCE
export ENFORCE_EAGER
export ALLOW_LONG_MAX_MODEL_LEN
export CLEAN_STALE_VLLM
export LOG_PATH
export PID_FILE
export PGID_FILE

cd "$ROOT_DIR"
if [[ "$LOGGING_MODE" -eq 1 ]]; then
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi
bash "$ROOT_DIR/run_server.sh" "${FORWARDED_ARGS[@]}"
