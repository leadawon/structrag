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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/32b/run_server.sh}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

usage() {
    cat <<EOF
Usage:
  bash scripts/32b/run_inference.sh
  bash scripts/32b/run_inference.sh sample5
  bash scripts/32b/run_inference.sh sample100
  bash scripts/32b/run_inference.sh sample9999
  bash scripts/32b/run_inference.sh single <dataset_id>
  bash scripts/32b/run_inference.sh --logging sample5

Behavior:
  - No args: runs sample5 then sample100
  - With args: forwards to run_inference.sh using the 32B model defaults
  - With --logging: writes detailed traces under $ROOT_DIR/32b_logging
  - Re-running the same command auto-resumes the latest incomplete matching run
    unless FORCE_NEW_RUN=1 is set

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  CUDA_VISIBLE_DEVICES=0,1,2,3
  CUDA_DEVICES=0,1,2,3
  TENSOR_PARALLEL_SIZE=4
  MAX_MODEL_LEN=32768
  STRUCTRAG_MAX_INPUT_TOKENS=32768
  SERVER_SCRIPT_PATH=$ROOT_DIR/scripts/32b/run_server.sh
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
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/32b_logging}"
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
ARGV=${FORWARDED_ARGS[*]:-default}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash scripts/32b/run_inference.sh sample5"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash scripts/32b/run_inference.sh"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export STRUCTRAG_MAX_INPUT_TOKENS
export SERVER_SCRIPT_PATH
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
