#!/usr/bin/env bash

# Run StructRAG on the full Loong dataset (all 1600 samples) with
# Qwen2-72B-Instruct in float16 precision (8 GPUs).
#
# Same model weights as scripts/72b/ (bfloat16); only dtype differs.
# Useful for environments where float16 is preferred or for ablation comparison.

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
CUDA_DEVICES="${CUDA_DEVICES:-$CUDA_VISIBLE_DEVICES}"
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
    TENSOR_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
fi
export CUDA_DEVICES
export TENSOR_PARALLEL_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

model_ready() {
    local model_dir="$1"
    [[ -d "$model_dir" ]] || return 1
    [[ -f "$model_dir/config.json" ]] || return 1
    if [[ ! -f "$model_dir/tokenizer.json" && ! -f "$model_dir/tokenizer.model" && ! -f "$model_dir/tokenizer_config.json" ]]; then
        return 1
    fi
    if [[ ! -f "$model_dir/model.safetensors.index.json" ]] && ! compgen -G "$model_dir/*.safetensors" >/dev/null; then
        return 1
    fi
    return 0
}

if [[ -z "${MODEL_DIR:-}" ]]; then
    MODEL_CANDIDATES=(
        "$ROOT_DIR/model/Qwen2-72B-Instruct"
        "/workspace/lambo/models/Qwen2-72B-Instruct"
        "/workspace/LAMBO/models/Qwen2-72B-Instruct"
        "/workspace/qwen/Qwen2-72B-Instruct"
    )
    MODEL_DIR="${MODEL_CANDIDATES[0]}"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        if model_ready "$candidate"; then
            MODEL_DIR="$candidate"
            break
        fi
    done
else
    MODEL_CANDIDATES=("$MODEL_DIR")
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen2-72B-Instruct-fp16}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-qwen2-72b-fp16-full}"
DTYPE="${DTYPE:-float16}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/72b_16bit/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen2_72b_fp16_vllm.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen2_72b_fp16_vllm.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

WORKER_COUNT="${WORKER_COUNT:-8}"
DATASET_NAME="${DATASET_NAME:-loong}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-1}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-/workspace/LAMBO}"

usage() {
    cat <<EOF
Usage:
  bash scripts/72b_16bit/run_inference_full.sh
  bash scripts/72b_16bit/run_inference_full.sh --logging

Behavior:
  - Runs StructRAG inference on the full Loong dataset (1600 samples)
  - Uses 8 workers (worker_id 0-7), 200 items each, --no_shuffle
  - Runs with Qwen2-72B-Instruct in float16 on 8 GPUs
  - After inference, scores using the same model as LLM judge
  - Saves EM-style structured metrics plus Loong LLM-as-eval metrics

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2-72B-Instruct  (shared with scripts/72b/)
  DATASET_NAME=loong
  WORKER_COUNT=8
  API_MODEL_NAME=Qwen2-72B-Instruct-fp16
  DTYPE=float16
  STRUCTRAG_ENABLE_THINKING=0
  OUTPUT_PATH_SUFFIX=qwen2-72b-fp16-full
  MAX_MODEL_LEN=32768
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  INCLUDE_ERROR_OUTPUTS_IN_SCORE=1

Examples:
  bash scripts/72b_16bit/run_inference_full.sh
  bash scripts/72b_16bit/run_inference_full.sh --logging
  FORCE_NEW_RUN=1 bash scripts/72b_16bit/run_inference_full.sh
  OUTPUT_PATH_SUFFIX=qwen2-72b-fp16-full-v2 bash scripts/72b_16bit/run_inference_full.sh
EOF
}

LOGGING_MODE=0
FORWARDED_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --logging) LOGGING_MODE=1 ;;
        *) FORWARDED_ARGS+=("$arg") ;;
    esac
done

setup_logging_mode() {
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/72b_16bit_logging}"
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

    cat > "$STRUCTRAG_LOGGING_DIR/inference/launch_full.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
API_MODEL_NAME=$API_MODEL_NAME
DTYPE=$DTYPE
STRUCTRAG_ENABLE_THINKING=$STRUCTRAG_ENABLE_THINKING
OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX
DATASET_NAME=$DATASET_NAME
WORKER_COUNT=$WORKER_COUNT
ARGV=${FORWARDED_ARGS[*]:-all_workers}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

if ! model_ready "$MODEL_DIR"; then
    echo "Model path is missing or incomplete: $MODEL_DIR"
    echo "Checked candidates:"
    for candidate in "${MODEL_CANDIDATES[@]}"; do echo "  - $candidate"; done
    echo "Download (or reuse from scripts/72b/):"
    echo "  bash scripts/72b_16bit/download_model.sh"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"; exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
export STRUCTRAG_ENABLE_THINKING
export OUTPUT_PATH_SUFFIX
export DTYPE
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
export DATASET_NAME
export WORKER_COUNT
export EVAL_MODEL_CONFIG
export GEN_MODEL_CONFIG
export INCLUDE_ERROR_OUTPUTS_IN_SCORE
export STRUCTURED_EVAL_PY_ROOT

if [[ "$LOGGING_MODE" -eq 1 ]]; then
    setup_logging_mode
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi

echo "Starting full Loong dataset run with Qwen2-72B-Instruct (float16)."
echo "model_dir=$MODEL_DIR"
echo "dtype=$DTYPE"
echo "dataset_name=$DATASET_NAME"
echo "worker_count=$WORKER_COUNT"
echo "api_model_name=$API_MODEL_NAME"
echo "output_path_suffix=$OUTPUT_PATH_SUFFIX"
echo ""

cd "$ROOT_DIR"
bash "$ROOT_DIR/run_inference.sh" all_workers --no_shuffle "${FORWARDED_ARGS[@]}"
