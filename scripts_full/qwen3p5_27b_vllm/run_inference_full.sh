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
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
DOWNLOAD_MODEL_SCRIPT="${DOWNLOAD_MODEL_SCRIPT:-$SCRIPT_DIR/download_model.sh}"
AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-1}"

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
        "$ROOT_DIR/model/Qwen3.5-27B"
        "$ROOT_DIR/model/Qwen3.5-27B-Instruct"
        "/workspace/lambo/models/Qwen3.5-27B"
        "/workspace/LAMBO/models/Qwen3.5-27B"
        "/workspace/qwen/Qwen3.5-27B"
        "/workspace/qwen/Qwen3.5-27B-Instruct"
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
API_MODEL_NAME="${API_MODEL_NAME:-Qwen3.5-27B}"
LLM_NAME="${LLM_NAME:-qwen35-27b-vllm}"
DATASET_NAME="${DATASET_NAME:-loong_full}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-qwen35-think-off-vllm}"
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
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_OUTPUT_PATH_SUFFIX="${RESUME_OUTPUT_PATH_SUFFIX:-}"
FORCE_NEW_RUN="${FORCE_NEW_RUN:-0}"

FULL_DATA_PATH="${FULL_DATA_PATH:-$ROOT_DIR/loong/Loong/data/loong_process.jsonl}"
LOONG_LINK_DIR="${LOONG_LINK_DIR:-$ROOT_DIR/loong/Loong_full}"
RESULT_FULL_DIR="${RESULT_FULL_DIR:-$ROOT_DIR/result_full/qwen35_27b_vllm}"
WORKER_COUNT="${WORKER_COUNT:-8}"
AUTO_SCORE="${AUTO_SCORE:-1}"
AUTO_SCORE_FORCE_OVERWRITE="${AUTO_SCORE_FORCE_OVERWRITE:-1}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-1}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-/workspace/LAMBO}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$ROOT_DIR/loong/Loong/config/models}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
STOP_SERVER_ON_EXIT="${STOP_SERVER_ON_EXIT:-1}"

usage() {
    cat <<EOF
Usage:
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh --logging
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh --dry-run
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh --smoke

Behavior:
  - Uses the scripts/27b vLLM flow, but runs the full Loong dataset.
  - Defaults to CUDA_VISIBLE_DEVICES=4,5,6,7.
  - Runs all workers with no shuffle.
  - Stops the vLLM server on exit by default to release GPU memory.
  - Keeps automatic scoring enabled by default, matching scripts/27b.

Flags:
  --logging                 Enable StructRAG logging.
  --dry-run                 Validate paths and print the command without running.
  --smoke                   Run 2 samples on worker 0.
  --no-score                Set AUTO_SCORE=0.
  --no-cleanup              Do not stop the vLLM server on exit.
  --resume-suffix <suffix>  Resume a specific output suffix.
  --fresh                   Disable auto-resume and force a new run.

Defaults:
  MODEL_DIR=$MODEL_DIR
  FULL_DATA_PATH=$FULL_DATA_PATH
  LLM_NAME=$LLM_NAME
  DATASET_NAME=$DATASET_NAME
  WORKER_COUNT=$WORKER_COUNT
  AUTO_SCORE=$AUTO_SCORE
  STOP_SERVER_ON_EXIT=$STOP_SERVER_ON_EXIT
  SERVER_SCRIPT_PATH=$SERVER_SCRIPT_PATH
  RESULT_FULL_DIR=$RESULT_FULL_DIR
EOF
}

LOGGING_MODE=0
DRY_RUN=0
SMOKE_RUN=0
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --logging)
            LOGGING_MODE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --smoke)
            SMOKE_RUN=1
            shift
            ;;
        --no-score)
            AUTO_SCORE=0
            shift
            ;;
        --no-cleanup)
            STOP_SERVER_ON_EXIT=0
            shift
            ;;
        --resume-suffix)
            shift
            RESUME_OUTPUT_PATH_SUFFIX="${1:-}"
            if [[ -z "$RESUME_OUTPUT_PATH_SUFFIX" ]]; then
                echo "--resume-suffix requires a value."
                exit 1
            fi
            AUTO_RESUME=1
            shift
            ;;
        --fresh)
            FORCE_NEW_RUN=1
            AUTO_RESUME=0
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
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

    cat > "$STRUCTRAG_LOGGING_DIR/inference/launch_full.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
API_MODEL_NAME=$API_MODEL_NAME
LLM_NAME=$LLM_NAME
DATASET_NAME=$DATASET_NAME
STRUCTRAG_ENABLE_THINKING=$STRUCTRAG_ENABLE_THINKING
OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX
FULL_DATA_PATH=$FULL_DATA_PATH
ARGV=${FORWARDED_ARGS[*]:-all_workers}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

setup_loong_full_dir() {
    mkdir -p "$LOONG_LINK_DIR/data"

    local data_link="$LOONG_LINK_DIR/data/loong_process.jsonl"
    if [[ -L "$data_link" || ! -e "$data_link" ]]; then
        ln -sfn "$FULL_DATA_PATH" "$data_link"
    else
        echo "Keeping existing non-symlink data file: $data_link"
    fi

    local config_link="$LOONG_LINK_DIR/config"
    local source_config="$ROOT_DIR/loong/Loong/config"
    if [[ -d "$source_config" && ( -L "$config_link" || ! -e "$config_link" ) ]]; then
        ln -sfn "$source_config" "$config_link"
    fi
}

download_model_if_missing() {
    if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Checked candidates:"
        for candidate in "${MODEL_CANDIDATES[@]}"; do
            echo "  - $candidate"
        done
        echo "Example:"
        echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh"
        exit 1
    fi

    if [[ ! -f "$DOWNLOAD_MODEL_SCRIPT" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Auto-download script not found: $DOWNLOAD_MODEL_SCRIPT"
        exit 1
    fi

    echo "Model path is missing or incomplete: $MODEL_DIR"
    echo "AUTO_DOWNLOAD_MODEL=1, downloading Qwen3.5 27B first."
    echo "Download target: $MODEL_DIR"
    MODEL_DIR="$MODEL_DIR" bash "$DOWNLOAD_MODEL_SCRIPT"

    if ! model_ready "$MODEL_DIR"; then
        echo "Download finished, but model files are still incomplete: $MODEL_DIR"
        exit 1
    fi
}

resolve_result_dir() {
    local eval_root="$ROOT_DIR/eval_results/$LLM_NAME"
    local candidate=""

    if [[ -n "$RESUME_OUTPUT_PATH_SUFFIX" ]]; then
        candidate="$eval_root/${DATASET_NAME}${RESUME_OUTPUT_PATH_SUFFIX}"
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    candidate="$(find "$eval_root" -mindepth 1 -maxdepth 1 -type d \
        -name "${DATASET_NAME}*ts-${RUN_TIMESTAMP}*" 2>/dev/null | head -1 || true)"
    if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    find "$eval_root" -mindepth 1 -maxdepth 1 -type d \
        -name "${DATASET_NAME}_*" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

stop_server_safely() {
    if [[ "$STOP_SERVER_ON_EXIT" != "1" ]]; then
        return 0
    fi

    echo "Stopping vLLM server to release GPU memory..."
    PYTHON_BIN="$PYTHON_BIN" \
    VENV_PATH="$VENV_PATH" \
    MODEL_DIR="$MODEL_DIR" \
    LOG_PATH="$LOG_PATH" \
    PID_FILE="$PID_FILE" \
    PGID_FILE="$PGID_FILE" \
    CLEAN_STALE_VLLM="$CLEAN_STALE_VLLM" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    CUDA_DEVICES="$CUDA_DEVICES" \
    bash "$SERVER_SCRIPT_PATH" --stop || true
}

cleanup_on_exit() {
    local exit_code=$?
    set +e
    stop_server_safely
    return "$exit_code"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN: validation only ==="
    echo "python_bin=$PYTHON_BIN"
    echo "model_dir=$MODEL_DIR"
    echo "model_ready=$(model_ready "$MODEL_DIR" && echo yes || echo no)"
    echo "full_data_path=$FULL_DATA_PATH"
    if [[ -f "$FULL_DATA_PATH" ]]; then
        echo "full_data_lines=$(wc -l < "$FULL_DATA_PATH")"
    else
        echo "full_data_missing=1"
    fi
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "server_script_path=$SERVER_SCRIPT_PATH"
    echo "llm_name=$LLM_NAME"
    echo "dataset_name=$DATASET_NAME"
    echo "worker_count=$WORKER_COUNT"
    echo "auto_score=$AUTO_SCORE"
    echo "stop_server_on_exit=$STOP_SERVER_ON_EXIT"
    echo "command=bash $ROOT_DIR/run_inference.sh all_workers --no_shuffle ${FORWARDED_ARGS[*]:-}"
    exit 0
fi

if ! model_ready "$MODEL_DIR"; then
    download_model_if_missing
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$FULL_DATA_PATH" ]]; then
    echo "Full Loong dataset not found: $FULL_DATA_PATH"
    exit 1
fi

setup_loong_full_dir
mkdir -p "$RESULT_FULL_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_TIMESTAMP

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
export LLM_NAME
export DATASET_NAME
export STRUCTRAG_ENABLE_THINKING
export OUTPUT_PATH_SUFFIX
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
export LOONG_DIR="$LOONG_LINK_DIR"
export EVAL_DATA_PATH="$FULL_DATA_PATH"
export WORKER_COUNT
export AUTO_SCORE
export AUTO_SCORE_FORCE_OVERWRITE
export EVAL_MODEL_CONFIG
export GEN_MODEL_CONFIG
export INCLUDE_ERROR_OUTPUTS_IN_SCORE
export STRUCTURED_EVAL_PY_ROOT
export MODEL_CONFIG_DIR
export AUTO_RESUME
export RESUME_OUTPUT_PATH_SUFFIX
export FORCE_NEW_RUN

if [[ "$LOGGING_MODE" -eq 1 ]]; then
    setup_logging_mode
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi

cd "$ROOT_DIR"
trap cleanup_on_exit EXIT

echo "=== Qwen3.5 27B vLLM full Loong inference ==="
echo "model_dir=$MODEL_DIR"
echo "full_data_path=$FULL_DATA_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "worker_count=$WORKER_COUNT"
echo "auto_score=$AUTO_SCORE"
echo "stop_server_on_exit=$STOP_SERVER_ON_EXIT"
echo "run_timestamp=$RUN_TIMESTAMP"

run_status=0
if [[ "$SMOKE_RUN" -eq 1 ]]; then
    WORKER_COUNT=1 bash "$ROOT_DIR/run_inference.sh" worker 0 --limit 2 --no_shuffle "${FORWARDED_ARGS[@]}" || run_status=$?
else
    bash "$ROOT_DIR/run_inference.sh" all_workers --no_shuffle "${FORWARDED_ARGS[@]}" || run_status=$?
fi

if [[ "$run_status" -ne 0 ]]; then
    exit "$run_status"
fi

EVAL_DIR="$(resolve_result_dir || true)"
if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then
    ln -sfn "$EVAL_DIR" "$RESULT_FULL_DIR/latest"
    echo ""
    echo "=== Results ==="
    echo "eval_results_dir=$EVAL_DIR"
    echo "result_full_latest=$RESULT_FULL_DIR/latest"
    echo "run_manifest=$EVAL_DIR/run_manifest.json"
else
    echo ""
    echo "WARNING: could not locate result directory. Check eval_results/$LLM_NAME/."
fi
