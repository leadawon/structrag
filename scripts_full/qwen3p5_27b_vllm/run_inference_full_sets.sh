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
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag_vllm}"
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
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
URL="${URL:-$HOST:$PORT}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen3.5-27B}"
LLM_NAME="${LLM_NAME:-qwen35-27b-vllm}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$SCRIPT_DIR/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm_gpu0123_sets.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm_gpu0123_sets.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

FULL_DATA_PATH="${FULL_DATA_PATH:-$ROOT_DIR/loong/Loong/data/loong_process.jsonl}"
LOONG_LINK_DIR="${LOONG_LINK_DIR:-$ROOT_DIR/loong/Loong_full}"
RESULT_SET_DIR="${RESULT_SET_DIR:-$ROOT_DIR/result_full/qwen35_27b_vllm_sets}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$SCRIPT_DIR/prepare_ordered_set_runs.py}"
SET_LIST="${SET_LIST:-1 2 3 4}"
DATASET_NAME_TEMPLATE="${DATASET_NAME_TEMPLATE:-loong_set{set_id}}"
OUTPUT_SUFFIX_TEMPLATE="${OUTPUT_SUFFIX_TEMPLATE:-_ordered_set{set_id}}"
WORKER_CHUNK_SIZE="${WORKER_CHUNK_SIZE:-200}"
AUTO_SCORE="${AUTO_SCORE:-1}"
AUTO_SCORE_FORCE_OVERWRITE="${AUTO_SCORE_FORCE_OVERWRITE:-1}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-0}"
INCLUDE_ERROR_RESULTS_FOR_REUSE="${INCLUDE_ERROR_RESULTS_FOR_REUSE:-0}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-/workspace/LAMBO}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$ROOT_DIR/loong/Loong/config/models}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
STOP_SERVER_ON_EXIT="${STOP_SERVER_ON_EXIT:-1}"
SOURCE_PRIORITY="${SOURCE_PRIORITY:-pulled,local}"
SERVER_STARTED_BY_THIS_SCRIPT=0

usage() {
    cat <<EOF
Usage:
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh --prepare-only
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh --dry-run
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh --sets "1 2"

Behavior:
  - Builds set1, set2, set3, set4 folders under result_full/qwen35_27b_vllm_sets.
  - Keeps pulled-server and local-server reused outputs separated under each set folder.
  - Seeds eval_results with successful reused rows, creates pending JSONL, then runs pending rows only.
  - Runs sets in numeric order.

Environment overrides:
  PULLED_RESULT_DIRS=/path/to/resultA:/path/to/resultB
  LOCAL_RESULT_DIRS=/path/to/resultC
  SOURCE_PRIORITY=pulled,local
  RESULT_SET_DIR=$RESULT_SET_DIR
  SET_LIST="$SET_LIST"
  AUTO_SCORE=$AUTO_SCORE
  STOP_SERVER_ON_EXIT=$STOP_SERVER_ON_EXIT
EOF
}

PREPARE_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --sets)
            shift
            SET_LIST="${1:-}"
            if [[ -z "$SET_LIST" ]]; then
                echo "--sets requires a value."
                exit 1
            fi
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
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

download_model_if_missing() {
    if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Checked candidates:"
        for candidate in "${MODEL_CANDIDATES[@]}"; do
            echo "  - $candidate"
        done
        exit 1
    fi

    if [[ ! -f "$DOWNLOAD_MODEL_SCRIPT" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Auto-download script not found: $DOWNLOAD_MODEL_SCRIPT"
        exit 1
    fi

    echo "Model path is missing or incomplete: $MODEL_DIR"
    echo "AUTO_DOWNLOAD_MODEL=1, downloading Qwen3.5 27B first."
    MODEL_DIR="$MODEL_DIR" bash "$DOWNLOAD_MODEL_SCRIPT"

    if ! model_ready "$MODEL_DIR"; then
        echo "Download finished, but model files are still incomplete: $MODEL_DIR"
        exit 1
    fi
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

check_endpoint_health() {
    local endpoint="$1"
    "$PYTHON_BIN" - "$endpoint" <<'PY'
import sys
import requests

endpoint = sys.argv[1]
try:
    response = requests.get(f"http://{endpoint}/health", timeout=5)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if response.status_code == 200 else 1)
PY
}

wait_for_endpoint_health() {
    local endpoint="$1"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if check_endpoint_health "$endpoint"; then
            echo "main server is healthy: http://$endpoint/health"
            return 0
        fi
        local now_ts
        now_ts="$(date +%s)"
        if (( now_ts - start_ts >= RESTART_WAIT_TIMEOUT )); then
            echo "Timed out waiting for main server health after ${RESTART_WAIT_TIMEOUT}s: http://$endpoint/health"
            if [[ -f "$LOG_PATH" ]]; then
                tail -n 120 "$LOG_PATH" | tr -d '\000' || true
            fi
            return 1
        fi
        echo "Waiting for main server health... elapsed=$((now_ts - start_ts))s timeout=${RESTART_WAIT_TIMEOUT}s"
        sleep "$RESTART_WAIT_INTERVAL"
    done
}

start_server_if_needed() {
    if check_endpoint_health "$URL"; then
        echo "main server already healthy: http://$URL/health"
        return 0
    fi

    echo "Starting vLLM server for ordered set inference..."
    PYTHON_BIN="$PYTHON_BIN" \
    VENV_PATH="$VENV_PATH" \
    MODEL_DIR="$MODEL_DIR" \
    LOG_PATH="$LOG_PATH" \
    PID_FILE="$PID_FILE" \
    PGID_FILE="$PGID_FILE" \
    CLEAN_STALE_VLLM="$CLEAN_STALE_VLLM" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    CUDA_DEVICES="$CUDA_DEVICES" \
    TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
    bash "$SERVER_SCRIPT_PATH" --detach
    SERVER_STARTED_BY_THIS_SCRIPT=1
    wait_for_endpoint_health "$URL"
}

stop_server_safely() {
    if [[ "$STOP_SERVER_ON_EXIT" != "1" ]]; then
        return 0
    fi
    if [[ "$SERVER_STARTED_BY_THIS_SCRIPT" != "1" ]]; then
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
    TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
    bash "$SERVER_SCRIPT_PATH" --stop || true
}

cleanup_on_exit() {
    local exit_code=$?
    set +e
    stop_server_safely
    return "$exit_code"
}

read_manifest_field() {
    local manifest_path="$1"
    local field="$2"
    "$PYTHON_BIN" - "$manifest_path" "$field" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2], "")
print(value)
PY
}

score_completed_set() {
    local dataset_name="$1"
    local output_suffix="$2"
    local worker_count="$3"
    if [[ "$AUTO_SCORE" != "1" ]]; then
        return 0
    fi

    FORCE_OVERWRITE="$AUTO_SCORE_FORCE_OVERWRITE" \
    INPUT_LLM_NAME="$LLM_NAME" \
    DATASET_NAME="$dataset_name" \
    OUTPUT_PATH_SUFFIX="$output_suffix" \
    WORKER_COUNT="$worker_count" \
    PROCESS_NUM_EVAL="${PROCESS_NUM_EVAL:-20}" \
    EVAL_MODEL_CONFIG="$EVAL_MODEL_CONFIG" \
    GEN_MODEL_CONFIG="$GEN_MODEL_CONFIG" \
    MODEL_CONFIG_DIR="$MODEL_CONFIG_DIR" \
    URL="$URL" \
    API_MODEL_NAME="$API_MODEL_NAME" \
    INCLUDE_ERROR_OUTPUTS_IN_SCORE="$INCLUDE_ERROR_OUTPUTS_IN_SCORE" \
    STRUCTURED_EVAL_PY_ROOT="$STRUCTURED_EVAL_PY_ROOT" \
    STRUCTRAG_ENABLE_THINKING="$STRUCTRAG_ENABLE_THINKING" \
    LOONG_DIR="$LOONG_LINK_DIR" \
    RUN_TIMESTAMP="$RUN_TIMESTAMP" \
    bash "$ROOT_DIR/run_score.sh"
}

run_prepare() {
    local include_error_args=()
    if [[ "$INCLUDE_ERROR_RESULTS_FOR_REUSE" == "1" ]]; then
        include_error_args=(--include-error-results)
    fi

    "$PYTHON_BIN" "$PREPARE_SCRIPT" \
        --full-data-path "$FULL_DATA_PATH" \
        --eval-results-root "$ROOT_DIR/eval_results" \
        --output-root "$RESULT_SET_DIR" \
        --llm-name "$LLM_NAME" \
        --sets "$SET_LIST" \
        --dataset-name-template "$DATASET_NAME_TEMPLATE" \
        --output-suffix-template "$OUTPUT_SUFFIX_TEMPLATE" \
        --worker-chunk-size "$WORKER_CHUNK_SIZE" \
        --prefer-source "$SOURCE_PRIORITY" \
        "${include_error_args[@]}"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$PREPARE_SCRIPT" ]]; then
    echo "Prepare script not found: $PREPARE_SCRIPT"
    exit 1
fi

if [[ ! -f "$FULL_DATA_PATH" ]]; then
    echo "Full Loong dataset not found: $FULL_DATA_PATH"
    exit 1
fi

if ! model_ready "$MODEL_DIR"; then
    download_model_if_missing
fi

setup_loong_full_dir
mkdir -p "$RESULT_SET_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_TIMESTAMP
export PYTHON_BIN
export VENV_PATH
export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
export LLM_NAME
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
export LOONG_DIR="$LOONG_LINK_DIR"
export AUTO_SCORE
export AUTO_SCORE_FORCE_OVERWRITE
export EVAL_MODEL_CONFIG
export GEN_MODEL_CONFIG
export INCLUDE_ERROR_OUTPUTS_IN_SCORE
export STRUCTURED_EVAL_PY_ROOT
export MODEL_CONFIG_DIR

cd "$ROOT_DIR"

echo "=== Preparing ordered Loong set inference ==="
echo "full_data_path=$FULL_DATA_PATH"
echo "result_set_dir=$RESULT_SET_DIR"
echo "set_list=$SET_LIST"
echo "source_priority=$SOURCE_PRIORITY"
run_prepare

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    echo "prepare_only=1, stopping before inference."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "=== DRY RUN: planned ordered set commands ==="
fi

trap cleanup_on_exit EXIT

if [[ "$DRY_RUN" -eq 0 ]]; then
    start_server_if_needed
fi

for set_id in $SET_LIST; do
    manifest_path="$RESULT_SET_DIR/set${set_id}/manifest.json"
    if [[ ! -f "$manifest_path" ]]; then
        echo "Missing manifest: $manifest_path"
        exit 1
    fi

    dataset_name="$(read_manifest_field "$manifest_path" dataset_name)"
    output_suffix="$(read_manifest_field "$manifest_path" output_path_suffix)"
    pending_data_path="$(read_manifest_field "$manifest_path" pending_data_path)"
    pending_count="$(read_manifest_field "$manifest_path" pending_count)"
    worker_count="$(read_manifest_field "$manifest_path" worker_count)"
    score_worker_count="$(read_manifest_field "$manifest_path" score_worker_count)"
    target_eval_dir="$(read_manifest_field "$manifest_path" target_eval_dir)"

    echo ""
    echo "=== set${set_id} ==="
    echo "dataset_name=$dataset_name"
    echo "output_suffix=$output_suffix"
    echo "pending_count=$pending_count"
    echo "worker_count=$worker_count"
    echo "target_eval_dir=$target_eval_dir"

    if [[ "$pending_count" -eq 0 ]]; then
        echo "set${set_id} has no pending samples."
        if [[ "$DRY_RUN" -eq 0 ]]; then
            score_completed_set "$dataset_name" "$output_suffix" "$score_worker_count"
        else
            echo "score command: DATASET_NAME=$dataset_name OUTPUT_PATH_SUFFIX=$output_suffix WORKER_COUNT=$score_worker_count bash run_score.sh"
        fi
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "inference command: DATASET_NAME=$dataset_name EVAL_DATA_PATH=$pending_data_path WORKER_COUNT=$worker_count OUTPUT_PATH_SUFFIX=$output_suffix RESUME_OUTPUT_PATH_SUFFIX=$output_suffix bash run_inference.sh all_workers --no_shuffle"
        continue
    fi

    DATASET_NAME="$dataset_name" \
    EVAL_DATA_PATH="$pending_data_path" \
    WORKER_COUNT="$worker_count" \
    OUTPUT_PATH_SUFFIX="$output_suffix" \
    RESUME_OUTPUT_PATH_SUFFIX="$output_suffix" \
    AUTO_RESUME=1 \
    FORCE_NEW_RUN=0 \
    MANAGE_SERVER=0 \
    bash "$ROOT_DIR/run_inference.sh" all_workers --no_shuffle

    run_prepare
done

echo ""
echo "=== Ordered set inference done ==="
echo "summary=$RESULT_SET_DIR/summary.json"
