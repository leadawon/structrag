#!/usr/bin/env bash
# Full Loong dataset inference (1600 samples) with meta-llama/Llama-3.1-8B-Instruct.
# Results land in eval_results/llama3p1-8b/ and are symlinked into result_full/.
# Judge is disabled (AUTO_SCORE=0).

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
VENV_PATH="${VENV_PATH:-/workspace/venvs/structragvenv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
DOWNLOAD_MODEL_SCRIPT="${DOWNLOAD_MODEL_SCRIPT:-$SCRIPT_DIR/download_model.sh}"
AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-0}"

# ---------------------------------------------------------------------------
# model_ready: check that model directory has all required files
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Model location
# ---------------------------------------------------------------------------
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
        if model_ready "$candidate"; then
            MODEL_DIR="$candidate"
            break
        fi
    done
else
    MODEL_CANDIDATES=("$MODEL_DIR")
fi

# ---------------------------------------------------------------------------
# Server & inference settings
# ---------------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
API_MODEL_NAME="${API_MODEL_NAME:-Llama-3.1-8B-Instruct}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-llama31-8b}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$SCRIPT_DIR/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/llama31_8b_vllm.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/llama31_8b_vllm.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

# ---------------------------------------------------------------------------
# Full dataset settings (1600 samples, no subset preparation)
# ---------------------------------------------------------------------------
# loong_process.jsonl lives in /workspace/Loong_long/ — we need a loong_dir
# wrapper so main.py's resolve_loong_dir() can find data/loong_process.jsonl.
FULL_DATA_PATH="${FULL_DATA_PATH:-/workspace/Loong_long/loong_process.jsonl}"
LOONG_LINK_DIR="${LOONG_LINK_DIR:-$ROOT_DIR/loong/Loong_full}"
RESULT_FULL_DIR="${RESULT_FULL_DIR:-$ROOT_DIR/result_full}"

# LLM_NAME / DATASET_NAME drive eval_results subdirectory naming.
# With 8 workers, each handles 200 samples → 8 × 200 = 1600 total.
LLM_NAME="${LLM_NAME:-llama3p1-8b}"
DATASET_NAME="${DATASET_NAME:-loong_full}"
WORKER_COUNT="${WORKER_COUNT:-8}"
LIMIT="${LIMIT:-}"
LIMIT_ARG=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG=(--limit "$LIMIT")
fi

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage:
  bash scripts_full/llama3p1_8b/run_inference_exper_full.sh
  bash scripts_full/llama3p1_8b/run_inference_exper_full.sh --dry-run
  bash scripts_full/llama3p1_8b/run_inference_exper_full.sh --smoke

Behavior:
  - Runs StructRAG over the full Loong dataset (1600 samples)
  - 8 workers × 200 samples each, no shuffle, no judge
  - Results written to eval_results/llama3p1-8b/loong_full_* and
    symlinked into result_full/

Flags:
  --dry-run   Validate paths and environment; do not start server or inference
  --smoke     Quick test with 2 samples on worker 0 (server must be running)

Defaults:
  MODEL_DIR=$MODEL_DIR
  FULL_DATA_PATH=$FULL_DATA_PATH
  API_MODEL_NAME=$API_MODEL_NAME
  WORKER_COUNT=$WORKER_COUNT
    LIMIT=$LIMIT (optional; passed through to main.py)
  AUTO_SCORE=0 (judge disabled)
  RESULT_FULL_DIR=$RESULT_FULL_DIR

Examples:
  bash scripts_full/llama3p1_8b/run_inference_exper_full.sh --dry-run
  CUDA_VISIBLE_DEVICES=0 bash scripts_full/llama3p1_8b/run_inference_exper_full.sh
    WORKER_COUNT=1 LIMIT=1 bash scripts_full/llama3p1_8b/run_inference_exper_full.sh  # first 1 only
EOF
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
SMOKE_RUN=0

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --smoke)
            SMOKE_RUN=1
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# setup_loong_link: create data/ symlink so resolve_loong_dir() works
# ---------------------------------------------------------------------------
setup_loong_link() {
    mkdir -p "$LOONG_LINK_DIR/data"
    local link_target="$LOONG_LINK_DIR/data/loong_process.jsonl"
    if [[ ! -e "$link_target" ]]; then
        ln -sf "$FULL_DATA_PATH" "$link_target"
        echo "loong_link_created=$link_target -> $FULL_DATA_PATH"
    fi
}

# ---------------------------------------------------------------------------
# download_model_if_missing
# ---------------------------------------------------------------------------
download_model_if_missing() {
    if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Checked candidates:"
        for candidate in "${MODEL_CANDIDATES[@]}"; do
            echo "  - $candidate"
        done
        echo "Options:"
        echo "  1. Download with: bash scripts_full/llama3p1_8b/download_model.sh"
        echo "  2. Provide path: MODEL_DIR=/path/to/model bash scripts_full/llama3p1_8b/run_inference_exper_full.sh"
        exit 1
    fi

    if [[ ! -f "$DOWNLOAD_MODEL_SCRIPT" ]]; then
        echo "Auto-download script not found: $DOWNLOAD_MODEL_SCRIPT"
        exit 1
    fi

    echo "Model missing: $MODEL_DIR — AUTO_DOWNLOAD_MODEL=1, downloading..."
    MODEL_DIR="$MODEL_DIR" PYTHON_BIN="$PYTHON_BIN" bash "$DOWNLOAD_MODEL_SCRIPT"

    if ! model_ready "$MODEL_DIR"; then
        echo "Download finished but model still incomplete: $MODEL_DIR"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Dry-run: validate environment and exit
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN — validation only, no server or inference will start ==="
    echo ""
    echo "python_bin=$PYTHON_BIN"
    if [[ -x "$PYTHON_BIN" ]]; then
        echo "python_ok=yes ($("$PYTHON_BIN" --version 2>&1))"
    else
        echo "python_ok=NO — $PYTHON_BIN not found or not executable"
    fi

    echo ""
    echo "full_data_path=$FULL_DATA_PATH"
    if [[ -f "$FULL_DATA_PATH" ]]; then
        LINES=$(wc -l < "$FULL_DATA_PATH")
        echo "full_data_ok=yes (${LINES} lines)"
    else
        echo "full_data_ok=NO — file not found"
    fi

    echo ""
    echo "model_dir=$MODEL_DIR"
    MODEL_STATUS="missing"
    for candidate in "${MODEL_CANDIDATES[@]}"; do
        if model_ready "$candidate"; then
            MODEL_STATUS="ready ($candidate)"
            break
        fi
    done
    echo "model_status=$MODEL_STATUS"

    echo ""
    echo "loong_link_dir=$LOONG_LINK_DIR"
    if [[ -e "$LOONG_LINK_DIR/data/loong_process.jsonl" ]]; then
        echo "loong_link=already exists"
    else
        echo "loong_link=will be created at runtime"
    fi

    echo ""
    echo "result_full_dir=$RESULT_FULL_DIR"
    echo "lllm_name=$LLM_NAME"
    echo "dataset_name=$DATASET_NAME"
    echo "worker_count=$WORKER_COUNT"
    echo "limit=${LIMIT:-none}"
    echo "api_model_name=$API_MODEL_NAME"
    echo "auto_score=0 (judge disabled)"

    echo ""
    echo "=== Commands that would run ==="
    echo "  # Setup loong link"
    echo "  mkdir -p $LOONG_LINK_DIR/data"
    echo "  ln -sf $FULL_DATA_PATH $LOONG_LINK_DIR/data/loong_process.jsonl"
    echo ""
    echo "  # Start vLLM server (detached)"
    echo "  bash $SERVER_SCRIPT_PATH --detach"
    echo ""
    echo "  # Run inference (all 1600 samples across 8 workers)"
    echo "  LOONG_DIR=$LOONG_LINK_DIR EVAL_DATA_PATH=$FULL_DATA_PATH \\"
    echo "  LLM_NAME=$LLM_NAME DATASET_NAME=$DATASET_NAME \\"
    echo "  WORKER_COUNT=$WORKER_COUNT AUTO_SCORE=0 \\"
    echo "  bash $ROOT_DIR/run_inference.sh all_workers --no_shuffle ${LIMIT_ARG[*]}"
    echo ""
    echo "  # Results symlinked to: $RESULT_FULL_DIR/latest"
    exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight checks (non-dry-run)
# ---------------------------------------------------------------------------
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found or not executable: $PYTHON_BIN"
    echo "Set VENV_PATH or PYTHON_BIN to a working Python environment."
    exit 1
fi

if [[ ! -f "$FULL_DATA_PATH" ]]; then
    echo "Full dataset not found: $FULL_DATA_PATH"
    echo "Expected: /workspace/Loong_long/loong_process.jsonl (1600 records)"
    exit 1
fi

if ! model_ready "$MODEL_DIR"; then
    download_model_if_missing
fi

# ---------------------------------------------------------------------------
# Set up LOONG_DIR symlink so resolve_loong_dir() in main.py works
# ---------------------------------------------------------------------------
setup_loong_link

# ---------------------------------------------------------------------------
# Result directory
# ---------------------------------------------------------------------------
mkdir -p "$RESULT_FULL_DIR"

# Fix a stable timestamp so we can locate the result dir after the run.
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_TIMESTAMP

echo "=== Llama 3.1 8B — Full Loong inference ==="
echo "model_dir=$MODEL_DIR"
echo "full_data_path=$FULL_DATA_PATH"
echo "worker_count=$WORKER_COUNT"
echo "auto_score=0"
echo "run_timestamp=$RUN_TIMESTAMP"
echo "result_full_dir=$RESULT_FULL_DIR"
echo ""

# ---------------------------------------------------------------------------
# Export vars consumed by run_inference.sh
# ---------------------------------------------------------------------------
export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
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
export DATASET_NAME
export LLM_NAME="$LLM_NAME"
export WORKER_COUNT
export AUTO_SCORE="0"
export AUTO_SCORE_FORCE_OVERWRITE="0"
export INCLUDE_ERROR_OUTPUTS_IN_SCORE="1"
export GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
export EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
# Disable scoring-related checks since AUTO_SCORE=0
export STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-}"

cd "$ROOT_DIR"

# Prepend structragvenv so that plain `python` calls inside run_inference.sh
# resolve to our venv (which has transformers/requests/tqdm for the client).
export PATH="$VENV_PATH/bin:$PATH"

# ---------------------------------------------------------------------------
# Smoke run: 2 samples on worker 0 (MANAGE_SERVER=1 auto-starts the server)
# ---------------------------------------------------------------------------
if [[ "$SMOKE_RUN" -eq 1 ]]; then
    echo "=== SMOKE RUN: 2 samples, worker 0, MANAGE_SERVER=1 ==="
    MANAGE_SERVER=1 WORKER_COUNT=1 \
        bash "$ROOT_DIR/run_inference.sh" worker 0 --limit 2 --no_shuffle
    echo "smoke_run=ok"
    exit 0
fi

# ---------------------------------------------------------------------------
# Full run: all 8 workers (8 × 200 = 1600 samples)
# ---------------------------------------------------------------------------
bash "$ROOT_DIR/run_inference.sh" all_workers --no_shuffle "${LIMIT_ARG[@]}"

# ---------------------------------------------------------------------------
# Post-run: link results into result_full/
# ---------------------------------------------------------------------------
EVAL_DIR=$(find "$ROOT_DIR/eval_results/$LLM_NAME" -mindepth 1 -maxdepth 1 -type d \
    -name "${DATASET_NAME}*ts-${RUN_TIMESTAMP}*" 2>/dev/null | head -1 || true)

if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then
    RUN_LABEL="$(basename "$EVAL_DIR")"
    ln -sfn "$EVAL_DIR" "$RESULT_FULL_DIR/latest"
    echo ""
    echo "=== Results ==="
    echo "eval_results_dir=$EVAL_DIR"
    echo "result_full_latest=$RESULT_FULL_DIR/latest"
    echo "final_output=$EVAL_DIR/final_output_0.jsonl"
    echo "final_errors=$EVAL_DIR/final_output_error_0.jsonl"
    echo "run_manifest=$EVAL_DIR/run_manifest.json"
else
    echo ""
    echo "WARNING: Could not locate result directory for ts=$RUN_TIMESTAMP"
    echo "Check eval_results/$LLM_NAME/ manually."
fi
