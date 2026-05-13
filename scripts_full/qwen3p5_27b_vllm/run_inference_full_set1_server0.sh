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
DATASET_NAME="${DATASET_NAME:-loong_set1_server0}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-_tail100}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$SCRIPT_DIR/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm_server0_set1_tail100.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm_server0_set1_tail100.pid}"
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
SOURCE_PENDING_PATH="${SOURCE_PENDING_PATH:-$ROOT_DIR/result_full/qwen35_27b_vllm_sets/set1/data/loong_process.jsonl}"
MAIN_SET1_EVAL_DIR="${MAIN_SET1_EVAL_DIR:-$ROOT_DIR/eval_results/$LLM_NAME/loong_set1_ordered_set1}"
SERVER0_RESULT_DIR="${SERVER0_RESULT_DIR:-$ROOT_DIR/result_full/qwen35_27b_vllm_sets/set1_server0}"
SERVER0_DATA_PATH="${SERVER0_DATA_PATH:-$SERVER0_RESULT_DIR/data/loong_process.jsonl}"
SERVER0_EVAL_DIR="$ROOT_DIR/eval_results/$LLM_NAME/${DATASET_NAME}${OUTPUT_PATH_SUFFIX}"
TAIL_COUNT="${TAIL_COUNT:-100}"
AUTO_SCORE="${AUTO_SCORE:-0}"
AUTO_SCORE_FORCE_OVERWRITE="${AUTO_SCORE_FORCE_OVERWRITE:-1}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-0}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-/workspace/LAMBO}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$ROOT_DIR/loong/Loong/config/models}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
STOP_SERVER_ON_EXIT="${STOP_SERVER_ON_EXIT:-1}"
ALLOW_FULL_DATA_FALLBACK="${ALLOW_FULL_DATA_FALLBACK:-0}"
SERVER_STARTED_BY_THIS_SCRIPT=0

usage() {
    cat <<EOF
Usage:
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh --prepare-only
  bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh --dry-run

Behavior:
  - Runs only the last TAIL_COUNT samples from set1 that are not already completed.
  - Writes the run to eval_results/$LLM_NAME/${DATASET_NAME}${OUTPUT_PATH_SUFFIX}.
  - Writes the prepared input and manifest to result_full/qwen35_27b_vllm_sets/set1_server0.
  - AUTO_SCORE defaults to 0 because this is a partial shard to merge back later.

Defaults:
  TAIL_COUNT=$TAIL_COUNT
  DATASET_NAME=$DATASET_NAME
  OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX
  SERVER0_EVAL_DIR=$SERVER0_EVAL_DIR
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
        --tail-count)
            shift
            TAIL_COUNT="${1:-}"
            if [[ -z "$TAIL_COUNT" ]]; then
                echo "--tail-count requires a value."
                exit 1
            fi
            shift
            ;;
        --score)
            AUTO_SCORE=1
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
        echo "Auto-download script not found: $DOWNLOAD_MODEL_SCRIPT"
        exit 1
    fi
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
    echo "Starting vLLM server for set1 server0 tail100..."
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
    if [[ "$STOP_SERVER_ON_EXIT" != "1" || "$SERVER_STARTED_BY_THIS_SCRIPT" != "1" ]]; then
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

prepare_server0_tail() {
    "$PYTHON_BIN" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

full_data_path = Path(r"$FULL_DATA_PATH")
source_pending_path = Path(r"$SOURCE_PENDING_PATH")
main_eval_dir = Path(r"$MAIN_SET1_EVAL_DIR")
server0_eval_dir = Path(r"$SERVER0_EVAL_DIR")
server0_data_path = Path(r"$SERVER0_DATA_PATH")
server0_result_dir = Path(r"$SERVER0_RESULT_DIR")
tail_count = int(r"$TAIL_COUNT")
allow_full_data_fallback = r"$ALLOW_FULL_DATA_FALLBACK" == "1"

def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def is_error(row):
    return (
        row.get("error_message") is not None
        or row.get("error_kind") is not None
        or row.get("used_time") == -100
        or row.get("generate_response") == "meet error"
    )

def collect_completed(eval_dir):
    ids = set()
    if not eval_dir.exists():
        return ids
    for path in sorted(eval_dir.glob("final_output_*.jsonl")):
        if path.name.startswith("final_output_error_"):
            continue
        for row in load_jsonl(path):
            data_id = row.get("id")
            if data_id is not None and not is_error(row):
                ids.add(str(data_id))
    return ids

completed = collect_completed(main_eval_dir) | collect_completed(server0_eval_dir)
using_existing_prepared_input = False
if source_pending_path.exists():
    base_path = source_pending_path
    base_rows = [
        row
        for row in load_jsonl(base_path)
        if int(row.get("set", -1)) == 1 and str(row.get("id")) not in completed
    ]
    selected = base_rows[-tail_count:] if tail_count > 0 else []
elif server0_data_path.exists():
    base_path = server0_data_path
    base_rows = [row for row in load_jsonl(base_path) if int(row.get("set", -1)) == 1]
    selected = base_rows
    using_existing_prepared_input = True
elif allow_full_data_fallback:
    base_path = full_data_path
    base_rows = [
        row
        for row in load_jsonl(base_path)
        if int(row.get("set", -1)) == 1 and str(row.get("id")) not in completed
    ]
    selected = base_rows[-tail_count:] if tail_count > 0 else []
else:
    raise SystemExit(
        "Cannot find the prepared set1 pending file. Copy "
        f"{server0_data_path.parent.parent} from the main server, or set "
        "SOURCE_PENDING_PATH to the main set1 pending file. "
        "Refusing to fall back to full_data because it may select the wrong 100 IDs."
    )

server0_data_path.parent.mkdir(parents=True, exist_ok=True)
with server0_data_path.open("w", encoding="utf-8") as fw:
    for row in selected:
        fw.write(json.dumps(row, ensure_ascii=False) + "\\n")

manifest = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "purpose": "set1 server0 tail shard",
    "tail_count": tail_count,
    "base_path": str(base_path),
    "using_existing_prepared_input": using_existing_prepared_input,
    "full_data_path": str(full_data_path),
    "source_pending_path": str(source_pending_path),
    "main_eval_dir": str(main_eval_dir),
    "server0_eval_dir": str(server0_eval_dir),
    "server0_data_path": str(server0_data_path),
    "completed_ids_excluded": len(completed),
    "remaining_before_tail": len(base_rows),
    "selected_count": len(selected),
    "first_selected_id": selected[0].get("id") if selected else None,
    "last_selected_id": selected[-1].get("id") if selected else None,
    "selected_ids": [row.get("id") for row in selected],
}
server0_result_dir.mkdir(parents=True, exist_ok=True)
(server0_result_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",
    encoding="utf-8",
)
print(f"server0_data_path={server0_data_path}")
print(f"selected_count={len(selected)}")
print(f"remaining_before_tail={len(base_rows)}")
print(f"first_selected_id={manifest['first_selected_id']}")
print(f"last_selected_id={manifest['last_selected_id']}")
PY
    ln -sfn "$SERVER0_EVAL_DIR" "$SERVER0_RESULT_DIR/eval_results"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
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
mkdir -p "$SERVER0_RESULT_DIR" "$SERVER0_EVAL_DIR"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_TIMESTAMP
export PYTHON_BIN
export VENV_PATH
export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export API_MODEL_NAME
export LLM_NAME
export DATASET_NAME
export OUTPUT_PATH_SUFFIX
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

echo "=== Preparing set1 server0 tail shard ==="
echo "server0_result_dir=$SERVER0_RESULT_DIR"
echo "server0_eval_dir=$SERVER0_EVAL_DIR"
echo "tail_count=$TAIL_COUNT"
prepare_server0_tail

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    echo "prepare_only=1, stopping before inference."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "=== DRY RUN ==="
    echo "DATASET_NAME=$DATASET_NAME EVAL_DATA_PATH=$SERVER0_DATA_PATH WORKER_COUNT=1 OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX RESUME_OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX AUTO_SCORE=$AUTO_SCORE bash run_inference.sh all_workers --no_shuffle"
    exit 0
fi

trap cleanup_on_exit EXIT
start_server_if_needed

DATASET_NAME="$DATASET_NAME" \
EVAL_DATA_PATH="$SERVER0_DATA_PATH" \
WORKER_COUNT=1 \
OUTPUT_PATH_SUFFIX="$OUTPUT_PATH_SUFFIX" \
RESUME_OUTPUT_PATH_SUFFIX="$OUTPUT_PATH_SUFFIX" \
AUTO_RESUME=1 \
FORCE_NEW_RUN=0 \
AUTO_SCORE="$AUTO_SCORE" \
MANAGE_SERVER=0 \
bash "$ROOT_DIR/run_inference.sh" all_workers --no_shuffle

echo ""
echo "=== set1 server0 tail shard done ==="
echo "server0_eval_dir=$SERVER0_EVAL_DIR"
echo "server0_result_dir=$SERVER0_RESULT_DIR"
echo "copy_back_hint=copy eval_results/$LLM_NAME/${DATASET_NAME}${OUTPUT_PATH_SUFFIX} back to the same path on the main server"
