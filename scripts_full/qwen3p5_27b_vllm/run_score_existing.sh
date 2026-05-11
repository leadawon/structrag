#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$SCRIPT_DIR/run_server.sh}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
WAIT_INTERVAL="${WAIT_INTERVAL:-15}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-1}"
START_SERVER="${START_SERVER:-0}"
STOP_SERVER_WHEN_DONE="${STOP_SERVER_WHEN_DONE:-$START_SERVER}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_DATASET_NAME="${DEFAULT_DATASET_NAME:-loong_full}"
DEFAULT_LLM_NAME="${DEFAULT_LLM_NAME:-qwen35-27b-vllm}"
DEFAULT_API_MODEL_NAME="${DEFAULT_API_MODEL_NAME:-Qwen3.5-27B}"
RERUN_TAG="${RERUN_TAG:-rerun-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN_MANIFEST="${RUN_MANIFEST:-}"
OUTPUT_PATH_SUFFIX_ARG=""
LATEST_MODE=0
SERVER_STARTED_BY_SCRIPT=0

usage() {
    cat <<EOF
Usage:
  bash scripts_full/qwen3p5_27b_vllm/run_score_existing.sh --latest
  bash scripts_full/qwen3p5_27b_vllm/run_score_existing.sh --output-path-suffix '_ts-...'
  bash scripts_full/qwen3p5_27b_vllm/run_score_existing.sh --manifest /abs/path/run_manifest.json
  START_SERVER=1 bash scripts_full/qwen3p5_27b_vllm/run_score_existing.sh --latest
  DRY_RUN=1 bash scripts_full/qwen3p5_27b_vllm/run_score_existing.sh --latest

Behavior:
  - Reuses completed inference outputs under eval_results/.../final_output*.jsonl
  - Restores scoring settings from the saved run_manifest.json
  - Re-runs structured eval + LAMBO v2 judge + Loong evaluator without rerunning inference
  - If START_SERVER=1, starts the 27B local judge server first and stops it afterwards

Defaults:
  DEFAULT_DATASET_NAME=loong_full
  DEFAULT_LLM_NAME=qwen35-27b-vllm
  DEFAULT_API_MODEL_NAME=Qwen3.5-27B
  FORCE_OVERWRITE=1
  START_SERVER=0
  STOP_SERVER_WHEN_DONE=<same as START_SERVER>
  DRY_RUN=0
  SERVER_SCRIPT_PATH=$SCRIPT_DIR/run_server.sh
EOF
}

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        --latest)
            LATEST_MODE=1
            ;;
        --manifest=*)
            RUN_MANIFEST="${arg#*=}"
            ;;
        --manifest)
            echo "--manifest requires a path value."
            exit 1
            ;;
        --output-path-suffix=*)
            OUTPUT_PATH_SUFFIX_ARG="${arg#*=}"
            ;;
        --output-path-suffix)
            echo "--output-path-suffix requires a suffix value."
            exit 1
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

find_latest_manifest() {
    "$PYTHON_BIN" - "$ROOT_DIR" "$DEFAULT_DATASET_NAME" "$DEFAULT_LLM_NAME" "$DEFAULT_API_MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_name = sys.argv[2]
llm_name = sys.argv[3]
api_model_name = sys.argv[4]

base = root / "eval_results" / llm_name
candidates = []
if base.exists():
    for manifest_path in base.glob(f"{dataset_name}_*/run_manifest.json"):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("action") != "all-workers":
            continue
        if api_model_name and data.get("api_model_name") not in {"", api_model_name}:
            continue
        candidates.append((manifest_path.stat().st_mtime, str(manifest_path)))

if not candidates:
    raise SystemExit(1)

print(max(candidates)[1])
PY
}

resolve_manifest_path() {
    if [[ -n "$RUN_MANIFEST" ]]; then
        printf '%s\n' "$RUN_MANIFEST"
        return 0
    fi

    if [[ -n "$OUTPUT_PATH_SUFFIX_ARG" ]]; then
        printf '%s\n' "$ROOT_DIR/eval_results/$DEFAULT_LLM_NAME/${DEFAULT_DATASET_NAME}${OUTPUT_PATH_SUFFIX_ARG}/run_manifest.json"
        return 0
    fi

    if [[ "$LATEST_MODE" == "1" ]]; then
        find_latest_manifest
        return 0
    fi

    find_latest_manifest
}

load_manifest_vars() {
    local manifest_env
    manifest_env="$("$PYTHON_BIN" - "$1" <<'PY'
import json
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
data = json.loads(path.read_text(encoding="utf-8"))

mapping = {
    "MANIFEST_PATH": str(path),
    "MANIFEST_LLM_NAME": data.get("llm_name", "qwen"),
    "MANIFEST_DATASET_NAME": data.get("dataset_name", "loong_full"),
    "MANIFEST_OUTPUT_PATH_SUFFIX": data.get("output_path_suffix", ""),
    "MANIFEST_EVAL_RESULTS_DIR": data.get("eval_results_dir", ""),
    "MANIFEST_WORKER_COUNT": data.get("worker_count", "1"),
    "MANIFEST_PROCESS_NUM_EVAL": data.get("process_num_eval", "20"),
    "MANIFEST_EVAL_MODEL_CONFIG": data.get("eval_model_config", "qwen_local_judge.yaml"),
    "MANIFEST_GEN_MODEL_CONFIG": data.get("gen_model_config", "qwen2.yaml"),
    "MANIFEST_URL": data.get("url", "127.0.0.1:1225"),
    "MANIFEST_API_MODEL_NAME": data.get("api_model_name", "Qwen3.5-27B"),
    "MANIFEST_STRUCTRAG_ENABLE_THINKING": data.get("structrag_enable_thinking", "0"),
    "MANIFEST_INCLUDE_ERROR_OUTPUTS_IN_SCORE": data.get("include_error_outputs_in_score", "1"),
    "MANIFEST_STRUCTURED_EVAL_PY_ROOT": data.get("structured_eval_py_root", "/workspace/LAMBO"),
    "MANIFEST_LOONG_DIR": data.get("loong_dir", ""),
    "MANIFEST_MODEL_CONFIG_DIR": data.get("model_config_dir", ""),
    "MANIFEST_TOKENIZER_PATH": data.get("tokenizer_path", ""),
    "MANIFEST_SERVER_MAX_MODEL_LEN": data.get("server_max_model_len", ""),
}

for key, value in mapping.items():
    print(f"{key}={shlex.quote(str(value or ''))}")
PY
)"
    eval "$manifest_env"
}

wait_for_server_health() {
    local url="$1"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if "$PYTHON_BIN" - "$url" <<'PY'
import sys
import requests

endpoint = sys.argv[1]
if endpoint.startswith(("http://", "https://")):
    health_url = endpoint.rstrip("/") + "/health"
else:
    health_url = f"http://{endpoint}/health"

resp = requests.get(health_url, timeout=5)
raise SystemExit(0 if resp.status_code == 200 else 1)
PY
        then
            echo "judge server is healthy: $url"
            return 0
        fi

        local now_ts
        now_ts="$(date +%s)"
        local elapsed_s
        elapsed_s=$((now_ts - start_ts))
        if (( elapsed_s >= WAIT_TIMEOUT )); then
            echo "Timed out waiting for judge server health after ${WAIT_TIMEOUT}s: $url"
            return 1
        fi

        echo "Waiting for judge server health... elapsed=${elapsed_s}s timeout=${WAIT_TIMEOUT}s"
        sleep "$WAIT_INTERVAL"
    done
}

cleanup() {
    set +e
    if [[ "$SERVER_STARTED_BY_SCRIPT" == "1" && "$STOP_SERVER_WHEN_DONE" == "1" ]]; then
        echo "Stopping judge server started by score wrapper..."
        bash "$SERVER_SCRIPT_PATH" --stop || true
    fi
}

trap cleanup EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

RUN_MANIFEST="$(resolve_manifest_path)"
if [[ ! -f "$RUN_MANIFEST" ]]; then
    echo "Run manifest not found: $RUN_MANIFEST"
    exit 1
fi

load_manifest_vars "$RUN_MANIFEST"

if [[ -z "$MANIFEST_OUTPUT_PATH_SUFFIX" ]]; then
    echo "output_path_suffix is missing in manifest: $MANIFEST_PATH"
    exit 1
fi

EVAL_RESULTS_DIR="${MANIFEST_EVAL_RESULTS_DIR:-$ROOT_DIR/eval_results/$MANIFEST_LLM_NAME/${MANIFEST_DATASET_NAME}${MANIFEST_OUTPUT_PATH_SUFFIX}}"
SCORE_LOG_PATH="${SCORE_LOG_PATH:-$EVAL_RESULTS_DIR/score_${RERUN_TAG}.log}"
SCORE_METADATA_PATH="${SCORE_METADATA_PATH:-$EVAL_RESULTS_DIR/score_manifest_${RERUN_TAG}.json}"
STRUCTURED_EVAL_OUTPUT_PATH="${STRUCTURED_EVAL_OUTPUT_PATH:-$EVAL_RESULTS_DIR/structured_eval.json}"
LAMBO_V2_JUDGE_OUTPUT_PATH="${LAMBO_V2_JUDGE_OUTPUT_PATH:-$EVAL_RESULTS_DIR/lambo_v2_llm_judge.json}"

echo "Re-running score from completed inference."
echo "manifest_path=$MANIFEST_PATH"
echo "llm_name=$MANIFEST_LLM_NAME"
echo "dataset_name=$MANIFEST_DATASET_NAME"
echo "output_path_suffix=$MANIFEST_OUTPUT_PATH_SUFFIX"
echo "eval_results_dir=$EVAL_RESULTS_DIR"
echo "score_log_path=$SCORE_LOG_PATH"
echo "score_metadata_path=$SCORE_METADATA_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, stopping before server start / score execution."
    exit 0
fi

if [[ "$START_SERVER" == "1" ]]; then
    export MODEL_DIR="${MODEL_DIR:-$MANIFEST_TOKENIZER_PATH}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$MANIFEST_SERVER_MAX_MODEL_LEN}"
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MANIFEST_API_MODEL_NAME}"
    echo "Starting judge server from wrapper..."
    bash "$SERVER_SCRIPT_PATH" --detach
    SERVER_STARTED_BY_SCRIPT=1
    wait_for_server_health "$MANIFEST_URL"
else
    echo "START_SERVER=0, assuming judge server is already available at $MANIFEST_URL"
fi

cd "$ROOT_DIR"
FORCE_OVERWRITE="$FORCE_OVERWRITE" \
INPUT_LLM_NAME="$MANIFEST_LLM_NAME" \
DATASET_NAME="$MANIFEST_DATASET_NAME" \
OUTPUT_PATH_SUFFIX="$MANIFEST_OUTPUT_PATH_SUFFIX" \
WORKER_COUNT="$MANIFEST_WORKER_COUNT" \
PROCESS_NUM_EVAL="$MANIFEST_PROCESS_NUM_EVAL" \
EVAL_MODEL_CONFIG="$MANIFEST_EVAL_MODEL_CONFIG" \
GEN_MODEL_CONFIG="$MANIFEST_GEN_MODEL_CONFIG" \
MODEL_CONFIG_DIR="$MANIFEST_MODEL_CONFIG_DIR" \
URL="$MANIFEST_URL" \
API_MODEL_NAME="$MANIFEST_API_MODEL_NAME" \
INCLUDE_ERROR_OUTPUTS_IN_SCORE="$MANIFEST_INCLUDE_ERROR_OUTPUTS_IN_SCORE" \
STRUCTURED_EVAL_PY_ROOT="$MANIFEST_STRUCTURED_EVAL_PY_ROOT" \
STRUCTRAG_ENABLE_THINKING="$MANIFEST_STRUCTRAG_ENABLE_THINKING" \
LOONG_DIR="$MANIFEST_LOONG_DIR" \
SCORE_LOG_PATH="$SCORE_LOG_PATH" \
SCORE_METADATA_PATH="$SCORE_METADATA_PATH" \
STRUCTURED_EVAL_OUTPUT_PATH="$STRUCTURED_EVAL_OUTPUT_PATH" \
LAMBO_V2_JUDGE_OUTPUT_PATH="$LAMBO_V2_JUDGE_OUTPUT_PATH" \
bash "$ROOT_DIR/run_score.sh"

echo "Score rerun completed."
echo "structured_eval_output_path=$STRUCTURED_EVAL_OUTPUT_PATH"
echo "lambo_v2_judge_output_path=$LAMBO_V2_JUDGE_OUTPUT_PATH"
