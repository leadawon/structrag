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
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-qwen35-think-off}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/27b/run_server.sh}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm.pid}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${SERVER_PID_FILE}.pgid}"
LOG_PATH="${LOG_PATH:-$SERVER_LOG_PATH}"
PID_FILE="${PID_FILE:-$SERVER_PID_FILE}"
PGID_FILE="${PGID_FILE:-$SERVER_PGID_FILE}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

EXPER99_PREPARE_SCRIPT="${EXPER99_PREPARE_SCRIPT:-/workspace/LAMBO/dawon_all/dawonv3/prepare_exper99_subset.py}"
EXPER99_SOURCE_PATH="${EXPER99_SOURCE_PATH:-$ROOT_DIR/loong/Loong/data/loong_process.jsonl}"
EXPER99_DATA_DIR="${EXPER99_DATA_DIR:-$ROOT_DIR/data}"
EXPER99_SUBSET_PATH="${EXPER99_SUBSET_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99.jsonl}"
EXPER99_INDICES_PATH="${EXPER99_INDICES_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99_indices.json}"
EXPER99_MANIFEST_PATH="${EXPER99_MANIFEST_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99_manifest.json}"
LAMBO_V2_EXPER99_MANIFEST="${LAMBO_V2_EXPER99_MANIFEST:-/workspace/LAMBO/logs/lambo_v2_exper99_qwen35_27b/manifest.json}"
VALIDATE_LAMBO_V2_EXPER99="${VALIDATE_LAMBO_V2_EXPER99:-1}"

usage() {
    cat <<EOF
Usage:
  bash scripts/27b/run_inference_exper99.sh
  bash scripts/27b/run_inference_exper99.sh --logging

Behavior:
  - Prepares the same deterministic SET1 balanced 99-sample subset used by LAMBO
  - Validates the prepared indices against LAMBO v2's exper99 manifest when present
  - Excludes known context-length failure indices and fills replacements per domain
  - Runs StructRAG on that subset with Qwen3.5 27B defaults
  - Includes final_output_error_*.jsonl in scoring as failed cases
  - Saves EM-style structured metrics plus Loong LLM-as-eval metrics

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen3.5-27B
  DATASET_NAME=loong_exper99
  API_MODEL_NAME=Qwen3.5-27B
  STRUCTRAG_ENABLE_THINKING=0
  OUTPUT_PATH_SUFFIX=qwen35-think-off
  AUTO_DOWNLOAD_MODEL=1
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  WORKER_COUNT=1
  VALIDATE_LAMBO_V2_EXPER99=1
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
    local logging_root="${STRUCTRAG_LOGGING_ROOT:-$ROOT_DIR/27b_logging}"
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

    cat > "$STRUCTRAG_LOGGING_DIR/inference/launch_exper99.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
API_MODEL_NAME=$API_MODEL_NAME
STRUCTRAG_ENABLE_THINKING=$STRUCTRAG_ENABLE_THINKING
OUTPUT_PATH_SUFFIX=$OUTPUT_PATH_SUFFIX
EXPER99_SUBSET_PATH=$EXPER99_SUBSET_PATH
LAMBO_V2_EXPER99_MANIFEST=$LAMBO_V2_EXPER99_MANIFEST
ARGV=${FORWARDED_ARGS[*]:-exper99}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

validate_lambo_v2_exper99() {
    if [[ "$VALIDATE_LAMBO_V2_EXPER99" != "1" ]]; then
        return 0
    fi

    if [[ ! -f "$LAMBO_V2_EXPER99_MANIFEST" ]]; then
        echo "LAMBO v2 exper99 manifest not found; skipping validation: $LAMBO_V2_EXPER99_MANIFEST"
        return 0
    fi

    "$PYTHON_BIN" - "$EXPER99_INDICES_PATH" "$LAMBO_V2_EXPER99_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

structrag_indices_path = Path(sys.argv[1])
lambo_manifest_path = Path(sys.argv[2])

structrag_payload = json.loads(structrag_indices_path.read_text(encoding="utf-8"))
structrag_indices = structrag_payload["indices"] if isinstance(structrag_payload, dict) else structrag_payload
lambo_manifest = json.loads(lambo_manifest_path.read_text(encoding="utf-8"))
lambo_indices = [item["selected_index"] for item in lambo_manifest]

if len(structrag_indices) != 99:
    raise SystemExit(f"StructRAG exper99 index count is {len(structrag_indices)}, expected 99")
if len(lambo_indices) != 99:
    raise SystemExit(f"LAMBO v2 exper99 index count is {len(lambo_indices)}, expected 99")
if structrag_indices != lambo_indices:
    only_structrag = sorted(set(structrag_indices) - set(lambo_indices))
    only_lambo = sorted(set(lambo_indices) - set(structrag_indices))
    raise SystemExit(
        "StructRAG exper99 indices differ from LAMBO v2 manifest "
        f"(same_set={set(structrag_indices) == set(lambo_indices)}, "
        f"only_structrag={only_structrag}, only_lambo={only_lambo})"
    )

print(
    "validated_lambo_v2_exper99=1 "
    f"count={len(structrag_indices)} first10={structrag_indices[:10]} last10={structrag_indices[-10:]}"
)
PY
}

download_model_if_missing() {
    if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Checked candidates:"
        for candidate in "${MODEL_CANDIDATES[@]}"; do
            echo "  - $candidate"
        done
        echo "Prior LAMBO v2 logs used: /workspace/lambo/models/Qwen3.5-27B"
        echo "Example:"
        echo "  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts/27b/run_inference_exper99.sh"
        exit 1
    fi

    if [[ ! -f "$DOWNLOAD_MODEL_SCRIPT" ]]; then
        echo "Model path is missing or incomplete: $MODEL_DIR"
        echo "Auto-download script not found: $DOWNLOAD_MODEL_SCRIPT"
        exit 1
    fi

    echo "Model path is missing or incomplete: $MODEL_DIR"
    echo "AUTO_DOWNLOAD_MODEL=1, downloading the LAMBO v2 Qwen3.5 27B model first."
    echo "Download target: $MODEL_DIR"
    MODEL_DIR="$MODEL_DIR" bash "$DOWNLOAD_MODEL_SCRIPT"

    if ! model_ready "$MODEL_DIR"; then
        echo "Download finished, but model files are still incomplete: $MODEL_DIR"
        exit 1
    fi
}

if ! model_ready "$MODEL_DIR"; then
    download_model_if_missing
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -f "$EXPER99_PREPARE_SCRIPT" ]]; then
    echo "Subset prepare script not found: $EXPER99_PREPARE_SCRIPT"
    exit 1
fi

"$PYTHON_BIN" "$EXPER99_PREPARE_SCRIPT" \
    --input "$EXPER99_SOURCE_PATH" \
    --subset-output "$EXPER99_SUBSET_PATH" \
    --indices-output "$EXPER99_INDICES_PATH" \
    --manifest-output "$EXPER99_MANIFEST_PATH"

validate_lambo_v2_exper99

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
export DATASET_NAME="${DATASET_NAME:-loong_exper99}"
export WORKER_COUNT="${WORKER_COUNT:-1}"
export EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
export GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
export INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-1}"
export STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-/workspace/LAMBO}"

if [[ "$LOGGING_MODE" -eq 1 ]]; then
    setup_logging_mode
    echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
    echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
fi

cd "$ROOT_DIR"

bash "$ROOT_DIR/run_inference.sh" exper99 --eval_data_path "$EXPER99_SUBSET_PATH" "${FORWARDED_ARGS[@]}"
