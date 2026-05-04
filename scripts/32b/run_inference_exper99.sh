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
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
STRUCTRAG_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-$MAX_MODEL_LEN}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/32b/run_server.sh}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-1800}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-15}"

EXPER99_PREPARE_SCRIPT="${EXPER99_PREPARE_SCRIPT:-/workspace/LAMBO/dawonv3/prepare_exper99_subset.py}"
EXPER99_SOURCE_PATH="${EXPER99_SOURCE_PATH:-$ROOT_DIR/loong/Loong/data/loong_process.jsonl}"
EXPER99_DATA_DIR="${EXPER99_DATA_DIR:-$ROOT_DIR/data}"
EXPER99_SUBSET_PATH="${EXPER99_SUBSET_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99.jsonl}"
EXPER99_INDICES_PATH="${EXPER99_INDICES_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99_indices.json}"
EXPER99_MANIFEST_PATH="${EXPER99_MANIFEST_PATH:-$EXPER99_DATA_DIR/loong_set1_balanced99_manifest.json}"

usage() {
    cat <<EOF
Usage:
  bash scripts/32b/run_inference_exper99.sh
  bash scripts/32b/run_inference_exper99.sh --logging

Behavior:
  - Prepares the same deterministic SET1 balanced 99-sample subset used by LAMBO/dawonv3/dawonv4
  - Excludes known context-length failure indices and fills replacements per domain
  - Runs StructRAG on that subset with 32B defaults
  - Includes final_output_error_*.jsonl in scoring as failed cases
  - Saves EM-style structured metrics plus Loong LLM-as-eval metrics

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  DATASET_NAME=loong_exper99
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  WORKER_COUNT=1
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

    cat > "$STRUCTRAG_LOGGING_DIR/inference/launch_exper99.env" <<EOF
STRUCTRAG_LOGGING=1
STRUCTRAG_LOGGING_ROOT=$logging_root
STRUCTRAG_LOGGING_RUN_ID=$STRUCTRAG_LOGGING_RUN_ID
STRUCTRAG_LOGGING_DIR=$STRUCTRAG_LOGGING_DIR
MODEL_DIR=$MODEL_DIR
EXPER99_SUBSET_PATH=$EXPER99_SUBSET_PATH
ARGV=${FORWARDED_ARGS[*]:-exper99}
EOF

    export STRUCTRAG_LOGGING=1
    export STRUCTRAG_LOGGING_RUN_ID
    export STRUCTRAG_LOGGING_DIR
}

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash scripts/32b/run_inference_exper99.sh"
    exit 1
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

export MODEL_PATH="$MODEL_DIR"
export TOKENIZER_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export STRUCTRAG_MAX_INPUT_TOKENS
export SERVER_SCRIPT_PATH
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
