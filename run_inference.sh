#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
LOONG_DIR="${LOONG_DIR:-$ROOT_DIR/loong/Loong}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
URL="${URL:-$HOST:$PORT}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen}"
LLM_NAME="${LLM_NAME:-qwen}"
DATASET_NAME="${DATASET_NAME:-loong}"

USER_OUTPUT_LABEL="${OUTPUT_PATH_SUFFIX:-}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
AUTO_SCORE="${AUTO_SCORE:-1}"
AUTO_SCORE_FORCE_OVERWRITE="${AUTO_SCORE_FORCE_OVERWRITE:-1}"
WORKER_COUNT="${WORKER_COUNT:-8}"
PROCESS_NUM_EVAL="${PROCESS_NUM_EVAL:-20}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$LOONG_DIR/config/models}"
GUIDED_DECODING_BACKEND="${STRUCTRAG_GUIDED_DECODING_BACKEND:-lm-format-enforcer}"

RUN_OUTPUT_PATH_SUFFIX=""
RUN_EVAL_RESULTS_DIR=""
RUN_INTERMEDIATE_DIR=""
RUN_METADATA_PATH=""
ACTION_DESCRIPTOR=""

usage() {
    cat <<EOF
Usage:
  bash run_inference.sh sample5
  bash run_inference.sh sample10
  bash run_inference.sh sample100
  bash run_inference.sh single <dataset_id>
  bash run_inference.sh worker <worker_id> [extra main.py args]
  bash run_inference.sh all_workers [extra main.py args]
  OUTPUT_PATH_SUFFIX=<existing_suffix> bash run_inference.sh merge

Behavior:
  - Automatically creates a run-specific result folder name with timestamp and settings
  - Automatically runs scoring after inference unless AUTO_SCORE=0
  - Writes run metadata to eval_results/.../run_manifest.json

Environment overrides:
  VENV_PATH=/workspace/venvs/structrag
  TOKENIZER_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  LOONG_DIR=$ROOT_DIR/loong/Loong
  URL=127.0.0.1:1225
  API_MODEL_NAME=Qwen
  OUTPUT_PATH_SUFFIX=_mylabel
  AUTO_SCORE=1
  PROCESS_NUM_EVAL=20
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml

Examples:
  bash run_inference.sh sample5
  bash run_inference.sh sample100
  OUTPUT_PATH_SUFFIX=_ablationA bash run_inference.sh sample5
  bash run_inference.sh single 13a4a371-6339-4c9d-82cf-fc9ab2bb017d
  AUTO_SCORE=0 bash run_inference.sh worker 0 --limit 1 --no_shuffle
EOF
}

slugify() {
    local input="${1:-}"
    local max_len="${2:-32}"
    local lowered
    lowered="$(printf '%s' "$input" | tr '[:upper:]' '[:lower:]')"
    local cleaned
    cleaned="$(printf '%s' "$lowered" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
    if [[ -z "$cleaned" ]]; then
        cleaned="na"
    fi
    printf '%s' "${cleaned:0:max_len}"
}

compute_run_suffix() {
    local action_desc="$1"
    local tokenizer_name
    tokenizer_name="$(basename "$TOKENIZER_PATH")"
    local gen_cfg_name
    gen_cfg_name="$(basename "$GEN_MODEL_CONFIG" .yaml)"
    local eval_cfg_name
    eval_cfg_name="$(basename "$EVAL_MODEL_CONFIG" .yaml)"
    local url_slug
    url_slug="$(slugify "$URL" 20)"
    local api_slug
    api_slug="$(slugify "$API_MODEL_NAME" 20)"
    local tok_slug
    tok_slug="$(slugify "$tokenizer_name" 28)"
    local action_slug
    action_slug="$(slugify "$action_desc" 24)"
    local gen_slug
    gen_slug="$(slugify "$gen_cfg_name" 16)"
    local eval_slug
    eval_slug="$(slugify "$eval_cfg_name" 16)"
    local guide_slug
    guide_slug="$(slugify "$GUIDED_DECODING_BACKEND" 18)"
    local label_slug=""
    if [[ -n "$USER_OUTPUT_LABEL" ]]; then
        label_slug="$(slugify "${USER_OUTPUT_LABEL#_}" 24)"
    fi

    local suffix="_ts-${RUN_TIMESTAMP}_act-${action_slug}_api-${api_slug}_tok-${tok_slug}_gen-${gen_slug}_eval-${eval_slug}_guide-${guide_slug}_url-${url_slug}"
    if [[ -n "$label_slug" ]]; then
        suffix="${suffix}_lbl-${label_slug}"
    fi
    printf '%s' "$suffix"
}

prepare_run_paths() {
    RUN_OUTPUT_PATH_SUFFIX="$(compute_run_suffix "$ACTION_DESCRIPTOR")"
    RUN_EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
    RUN_INTERMEDIATE_DIR="$ROOT_DIR/intermediate_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
    RUN_METADATA_PATH="$RUN_EVAL_RESULTS_DIR/run_manifest.json"
    mkdir -p "$RUN_EVAL_RESULTS_DIR" "$RUN_INTERMEDIATE_DIR"
}

write_run_metadata() {
    local status="$1"
    mkdir -p "$RUN_EVAL_RESULTS_DIR"
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = {
    "status": "$status",
    "run_timestamp": "$RUN_TIMESTAMP",
    "action": "$ACTION_DESCRIPTOR",
    "llm_name": "$LLM_NAME",
    "dataset_name": "$DATASET_NAME",
    "url": "$URL",
    "api_model_name": "$API_MODEL_NAME",
    "tokenizer_path": "$TOKENIZER_PATH",
    "loong_dir": "$LOONG_DIR",
    "output_path_suffix": "$RUN_OUTPUT_PATH_SUFFIX",
    "eval_results_dir": "$RUN_EVAL_RESULTS_DIR",
    "intermediate_results_dir": "$RUN_INTERMEDIATE_DIR",
    "auto_score": "$AUTO_SCORE",
    "worker_count": "$WORKER_COUNT",
    "process_num_eval": "$PROCESS_NUM_EVAL",
    "eval_model_config": "$EVAL_MODEL_CONFIG",
    "gen_model_config": "$GEN_MODEL_CONFIG",
    "model_config_dir": "$MODEL_CONFIG_DIR",
    "guided_decoding_backend": "$GUIDED_DECODING_BACKEND",
}

Path(r"$RUN_METADATA_PATH").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
}

run_main() {
    local worker_id="$1"
    shift
    cd "$ROOT_DIR"
    STRUCTRAG_GUIDED_DECODING_BACKEND="$GUIDED_DECODING_BACKEND" "$PYTHON_BIN" main.py \
        --url "$URL" \
        --worker_id "$worker_id" \
        --llm_name "$LLM_NAME" \
        --dataset_name "$DATASET_NAME" \
        --loong_dir "$LOONG_DIR" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --api_model_name "$API_MODEL_NAME" \
        --output_path_suffix "$RUN_OUTPUT_PATH_SUFFIX" \
        "$@"
}

run_auto_score() {
    if [[ "$AUTO_SCORE" != "1" ]]; then
        echo "AUTO_SCORE=0, skipping scoring."
        return 10
    fi

    echo ""
    echo "Starting automatic scoring..."
    cd "$ROOT_DIR"
    FORCE_OVERWRITE="$AUTO_SCORE_FORCE_OVERWRITE" \
    INPUT_LLM_NAME="$LLM_NAME" \
    DATASET_NAME="$DATASET_NAME" \
    OUTPUT_PATH_SUFFIX="$RUN_OUTPUT_PATH_SUFFIX" \
    WORKER_COUNT="$WORKER_COUNT" \
    PROCESS_NUM_EVAL="$PROCESS_NUM_EVAL" \
    EVAL_MODEL_CONFIG="$EVAL_MODEL_CONFIG" \
    GEN_MODEL_CONFIG="$GEN_MODEL_CONFIG" \
    MODEL_CONFIG_DIR="$MODEL_CONFIG_DIR" \
    LOONG_DIR="$LOONG_DIR" \
    RUN_TIMESTAMP="$RUN_TIMESTAMP" \
    bash "$ROOT_DIR/run_score.sh"
    return 0
}

show_run_summary() {
    echo ""
    echo "Run completed."
    echo "run_timestamp=$RUN_TIMESTAMP"
    echo "action=$ACTION_DESCRIPTOR"
    echo "output_path_suffix=$RUN_OUTPUT_PATH_SUFFIX"
    echo "eval_results_dir=$RUN_EVAL_RESULTS_DIR"
    echo "intermediate_results_dir=$RUN_INTERMEDIATE_DIR"
    echo "run_metadata_path=$RUN_METADATA_PATH"
    echo "final_output_path=$RUN_EVAL_RESULTS_DIR/final_output_0.jsonl"
    echo "final_error_path=$RUN_EVAL_RESULTS_DIR/final_output_error_0.jsonl"
    if [[ "$AUTO_SCORE" == "1" ]]; then
        echo "score_log_path=$RUN_EVAL_RESULTS_DIR/score.log"
        echo "score_metadata_path=$RUN_EVAL_RESULTS_DIR/score_manifest.json"
    fi
}

finalize_after_inference() {
    write_run_metadata "inference_completed"
    if run_auto_score; then
        write_run_metadata "scored"
    else
        write_run_metadata "inference_completed"
    fi
    show_run_summary
}

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -e "$TOKENIZER_PATH" ]]; then
    echo "Tokenizer path not found: $TOKENIZER_PATH"
    exit 1
fi

if [[ ! -e "$LOONG_DIR/data/loong_process.jsonl" ]]; then
    echo "Loong dataset not found: $LOONG_DIR/data/loong_process.jsonl"
    exit 1
fi

case "$ACTION" in
    sample5)
        ACTION_DESCRIPTOR="sample5"
        prepare_run_paths
        write_run_metadata "running"
        run_main 0 --limit 5 --no_shuffle "$@"
        finalize_after_inference
        ;;
    sample10)
        ACTION_DESCRIPTOR="sample10"
        prepare_run_paths
        write_run_metadata "running"
        run_main 0 --limit 10 --no_shuffle "$@"
        finalize_after_inference
        ;;
    sample100)
        ACTION_DESCRIPTOR="sample100"
        prepare_run_paths
        write_run_metadata "running"
        run_main 0 --limit 100 --no_shuffle "$@"
        finalize_after_inference
        ;;
    single)
        DATASET_ID="${1:?dataset_id is required}"
        shift
        ACTION_DESCRIPTOR="single-$(slugify "$DATASET_ID" 16)"
        prepare_run_paths
        write_run_metadata "running"
        run_main 0 --only_id "$DATASET_ID" --limit 1 --no_shuffle "$@"
        finalize_after_inference
        ;;
    worker)
        WORKER_ID="${1:?worker_id is required}"
        shift
        ACTION_DESCRIPTOR="worker-${WORKER_ID}"
        prepare_run_paths
        write_run_metadata "running"
        run_main "$WORKER_ID" "$@"
        finalize_after_inference
        ;;
    all_workers)
        ACTION_DESCRIPTOR="all-workers"
        prepare_run_paths
        write_run_metadata "running"
        for worker_id in 0 1 2 3 4 5 6 7; do
            run_main "$worker_id" "$@"
        done
        finalize_after_inference
        ;;
    merge)
        if [[ -z "$USER_OUTPUT_LABEL" ]]; then
            echo "merge action requires OUTPUT_PATH_SUFFIX to point to an existing run."
            exit 1
        fi
        ACTION_DESCRIPTOR="merge-only"
        RUN_OUTPUT_PATH_SUFFIX="$USER_OUTPUT_LABEL"
        RUN_EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
        RUN_INTERMEDIATE_DIR="$ROOT_DIR/intermediate_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
        RUN_METADATA_PATH="$RUN_EVAL_RESULTS_DIR/run_manifest.json"
        write_run_metadata "merge_only"
        cd "$ROOT_DIR"
        "$PYTHON_BIN" do_merge_each_batch.py \
            --llm_name "$LLM_NAME" \
            --dataset_name "$DATASET_NAME" \
            --output_path_suffix "$RUN_OUTPUT_PATH_SUFFIX" \
            --loong_dir "$LOONG_DIR" \
            "$@"
        show_run_summary
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown action: $ACTION"
        usage
        exit 1
        ;;
esac
