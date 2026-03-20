#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
LOONG_DIR="${LOONG_DIR:-$ROOT_DIR/loong/Loong}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
URL="${URL:-$HOST:$PORT}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen}"
API_MODEL_NAME="${API_MODEL_NAME:-$SERVED_MODEL_NAME}"
LLM_NAME="${LLM_NAME:-qwen}"
DATASET_NAME="${DATASET_NAME:-loong}"
OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-$ROOT_DIR/tmp}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-}"

usage() {
    cat <<EOF
Usage:
  bash inference.sh server
  bash inference.sh sample5
  bash inference.sh sample10
  bash inference.sh single <dataset_id>
  bash inference.sh worker <worker_id>
  bash inference.sh merge
  bash inference.sh all_workers

Environment overrides:
  VENV_PATH=/workspace/venvs/structrag
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  TOKENIZER_PATH=\$MODEL_PATH
  LOONG_DIR=$ROOT_DIR/loong/Loong
  URL=127.0.0.1:1225
  CUDA_DEVICES=0,1,2,3
  TENSOR_PARALLEL_SIZE=4
  OUTPUT_PATH_SUFFIX=_debug

Examples:
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash inference.sh server
  OUTPUT_PATH_SUFFIX=_sample5 bash inference.sh sample5
  OUTPUT_PATH_SUFFIX=_one bash inference.sh single 13a4a371-6339-4c9d-82cf-fc9ab2bb017d
EOF
}

run_main() {
    local worker_id="$1"
    shift
    if [[ ! -e "$LOONG_DIR/data/loong_process.jsonl" ]]; then
        echo "Loong dataset not found: $LOONG_DIR/data/loong_process.jsonl"
        exit 1
    fi
    if [[ ! -e "$TOKENIZER_PATH" ]]; then
        echo "Tokenizer path not found: $TOKENIZER_PATH"
        exit 1
    fi
    cd "$ROOT_DIR"
    "$PYTHON_BIN" main.py \
        --url "$URL" \
        --worker_id "$worker_id" \
        --llm_name "$LLM_NAME" \
        --dataset_name "$DATASET_NAME" \
        --loong_dir "$LOONG_DIR" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --api_model_name "$API_MODEL_NAME" \
        --output_path_suffix "$OUTPUT_PATH_SUFFIX" \
        "$@"
}

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

case "$ACTION" in
    server)
        if [[ ! -e "$MODEL_PATH" ]]; then
            echo "Model path not found: $MODEL_PATH"
            exit 1
        fi
        if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
            echo "vllm is not installed in: $PYTHON_BIN"
            exit 1
        fi
        mkdir -p "$OUTLINES_CACHE_DIR"
        cd "$ROOT_DIR"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" nohup "$PYTHON_BIN" \
            -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --port "$PORT" \
            --disable-custom-all-reduce \
            > "$ROOT_DIR/vllm.log" 2>&1 &
        echo "vLLM server started. log=$ROOT_DIR/vllm.log"
        ;;
    sample5)
        run_main 0 --limit 5 --no_shuffle "$@"
        ;;
    sample10)
        run_main 0 --limit 10 --no_shuffle "$@"
        ;;
    single)
        DATASET_ID="${1:?dataset_id is required}"
        shift
        run_main 0 --only_id "$DATASET_ID" --limit 1 --no_shuffle "$@"
        ;;
    worker)
        WORKER_ID="${1:?worker_id is required}"
        shift
        run_main "$WORKER_ID" "$@"
        ;;
    all_workers)
        for worker_id in 0 1 2 3 4 5 6 7; do
            run_main "$worker_id" "$@"
        done
        ;;
    merge)
        cd "$ROOT_DIR"
        "$PYTHON_BIN" do_merge_each_batch.py \
            --llm_name "$LLM_NAME" \
            --dataset_name "$DATASET_NAME" \
            --output_path_suffix "$OUTPUT_PATH_SUFFIX" \
            --loong_dir "$LOONG_DIR" \
            "$@"
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
