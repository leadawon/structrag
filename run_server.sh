#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen}"
OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-$ROOT_DIR/tmp}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/vllm.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/vllm.pid}"
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-lm-format-enforcer}"

usage() {
    cat <<EOF
Usage:
  bash run_server.sh
  bash run_server.sh --detach
  bash run_server.sh --stop

Environment overrides:
  VENV_PATH=/workspace/venvs/structrag
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  HOST=127.0.0.1
  PORT=1225
  CUDA_DEVICES=0,1,2,3
  TENSOR_PARALLEL_SIZE=4
  SERVED_MODEL_NAME=Qwen
  GUIDED_DECODING_BACKEND=lm-format-enforcer
  LOG_PATH=$ROOT_DIR/vllm.log
  PID_FILE=$ROOT_DIR/vllm.pid

Example:
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash run_server.sh
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash run_server.sh --detach
EOF
}

DETACH_MODE=0
STOP_MODE=0

case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --detach)
        DETACH_MODE=1
        ;;
    --stop)
        STOP_MODE=1
        ;;
    "")
        ;;
    *)
        echo "Unknown option: ${1}"
        usage
        exit 1
        ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ "$STOP_MODE" -eq 1 ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "PID file not found: $PID_FILE"
        exit 1
    fi

    SERVER_PID="$(cat "$PID_FILE")"
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID"
        rm -f "$PID_FILE"
        echo "Stopped vLLM server pid=$SERVER_PID"
    else
        rm -f "$PID_FILE"
        echo "Process not running, removed stale PID file: $SERVER_PID"
    fi
    exit 0
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
    echo "vllm is not installed in: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$OUTLINES_CACHE_DIR"
mkdir -p "$(dirname "$LOG_PATH")"

cd "$ROOT_DIR"
if [[ "$DETACH_MODE" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" nohup "$PYTHON_BIN" \
        -m vllm.entrypoints.openai.api_server \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --guided-decoding-backend "$GUIDED_DECODING_BACKEND" \
        --disable-custom-all-reduce \
        > "$LOG_PATH" 2>&1 &

    SERVER_PID="$!"
    echo "$SERVER_PID" > "$PID_FILE"
    echo "vLLM server started in background"
    echo "pid=$SERVER_PID"
    echo "model_path=$MODEL_PATH"
    echo "url=http://$HOST:$PORT/v1/chat/completions"
    echo "log_path=$LOG_PATH"
    echo "pid_file=$PID_FILE"
else
    echo "Starting vLLM server in foreground. Press Ctrl+C to stop."
    echo "model_path=$MODEL_PATH"
    echo "url=http://$HOST:$PORT/v1/chat/completions"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" exec "$PYTHON_BIN" \
        -m vllm.entrypoints.openai.api_server \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --guided-decoding-backend "$GUIDED_DECODING_BACKEND" \
        --disable-custom-all-reduce
fi
