#!/usr/bin/env bash
# HuggingFace transformers inference server — drop-in replacement for vLLM run_server.sh
# Usage: bash run_hf_server.sh [--detach] [--stop]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

PYTHON_BIN="${SERVER_PYTHON_BIN:-${PYTHON_BIN:-/workspace/venvs/real_dreamvenv/bin/python}}"
MODEL_DIR="${MODEL_DIR:-${MODEL_PATH:-$ROOT_DIR/model/Qwen3.5-27B}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/logs/qwen35_27b_vllm.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/logs/qwen35_27b_vllm.pid}"
PGID_FILE="${PGID_FILE:-${PID_FILE}.pgid}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

SERVER_SCRIPT="$SCRIPT_DIR/hf_inference_server.py"

mkdir -p "$(dirname "$LOG_PATH")"

# ---------- --stop ----------
if [[ "${1:-}" == "--stop" ]]; then
    stopped=0
    if [[ -f "$PGID_FILE" ]]; then
        pgid=$(cat "$PGID_FILE")
        if kill -0 -- "-$pgid" 2>/dev/null; then
            kill -- "-$pgid" 2>/dev/null && echo "Stopped vLLM server pgid=$pgid" && stopped=1
        fi
        rm -f "$PGID_FILE"
    fi
    if [[ -f "$PID_FILE" ]]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo "Stopped vLLM server pid=$pid" && stopped=1
        fi
        rm -f "$PID_FILE"
    fi
    [[ $stopped -eq 0 ]] && echo "PID/PGID file not found: $PID_FILE / $PGID_FILE"
    exit 0
fi

# ---------- start ----------
echo "Starting HF inference server: $MODEL_DIR on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  python: $PYTHON_BIN"
echo "  log: $LOG_PATH"

GPU_IDS="${CUDA_VISIBLE_DEVICES}"

if [[ "${1:-}" == "--detach" ]]; then
    nohup "$PYTHON_BIN" "$SERVER_SCRIPT" \
        --model "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --gpu-ids "$GPU_IDS" \
        --max-model-len "$MAX_MODEL_LEN" \
        > "$LOG_PATH" 2>&1 &
    server_pid=$!
    echo $server_pid > "$PID_FILE"
    echo $server_pid > "$PGID_FILE"
    echo "HF server started: pid=$server_pid"

    # Wait until port is listening (model loading can take 5+ min)
    echo "Waiting for server to be ready (timeout=${READY_TIMEOUT}s)..."
    elapsed=0
    while [[ $elapsed -lt $READY_TIMEOUT ]]; do
        # Check port listening
        if python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('${HOST}', ${PORT}))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
            echo "Server ready after ${elapsed}s"
            exit 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "Server did not become ready within ${READY_TIMEOUT}s"
    exit 1
else
    # foreground
    exec "$PYTHON_BIN" "$SERVER_SCRIPT" \
        --model "$MODEL_DIR" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --gpu-ids "$GPU_IDS" \
        --max-model-len "$MAX_MODEL_LEN"
fi
