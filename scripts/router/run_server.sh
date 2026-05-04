#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROUTER_GGUF_PATH="${ROUTER_GGUF_PATH:-$ROOT_DIR/model/qwen2.5-7b-structrag-router-q8_0.gguf}"
ROUTER_HF_REPO="${ROUTER_HF_REPO:-selimsheker/Qwen2.5-7B-StructRAG-router-Q8_0-GGUF}"
ROUTER_HF_FILE="${ROUTER_HF_FILE:-qwen2.5-7b-structrag-router-q8_0.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1226}"
CTX_SIZE="${CTX_SIZE:-4096}"
PARALLEL="${PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/logs/learned_router_gguf.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/logs/learned_router_gguf.pid}"
LLAMA_SERVER_EXTRA_ARGS="${LLAMA_SERVER_EXTRA_ARGS:-}"
SYSTEM_LIBSTDCPP_DIR="${SYSTEM_LIBSTDCPP_DIR:-/usr/lib/x86_64-linux-gnu}"

usage() {
    cat <<EOF
Usage:
  bash scripts/router/run_server.sh
  bash scripts/router/run_server.sh --detach
  bash scripts/router/run_server.sh --stop

Defaults:
  ROUTER_GGUF_PATH=$ROOT_DIR/model/qwen2.5-7b-structrag-router-q8_0.gguf
  ROUTER_HF_REPO=$ROUTER_HF_REPO
  ROUTER_HF_FILE=$ROUTER_HF_FILE
  HOST=127.0.0.1
  PORT=1226
  CTX_SIZE=4096
  PARALLEL=1
  N_GPU_LAYERS=0

Notes:
  - If ROUTER_GGUF_PATH exists, it is used directly.
  - Otherwise the script falls back to llama.cpp's --hf-repo/--hf-file mode.
  - Direct HF loading requires a llama-server build with Hugging Face download support.
  - Default is CPU router (N_GPU_LAYERS=0) to avoid VRAM contention with the main vLLM model.
EOF
}

prepare_runtime_env() {
    if [[ -d "$SYSTEM_LIBSTDCPP_DIR" ]]; then
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            export LD_LIBRARY_PATH="$SYSTEM_LIBSTDCPP_DIR:$LD_LIBRARY_PATH"
        else
            export LD_LIBRARY_PATH="$SYSTEM_LIBSTDCPP_DIR"
        fi
    fi
}

resolve_llama_server_bin() {
    local candidate=""
    for candidate in \
        "$LLAMA_SERVER_BIN" \
        "$ROOT_DIR/llama.cpp/build/bin/llama-server" \
        "$ROOT_DIR/llama.cpp/bin/llama-server" \
        "$(command -v llama-server 2>/dev/null || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

build_cmd() {
    ROUTER_CMD=(
        "$RESOLVED_LLAMA_SERVER_BIN"
        --host "$HOST"
        --port "$PORT"
        -c "$CTX_SIZE"
        -np "$PARALLEL"
    )

    if [[ -n "$N_GPU_LAYERS" ]]; then
        ROUTER_CMD+=(-ngl "$N_GPU_LAYERS")
    fi

    if [[ -f "$ROUTER_GGUF_PATH" ]]; then
        ROUTER_CMD+=(-m "$ROUTER_GGUF_PATH")
    else
        ROUTER_CMD+=(--hf-repo "$ROUTER_HF_REPO" --hf-file "$ROUTER_HF_FILE")
    fi

    if [[ -n "$LLAMA_SERVER_EXTRA_ARGS" ]]; then
        local extra_args=()
        read -r -a extra_args <<< "$LLAMA_SERVER_EXTRA_ARGS"
        ROUTER_CMD+=("${extra_args[@]}")
    fi
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

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$PID_FILE")"

if [[ "$STOP_MODE" -eq 1 ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "PID file not found: $PID_FILE"
        exit 1
    fi

    ROUTER_PID="$(cat "$PID_FILE")"
    if kill -0 "$ROUTER_PID" >/dev/null 2>&1; then
        kill "$ROUTER_PID"
        rm -f "$PID_FILE"
        echo "Stopped learned router server pid=$ROUTER_PID"
    else
        rm -f "$PID_FILE"
        echo "Process not running, removed stale PID file: $ROUTER_PID"
    fi
    exit 0
fi

if ! RESOLVED_LLAMA_SERVER_BIN="$(resolve_llama_server_bin)"; then
    echo "llama-server binary not found."
    echo "Set LLAMA_SERVER_BIN=/path/to/llama-server or install/build llama.cpp first."
    echo "Reference:"
    echo "  https://huggingface.co/selimsheker/Qwen2.5-7B-StructRAG-router-Q8_0-GGUF"
    exit 1
fi

prepare_runtime_env
build_cmd

if [[ "$DETACH_MODE" -eq 1 ]]; then
    nohup "${ROUTER_CMD[@]}" > "$LOG_PATH" 2>&1 &
    ROUTER_PID="$!"
    echo "$ROUTER_PID" > "$PID_FILE"
    echo "Learned router server started in background"
    echo "pid=$ROUTER_PID"
    echo "router_url=http://$HOST:$PORT/v1/chat/completions"
    echo "log_path=$LOG_PATH"
    echo "pid_file=$PID_FILE"
    if [[ -f "$ROUTER_GGUF_PATH" ]]; then
        echo "router_model_path=$ROUTER_GGUF_PATH"
    else
        echo "router_hf_repo=$ROUTER_HF_REPO"
        echo "router_hf_file=$ROUTER_HF_FILE"
    fi
else
    echo "Starting learned router server in foreground. Press Ctrl+C to stop."
    echo "router_url=http://$HOST:$PORT/v1/chat/completions"
    if [[ -f "$ROUTER_GGUF_PATH" ]]; then
        echo "router_model_path=$ROUTER_GGUF_PATH"
    else
        echo "router_hf_repo=$ROUTER_HF_REPO"
        echo "router_hf_file=$ROUTER_HF_FILE"
    fi
    exec "${ROUTER_CMD[@]}"
fi
