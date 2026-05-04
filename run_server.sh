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
PGID_FILE="${PGID_FILE:-${PID_FILE}.pgid}"
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-lm-format-enforcer}"
DTYPE="${DTYPE:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
QUANTIZATION="${QUANTIZATION:-}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN:-0}"
CLEAN_STALE_VLLM="${CLEAN_STALE_VLLM:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  DTYPE=bfloat16
  REASONING_PARSER=qwen3
  MAX_MODEL_LEN=65536
  GPU_MEMORY_UTILIZATION=0.9
  MAX_NUM_SEQS=8
  QUANTIZATION=awq
  DISABLE_CUSTOM_ALL_REDUCE=1
  ENFORCE_EAGER=1
  TRUST_REMOTE_CODE=1
  ALLOW_LONG_MAX_MODEL_LEN=1
  CLEAN_STALE_VLLM=0
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  LOG_PATH=$ROOT_DIR/vllm.log
  PID_FILE=$ROOT_DIR/vllm.pid
  PGID_FILE=$ROOT_DIR/vllm.pid.pgid

Example:
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash run_server.sh
  MODEL_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct bash run_server.sh --detach
EOF
}

port_in_use() {
    "$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sys.exit(0)
    sys.exit(1)
PY
}

vllm_arg_supported() {
    local arg_name="$1"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -F -- "$arg_name" >/dev/null
}

build_server_cmd() {
    SERVER_CMD=(
        "$PYTHON_BIN"
        -m
        vllm.entrypoints.openai.api_server
        --host
        "$HOST"
        --port
        "$PORT"
        --model
        "$MODEL_PATH"
        --served-model-name
        "$SERVED_MODEL_NAME"
        --tensor-parallel-size
        "$TENSOR_PARALLEL_SIZE"
    )

    if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
        SERVER_CMD+=(--disable-custom-all-reduce)
    fi

    if [[ -n "$GUIDED_DECODING_BACKEND" ]]; then
        if vllm_arg_supported "--guided-decoding-backend"; then
            SERVER_CMD+=(--guided-decoding-backend "$GUIDED_DECODING_BACKEND")
        else
            echo "warning=GUIDED_DECODING_BACKEND=$GUIDED_DECODING_BACKEND requested, but this vLLM does not support --guided-decoding-backend; skipping it."
        fi
    fi

    if [[ -n "$MAX_MODEL_LEN" ]]; then
        SERVER_CMD+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ -n "$DTYPE" ]]; then
        SERVER_CMD+=(--dtype "$DTYPE")
    fi
    if [[ -n "$REASONING_PARSER" ]]; then
        if vllm_arg_supported "--reasoning-parser"; then
            SERVER_CMD+=(--reasoning-parser "$REASONING_PARSER")
        else
            echo "warning=REASONING_PARSER=$REASONING_PARSER requested, but this vLLM does not support --reasoning-parser; skipping it."
        fi
    fi
    if [[ -n "$QUANTIZATION" ]]; then
        if vllm_arg_supported "--quantization"; then
            SERVER_CMD+=(--quantization "$QUANTIZATION")
        else
            echo "warning=QUANTIZATION=$QUANTIZATION requested, but this vLLM does not support --quantization; skipping it."
        fi
    fi
    if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
        SERVER_CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi
    if [[ -n "$MAX_NUM_SEQS" ]]; then
        SERVER_CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
    fi
    if [[ "$ENFORCE_EAGER" == "1" ]]; then
        SERVER_CMD+=(--enforce-eager)
    fi
    if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
        SERVER_CMD+=(--trust-remote-code)
    fi
}

pid_is_alive() {
    local pid="${1:-}"
    [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

load_process_tree_pids() {
    local root_pid="$1"
    "$PYTHON_BIN" - "$root_pid" <<'PY'
import subprocess
import sys
from collections import defaultdict

root = int(sys.argv[1])
output = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
children = defaultdict(list)
for raw_line in output.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    pid_str, ppid_str = line.split(None, 1)
    children[int(ppid_str)].append(int(pid_str))

ordered = []

def walk(pid: int) -> None:
    for child in children.get(pid, []):
        walk(child)
    ordered.append(pid)

walk(root)
for pid in ordered:
    print(pid)
PY
}

load_process_group_id() {
    local root_pid="$1"
    "$PYTHON_BIN" - "$root_pid" <<'PY'
import subprocess
import sys

pid = sys.argv[1]
try:
    output = subprocess.check_output(
        ["ps", "-o", "pgid=", "-p", pid],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except subprocess.CalledProcessError:
    output = ""

print(output)
PY
}

collect_alive_pids() {
    local alive=()
    local pid
    for pid in "$@"; do
        if pid_is_alive "$pid"; then
            alive+=("$pid")
        fi
    done
    printf '%s\n' "${alive[@]}"
}

wait_for_pids_exit() {
    local timeout_s="$1"
    shift
    local start_ts
    start_ts="$(date +%s)"

    while [[ $# -gt 0 ]]; do
        mapfile -t alive_pids < <(collect_alive_pids "$@")
        if [[ "${#alive_pids[@]}" -eq 0 ]]; then
            return 0
        fi

        local now_ts
        now_ts="$(date +%s)"
        if (( now_ts - start_ts >= timeout_s )); then
            printf '%s\n' "${alive_pids[@]}"
            return 1
        fi

        sleep 1
        set -- "${alive_pids[@]}"
    done

    return 0
}

signal_process_group() {
    local signal_name="$1"
    local pgid="$2"
    if [[ -n "$pgid" ]]; then
        kill "-${signal_name}" -- "-${pgid}" >/dev/null 2>&1 || true
    fi
}

signal_pid_list() {
    local signal_name="$1"
    shift
    local pid
    for pid in "$@"; do
        kill "-${signal_name}" "$pid" >/dev/null 2>&1 || true
    done
}

stop_vllm_processes() {
    local root_pid="${1:-}"
    local pgid="${2:-}"
    local tree_pids=()

    if [[ -n "$root_pid" ]] && pid_is_alive "$root_pid"; then
        mapfile -t tree_pids < <(load_process_tree_pids "$root_pid")
    elif [[ -n "$root_pid" ]]; then
        tree_pids=("$root_pid")
    fi

    if [[ "${#tree_pids[@]}" -gt 0 ]]; then
        echo "Stopping vLLM process tree: ${tree_pids[*]}"
    elif [[ -n "$pgid" ]]; then
        echo "Stopping vLLM process group: $pgid"
    fi

    signal_process_group TERM "$pgid"
    signal_pid_list TERM "${tree_pids[@]}"

    local remaining_pids=()
    if ! mapfile -t remaining_pids < <(wait_for_pids_exit 10 "${tree_pids[@]}"); then
        if [[ "${#remaining_pids[@]}" -gt 0 ]]; then
            echo "Force killing remaining vLLM processes: ${remaining_pids[*]}"
            signal_process_group KILL "$pgid"
            signal_pid_list KILL "${remaining_pids[@]}"
            wait_for_pids_exit 5 "${remaining_pids[@]}" >/dev/null || true
        fi
    fi
}

find_stale_vllm_pids() {
    "$PYTHON_BIN" - "$PORT" "$MODEL_PATH" <<'PY'
import os
import subprocess
import sys

port = sys.argv[1]
model_path = sys.argv[2]
self_pid = os.getpid()

try:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,comm=,args="],
        text=True,
        errors="replace",
    )
except Exception:
    raise SystemExit(0)

rows = []
api_server_pids = []
matching_api_server_pids = []
for raw_line in output.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    parts = line.split(None, 2)
    if len(parts) < 3:
        continue
    pid = int(parts[0])
    comm = parts[1]
    args = parts[2]
    if pid == self_pid:
        continue
    rows.append((pid, comm, args))
    if "vllm.entrypoints.openai.api_server" in args:
        api_server_pids.append(pid)
        if (
            f"--port {port}" in args
            or f"--port={port}" in args
            or model_path in args
        ):
            matching_api_server_pids.append(pid)

stale = set(matching_api_server_pids)

# A live API server owns its worker tree. If there is no API server at all,
# leftover EngineCore/Worker_TP processes are stale and will block GPU memory.
if not api_server_pids:
    for pid, comm, args in rows:
        if "VLLM::Worker_TP" in comm or "VLLM::Worker_TP" in args:
            stale.add(pid)
        elif comm == "EngineCore" or "EngineCore" in args:
            stale.add(pid)

for pid in sorted(stale):
    print(pid)
PY
}

stop_stale_vllm_processes() {
    if [[ "$CLEAN_STALE_VLLM" != "1" ]]; then
        return 1
    fi

    local stale_pids=()
    mapfile -t stale_pids < <(find_stale_vllm_pids)
    if [[ "${#stale_pids[@]}" -eq 0 ]]; then
        return 1
    fi

    echo "Stopping stale vLLM processes: ${stale_pids[*]}"
    signal_pid_list TERM "${stale_pids[@]}"

    local remaining_pids=()
    if ! mapfile -t remaining_pids < <(wait_for_pids_exit 10 "${stale_pids[@]}"); then
        if [[ "${#remaining_pids[@]}" -gt 0 ]]; then
            echo "Force killing stale vLLM processes: ${remaining_pids[*]}"
            signal_pid_list KILL "${remaining_pids[@]}"
            wait_for_pids_exit 5 "${remaining_pids[@]}" >/dev/null || true
        fi
    fi

    return 0
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
    local_server_pid=""
    local_server_pgid=""

    if [[ -f "$PID_FILE" ]]; then
        local_server_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    fi
    if [[ -f "$PGID_FILE" ]]; then
        local_server_pgid="$(tr -d '[:space:]' < "$PGID_FILE")"
    fi

    if [[ -z "$local_server_pgid" && -n "$local_server_pid" ]] && pid_is_alive "$local_server_pid"; then
        candidate_pgid="$(load_process_group_id "$local_server_pid" | tr -d '[:space:]')"
        if [[ "$candidate_pgid" == "$local_server_pid" ]]; then
            local_server_pgid="$candidate_pgid"
        fi
    fi

    if [[ -z "$local_server_pid" && -z "$local_server_pgid" ]]; then
        if stop_stale_vllm_processes; then
            echo "Stopped stale vLLM processes"
            exit 0
        fi
        echo "PID/PGID file not found: $PID_FILE / $PGID_FILE"
        if [[ "$CLEAN_STALE_VLLM" == "1" ]]; then
            exit 0
        fi
        exit 1
    fi

    stop_vllm_processes "$local_server_pid" "$local_server_pgid"
    stop_stale_vllm_processes || true

    rm -f "$PID_FILE" "$PGID_FILE"
    if [[ -n "$local_server_pid" || -n "$local_server_pgid" ]]; then
        echo "Stopped vLLM server pid=${local_server_pid:-na} pgid=${local_server_pgid:-na}"
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

if port_in_use; then
    echo "Port already in use: $HOST:$PORT"
    echo "Another server may already be running."
    echo "Check with:"
    echo "  ps -ef | grep vllm.entrypoints.openai.api_server"
    echo "If this project started it with --detach, stop it with:"
    echo "  bash run_server.sh --stop"
    echo "Or choose a different port, for example:"
    echo "  PORT=1226 bash run_server.sh"
    exit 1
fi

stop_stale_vllm_processes || true

mkdir -p "$OUTLINES_CACHE_DIR"
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$PID_FILE")"

cd "$ROOT_DIR"
build_server_cmd
if [[ "$DETACH_MODE" -eq 1 ]]; then
    nohup env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG_MAX_MODEL_LEN" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    setsid "${SERVER_CMD[@]}" > "$LOG_PATH" 2>&1 &

    SERVER_PID="$!"
    SERVER_PGID="$SERVER_PID"
    echo "$SERVER_PID" > "$PID_FILE"
    echo "$SERVER_PGID" > "$PGID_FILE"
    echo "vLLM server started in background"
    echo "pid=$SERVER_PID"
    echo "pgid=$SERVER_PGID"
    echo "model_path=$MODEL_PATH"
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        echo "max_model_len=$MAX_MODEL_LEN"
    fi
    if [[ "$ALLOW_LONG_MAX_MODEL_LEN" == "1" ]]; then
        echo "allow_long_max_model_len=1"
    fi
    echo "url=http://$HOST:$PORT/v1/chat/completions"
    echo "log_path=$LOG_PATH"
    echo "pid_file=$PID_FILE"
    echo "pgid_file=$PGID_FILE"
else
    echo "Starting vLLM server in foreground. Press Ctrl+C to stop."
    echo "model_path=$MODEL_PATH"
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        echo "max_model_len=$MAX_MODEL_LEN"
    fi
    if [[ "$ALLOW_LONG_MAX_MODEL_LEN" == "1" ]]; then
        echo "allow_long_max_model_len=1"
    fi
    echo "url=http://$HOST:$PORT/v1/chat/completions"
    echo "log_path=$LOG_PATH"
    exec > >(tee -a "$LOG_PATH") 2>&1
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    OUTLINES_CACHE_DIR="$OUTLINES_CACHE_DIR" \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG_MAX_MODEL_LEN" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    exec "${SERVER_CMD[@]}"
fi
