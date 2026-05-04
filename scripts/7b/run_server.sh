#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export CUDA_VISIBLE_DEVICES
CUDA_DEVICES="${CUDA_DEVICES:-$CUDA_VISIBLE_DEVICES}"
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  TENSOR_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
fi
export CUDA_DEVICES
export TENSOR_PARALLEL_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN:-0}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/logs/qwen7b_vllm.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/logs/qwen7b_vllm.pid}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage:
  bash scripts/7b/run_server.sh
  bash scripts/7b/run_server.sh --detach
  bash scripts/7b/run_server.sh --stop

Defaults:
  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct
  MAX_MODEL_LEN=32768
  GPU_MEMORY_UTILIZATION=0.92
  MAX_NUM_SEQS=2
  ENFORCE_EAGER=1
  ALLOW_LONG_MAX_MODEL_LEN=0
EOF
    exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
    echo "Model path not found: $MODEL_DIR"
    echo "Example:"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct bash scripts/7b/run_server.sh"
    echo "  MODEL_DIR=$ROOT_DIR/model/Qwen2.5-7B-Instruct bash scripts/7b/run_server.sh --detach"
    exit 1
fi

export MODEL_PATH="$MODEL_DIR"
export MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION
export MAX_NUM_SEQS
export ENFORCE_EAGER
export ALLOW_LONG_MAX_MODEL_LEN
export LOG_PATH
export PID_FILE

cd "$ROOT_DIR"
bash "$ROOT_DIR/run_server.sh" "$@"
