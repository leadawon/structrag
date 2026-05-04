#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
JOBS="${JOBS:-$(nproc)}"
ENABLE_CUDA="${ENABLE_CUDA:-1}"

usage() {
    cat <<EOF
Usage:
  bash scripts/router/setup_llama_cpp.sh

Environment overrides:
  LLAMA_CPP_DIR=$ROOT_DIR/llama.cpp
  LLAMA_CPP_REPO=$LLAMA_CPP_REPO
  BUILD_DIR=$LLAMA_CPP_DIR/build
  ENABLE_CUDA=1
  JOBS=$JOBS
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ -z "$CMAKE_BIN" ]]; then
    echo "cmake not found."
    exit 1
fi

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
    git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin
    git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
    -S "$LLAMA_CPP_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
)

if [[ "$ENABLE_CUDA" == "1" ]]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
fi

"$CMAKE_BIN" "${CMAKE_ARGS[@]}"
"$CMAKE_BIN" --build "$BUILD_DIR" --config Release -j "$JOBS" --target llama-server

echo "llama-server built successfully:"
echo "  $BUILD_DIR/bin/llama-server"
