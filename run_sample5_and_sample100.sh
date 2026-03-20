#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

echo "[1/2] Running sample5..."
bash "$ROOT_DIR/run_inference.sh" sample5

echo ""
echo "[2/2] Running sample100..."
bash "$ROOT_DIR/run_inference.sh" sample100

echo ""
echo "All runs completed."
