#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2-72B-Instruct-AWQ}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen2-72B-Instruct-AWQ}"
REVISION="${REVISION:-main}"
MAX_WORKERS="${MAX_WORKERS:-8}"
HF_TRANSFER="${HF_TRANSFER:-0}"
DRY_RUN=0

usage() {
    cat <<EOF
Usage:
  bash scripts/72b_4bit/download_model.sh
  bash scripts/72b_4bit/download_model.sh --dry-run

Downloads the Qwen2-72B-Instruct-AWQ (4-bit) model into StructRAG's model directory.
AWQ quantization allows running 72B on 4 GPUs instead of 8.

Defaults:
  MODEL_ID=$MODEL_ID
  MODEL_DIR=$MODEL_DIR
  REVISION=$REVISION
  PYTHON_BIN=$PYTHON_BIN
  MAX_WORKERS=$MAX_WORKERS
  HF_TRANSFER=$HF_TRANSFER

Examples:
  bash scripts/72b_4bit/download_model.sh
  HF_TOKEN=hf_xxx bash scripts/72b_4bit/download_model.sh
  MODEL_DIR=/path/to/Qwen2-72B-Instruct-AWQ bash scripts/72b_4bit/download_model.sh
EOF
}

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN=/path/to/python or VENV_PATH=/path/to/venv."
    exit 1
fi

mkdir -p "$(dirname "$MODEL_DIR")"

echo "model_id=$MODEL_ID"
echo "model_dir=$MODEL_DIR"
echo "revision=$REVISION"
echo "python_bin=$PYTHON_BIN"
echo "max_workers=$MAX_WORKERS"
echo ""
echo "Disk space near target:"
df -h "$(dirname "$MODEL_DIR")"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run only; no files downloaded."
    exit 0
fi

if [[ "$HF_TRANSFER" == "1" ]]; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

"$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" "$REVISION" "$MAX_WORKERS" <<'PY'
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    raise SystemExit(
        "huggingface_hub is not installed in this Python environment.\n"
        "Install it first, for example:\n"
        f"  {sys.executable} -m pip install -U huggingface_hub"
    ) from exc

model_id, model_dir, revision, max_workers = sys.argv[1:5]
target = Path(model_dir)
target.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
allow_patterns = os.environ.get("ALLOW_PATTERNS")
ignore_patterns = os.environ.get("IGNORE_PATTERNS")

kwargs = {
    "repo_id": model_id,
    "repo_type": "model",
    "revision": revision,
    "local_dir": str(target),
    "max_workers": int(max_workers),
}
if token:
    kwargs["token"] = token
if allow_patterns:
    kwargs["allow_patterns"] = [item.strip() for item in allow_patterns.split(",") if item.strip()]
if ignore_patterns:
    kwargs["ignore_patterns"] = [item.strip() for item in ignore_patterns.split(",") if item.strip()]

snapshot_path = snapshot_download(**kwargs)

config_path = target / "config.json"
index_path = target / "model.safetensors.index.json"
safetensors_files = sorted(target.glob("*.safetensors"))
tokenizer_candidates = [
    target / "tokenizer.json",
    target / "tokenizer.model",
    target / "tokenizer_config.json",
]

missing = []
if not config_path.exists():
    missing.append("config.json")
if not any(path.exists() for path in tokenizer_candidates):
    missing.append("tokenizer files")
if not index_path.exists() and not safetensors_files:
    missing.append("safetensors weights")

print("")
print(f"download_path={snapshot_path}")
print(f"model_dir={target}")
if missing:
    raise SystemExit(f"download_incomplete=expected files were not found: {', '.join(missing)}")
else:
    print("download_status=ok")
PY

echo ""
echo "Done. You can now run:"
echo "  bash scripts/72b_4bit/run_inference_full.sh"
