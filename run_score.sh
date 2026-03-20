#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
LOONG_DIR="${LOONG_DIR:-$ROOT_DIR/loong/Loong}"

INPUT_LLM_NAME="${INPUT_LLM_NAME:-qwen}"
DATASET_NAME="${DATASET_NAME:-loong}"
OUTPUT_PATH_SUFFIX="${OUTPUT_PATH_SUFFIX:-${1:-}}"
WORKER_COUNT="${WORKER_COUNT:-8}"
PROCESS_NUM_EVAL="${PROCESS_NUM_EVAL:-20}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$LOONG_DIR/config/models}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-}"

FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

safe_suffix="${OUTPUT_PATH_SUFFIX//\//_}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-${INPUT_LLM_NAME}${safe_suffix}}"
EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$INPUT_LLM_NAME/${DATASET_NAME}${OUTPUT_PATH_SUFFIX}"
LOONG_OUTPUT_DIR="$LOONG_DIR/output/$OUTPUT_MODEL_NAME"
GENERATE_OUTPUT_PATH="$LOONG_OUTPUT_DIR/loong_generate.jsonl"
EVALUATE_OUTPUT_PATH="$LOONG_OUTPUT_DIR/loong_evaluate.jsonl"
SCORE_LOG_PATH="${SCORE_LOG_PATH:-$EVAL_RESULTS_DIR/score.log}"
SCORE_METADATA_PATH="${SCORE_METADATA_PATH:-$EVAL_RESULTS_DIR/score_manifest.json}"

usage() {
    cat <<EOF
Usage:
  bash run_score.sh
  bash run_score.sh _sample5

Environment overrides:
  VENV_PATH=/workspace/venvs/structrag
  INPUT_LLM_NAME=qwen
  DATASET_NAME=loong
  OUTPUT_PATH_SUFFIX=_sample5
  OUTPUT_MODEL_NAME=qwen_sample5
  WORKER_COUNT=8
  PROCESS_NUM_EVAL=20
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  FORCE_OVERWRITE=1

What it does:
  1. Merge eval_results/<llm>/<dataset><suffix>/final_output_*.jsonl
  2. Write merged generations to loong/Loong/output/<output_model_name>/loong_generate.jsonl
  3. Run Loong step3_model_evaluate.py
  4. Run Loong step4_cal_metric.py
  5. Save scoring logs to eval_results/.../score.log
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -d "$LOONG_DIR" ]]; then
    echo "Loong directory not found: $LOONG_DIR"
    exit 1
fi

if [[ ! -d "$EVAL_RESULTS_DIR" ]]; then
    echo "Eval results directory not found: $EVAL_RESULTS_DIR"
    exit 1
fi

if [[ "$FORCE_OVERWRITE" != "1" ]]; then
    if [[ -e "$GENERATE_OUTPUT_PATH" || -e "$EVALUATE_OUTPUT_PATH" ]]; then
        echo "Output already exists under: $LOONG_OUTPUT_DIR"
        echo "Set FORCE_OVERWRITE=1 to overwrite."
        exit 1
    fi
else
    rm -f "$GENERATE_OUTPUT_PATH" "$EVALUATE_OUTPUT_PATH"
fi

mkdir -p "$LOONG_OUTPUT_DIR"
mkdir -p "$EVAL_RESULTS_DIR"

exec > >(tee -a "$SCORE_LOG_PATH") 2>&1

echo "Merging generations..."
echo "eval_results_dir=$EVAL_RESULTS_DIR"
echo "output_model_name=$OUTPUT_MODEL_NAME"
echo "generate_output_path=$GENERATE_OUTPUT_PATH"
echo "evaluate_output_path=$EVALUATE_OUTPUT_PATH"
echo "score_log_path=$SCORE_LOG_PATH"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

eval_results_dir = Path(r"$EVAL_RESULTS_DIR")
generate_output_path = Path(r"$GENERATE_OUTPUT_PATH")
worker_count = int(r"$WORKER_COUNT")

total_datas = []
for worker_id in range(worker_count):
    worker_output_path = eval_results_dir / f"final_output_{worker_id}.jsonl"
    if worker_output_path.exists():
        worker_datas = [json.loads(line) for line in open(worker_output_path)]
        print(f"worker_id={worker_id}, len={len(worker_datas)}")
        total_datas.extend(worker_datas)

if not total_datas:
    raise SystemExit(f"No merged results found in {eval_results_dir}")

with open(generate_output_path, "w", encoding="utf-8") as fw:
    for data in total_datas:
        fw.write(json.dumps(data, ensure_ascii=False) + "\\n")

used_times = [data.get("used_time") for data in total_datas if isinstance(data.get("used_time"), (int, float))]
print(f"merged_samples={len(total_datas)}")
if used_times:
    print(f"avg_used_time_min={sum(used_times)/len(used_times):.4f}")
PY

echo ""
echo "Running evaluator..."
cd "$LOONG_DIR/src"
"$PYTHON_BIN" step3_model_evaluate.py \
    --models "$GEN_MODEL_CONFIG" \
    --eval_model "$EVAL_MODEL_CONFIG" \
    --output_path "$GENERATE_OUTPUT_PATH" \
    --evaluate_output_path "$EVALUATE_OUTPUT_PATH" \
    --model_config_dir "$MODEL_CONFIG_DIR" \
    --process_num_eval "$PROCESS_NUM_EVAL"

echo ""
echo "Checking evaluator outputs..."
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

evaluate_output_path = Path(r"$EVALUATE_OUTPUT_PATH")
if not evaluate_output_path.exists():
    raise SystemExit(f"Evaluator output not found: {evaluate_output_path}")

rows = [json.loads(line) for line in evaluate_output_path.open(encoding="utf-8")]
nonempty = [row for row in rows if str(row.get("eval_response", "")).strip()]
scored = [row for row in nonempty if "[[" in str(row.get("eval_response", ""))]

print(f"evaluate_rows={len(rows)}")
print(f"evaluate_nonempty={len(nonempty)}")
print(f"evaluate_scored={len(scored)}")

if not scored:
    raise SystemExit(
        "Evaluator produced 0 valid scored responses. "
        "Check EVAL_MODEL_CONFIG / judge server connectivity before metric calculation."
    )
PY

echo ""
echo "Calculating metrics..."
"$PYTHON_BIN" step4_cal_metric.py \
    --models "$GEN_MODEL_CONFIG" \
    --eval_model "$EVAL_MODEL_CONFIG" \
    --output_path "$GENERATE_OUTPUT_PATH" \
    --evaluate_output_path "$EVALUATE_OUTPUT_PATH" \
    --model_config_dir "$MODEL_CONFIG_DIR" \
    --process_num_eval "$PROCESS_NUM_EVAL"

echo ""
echo "Done."
echo "generate_output_path=$GENERATE_OUTPUT_PATH"
echo "evaluate_output_path=$EVALUATE_OUTPUT_PATH"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = {
    "run_timestamp": "$RUN_TIMESTAMP",
    "input_llm_name": "$INPUT_LLM_NAME",
    "dataset_name": "$DATASET_NAME",
    "output_path_suffix": "$OUTPUT_PATH_SUFFIX",
    "output_model_name": "$OUTPUT_MODEL_NAME",
    "worker_count": "$WORKER_COUNT",
    "process_num_eval": "$PROCESS_NUM_EVAL",
    "eval_model_config": "$EVAL_MODEL_CONFIG",
    "gen_model_config": "$GEN_MODEL_CONFIG",
    "eval_results_dir": "$EVAL_RESULTS_DIR",
    "loong_output_dir": "$LOONG_OUTPUT_DIR",
    "generate_output_path": "$GENERATE_OUTPUT_PATH",
    "evaluate_output_path": "$EVALUATE_OUTPUT_PATH",
    "score_log_path": "$SCORE_LOG_PATH",
}

Path(r"$SCORE_METADATA_PATH").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"score_metadata_path=$SCORE_METADATA_PATH")
PY
