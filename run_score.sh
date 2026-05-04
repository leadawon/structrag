#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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
URL="${URL:-127.0.0.1:1225}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
if [[ "${STRUCTRAG_ENABLE_THINKING,,}" == "1" || "${STRUCTRAG_ENABLE_THINKING,,}" == "true" || "${STRUCTRAG_ENABLE_THINKING,,}" == "yes" || "${STRUCTRAG_ENABLE_THINKING,,}" == "on" ]]; then
    OPENAI_THINKING_MODULES="${OPENAI_THINKING_MODULES:-}"
else
    OPENAI_THINKING_MODULES=""
fi

FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-0}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-}"
if [[ "$URL" == http://* || "$URL" == https://* ]]; then
    if [[ "$URL" == */v1 ]]; then
        DEFAULT_LAMBO_V2_JUDGE_BASE_URL="$URL"
    else
        DEFAULT_LAMBO_V2_JUDGE_BASE_URL="${URL%/}/v1"
    fi
else
    DEFAULT_LAMBO_V2_JUDGE_BASE_URL="http://$URL/v1"
fi
LAMBO_V2_JUDGE="${LAMBO_V2_JUDGE:-1}"
LAMBO_V2_JUDGE_STRICT="${LAMBO_V2_JUDGE_STRICT:-0}"
LAMBO_V2_JUDGE_PY_ROOT="${LAMBO_V2_JUDGE_PY_ROOT:-$STRUCTURED_EVAL_PY_ROOT}"
LAMBO_V2_JUDGE_BASE_URL="${LAMBO_V2_JUDGE_BASE_URL:-$DEFAULT_LAMBO_V2_JUDGE_BASE_URL}"
LAMBO_V2_JUDGE_MODEL="${LAMBO_V2_JUDGE_MODEL:-$API_MODEL_NAME}"
LAMBO_V2_JUDGE_API_KEY="${LAMBO_V2_JUDGE_API_KEY:-EMPTY}"

safe_suffix="${OUTPUT_PATH_SUFFIX//\//_}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-${INPUT_LLM_NAME}${safe_suffix}}"
EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$INPUT_LLM_NAME/${DATASET_NAME}${OUTPUT_PATH_SUFFIX}"
LOONG_OUTPUT_DIR="$LOONG_DIR/output/$OUTPUT_MODEL_NAME"
GENERATE_OUTPUT_PATH="$LOONG_OUTPUT_DIR/loong_generate.jsonl"
EVALUATE_OUTPUT_PATH="$LOONG_OUTPUT_DIR/loong_evaluate.jsonl"
SCORE_LOG_PATH="${SCORE_LOG_PATH:-$EVAL_RESULTS_DIR/score.log}"
SCORE_METADATA_PATH="${SCORE_METADATA_PATH:-$EVAL_RESULTS_DIR/score_manifest.json}"
STRUCTURED_EVAL_OUTPUT_PATH="${STRUCTURED_EVAL_OUTPUT_PATH:-$EVAL_RESULTS_DIR/structured_eval.json}"
LAMBO_V2_JUDGE_OUTPUT_PATH="${LAMBO_V2_JUDGE_OUTPUT_PATH:-$EVAL_RESULTS_DIR/lambo_v2_llm_judge.json}"

usage() {
    cat <<EOF
Usage:
  bash run_score.sh
  bash run_score.sh _sample5

Environment overrides:
  PYTHON_BIN=/path/to/python3
  INPUT_LLM_NAME=qwen
  DATASET_NAME=loong
  OUTPUT_PATH_SUFFIX=_sample5
  OUTPUT_MODEL_NAME=qwen_sample5
  WORKER_COUNT=8
  PROCESS_NUM_EVAL=20
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  FORCE_OVERWRITE=1
  INCLUDE_ERROR_OUTPUTS_IN_SCORE=0
  STRUCTURED_EVAL_PY_ROOT=/path/to/LAMBO
  LAMBO_V2_JUDGE=1
  LAMBO_V2_JUDGE_BASE_URL=http://127.0.0.1:1225/v1
  LAMBO_V2_JUDGE_MODEL=Qwen
  STRUCTRAG_ENABLE_THINKING=0

What it does:
  1. Merge eval_results/<llm>/<dataset><suffix>/final_output_*.jsonl
  2. Optionally merge final_output_error_*.jsonl when INCLUDE_ERROR_OUTPUTS_IN_SCORE=1
  3. Write merged generations to loong/Loong/output/<output_model_name>/loong_generate.jsonl
  4. Save EM-style structured metrics to eval_results/.../structured_eval.json
  5. Run LAMBO v2-style LLM judge to eval_results/.../lambo_v2_llm_judge.json
  6. Run Loong step3_model_evaluate.py and step4_cal_metric.py
  7. Save scoring logs to eval_results/.../score.log
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
echo "structrag_enable_thinking=$STRUCTRAG_ENABLE_THINKING"

EVAL_MODEL_CONFIG_FOR_RUN="$EVAL_MODEL_CONFIG"
TMP_MODEL_CONFIG_DIR=""
cleanup_score_tmp() {
    if [[ -n "$TMP_MODEL_CONFIG_DIR" && -d "$TMP_MODEL_CONFIG_DIR" ]]; then
        rm -rf "$TMP_MODEL_CONFIG_DIR"
    fi
}
trap cleanup_score_tmp EXIT

if [[ "$EVAL_MODEL_CONFIG" == "qwen_local_judge.yaml" || "$EVAL_MODEL_CONFIG" == */qwen_local_judge.yaml ]]; then
    EVAL_MODEL_CONFIG_FOR_RUN="$(basename "$EVAL_MODEL_CONFIG")"
    TMP_MODEL_CONFIG_DIR="$(mktemp -d)"
    cp "$MODEL_CONFIG_DIR"/*.yaml "$TMP_MODEL_CONFIG_DIR"/
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

path = Path(r"$TMP_MODEL_CONFIG_DIR") / "qwen_local_judge.yaml"
api_url = r"$LAMBO_V2_JUDGE_BASE_URL"
api_name = r"$API_MODEL_NAME"
enable_thinking = r"$STRUCTRAG_ENABLE_THINKING"

path.write_text(
    "\n".join(
        [
            'type: "vllm"',
            "args:",
            f'  api_name: "{api_name}"',
            '  api_key: "EMPTY"',
            f'  api_url: "{api_url}"',
            "run_args:",
            "  temperature: 0.0",
            f"  enable_thinking: {json.dumps(enable_thinking.strip().lower() in {'1', 'true', 'yes', 'on'})}",
            "",
        ]
    ),
    encoding="utf-8",
)
print(f"judge_model_config_override={path}")
print(f"judge_api_name={api_name}")
print(f"judge_api_url={api_url}")
print(f"judge_enable_thinking={enable_thinking}")
PY
    MODEL_CONFIG_DIR="$TMP_MODEL_CONFIG_DIR"
fi

"$PYTHON_BIN" - <<PY
import json
import importlib
import sys
from pathlib import Path

eval_results_dir = Path(r"$EVAL_RESULTS_DIR")
generate_output_path = Path(r"$GENERATE_OUTPUT_PATH")
structured_eval_output_path = Path(r"$STRUCTURED_EVAL_OUTPUT_PATH")
structured_eval_py_root = Path(r"$STRUCTURED_EVAL_PY_ROOT")
worker_count = int(r"$WORKER_COUNT")
include_error_outputs = r"$INCLUDE_ERROR_OUTPUTS_IN_SCORE" == "1"

def load_jsonl_records(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"warning: skipping malformed JSONL line "
                    f"path={path} line={lineno} error={exc}"
                )
    return rows

total_datas = []
seen_ids = set()
for worker_id in range(worker_count):
    worker_output_path = eval_results_dir / f"final_output_{worker_id}.jsonl"
    if worker_output_path.exists():
        worker_datas = load_jsonl_records(worker_output_path)
        deduped_worker_datas = []
        duplicate_count = 0
        for data in worker_datas:
            data_id = data.get("id")
            if data_id is not None and data_id in seen_ids:
                duplicate_count += 1
                continue
            if data_id is not None:
                seen_ids.add(data_id)
            deduped_worker_datas.append(data)
        print(
            f"worker_id={worker_id}, len={len(deduped_worker_datas)}, "
            f"duplicates_skipped={duplicate_count}"
        )
        total_datas.extend(deduped_worker_datas)

if include_error_outputs:
    for worker_id in range(worker_count):
        worker_error_path = eval_results_dir / f"final_output_error_{worker_id}.jsonl"
        if worker_error_path.exists():
            error_datas = load_jsonl_records(worker_error_path)
            deduped_error_datas = []
            duplicate_count = 0
            for data in error_datas:
                data_id = data.get("id")
                if data_id is not None and data_id in seen_ids:
                    duplicate_count += 1
                    continue
                if data_id is not None:
                    seen_ids.add(data_id)
                data.setdefault("generate_response", "meet error")
                data.setdefault("used_time", -100)
                deduped_error_datas.append(data)
            print(
                f"worker_id={worker_id}, error_len={len(deduped_error_datas)}, "
                f"error_duplicates_skipped={duplicate_count}"
            )
            total_datas.extend(deduped_error_datas)

if not total_datas:
    raise SystemExit(f"No merged results found in {eval_results_dir}")

with open(generate_output_path, "w", encoding="utf-8") as fw:
    for data in total_datas:
        fw.write(json.dumps(data, ensure_ascii=False) + "\\n")

if str(structured_eval_py_root) not in sys.path:
    sys.path.insert(0, str(structured_eval_py_root))

evaluate_predictions = None
structured_eval_module_name = None
last_import_error = None
for module_name in (
    "lambo_v2.eval.structured_eval",
    "lambo.eval.structured_eval",
    "script.anchor.evaluate_structured",
    "dawon.anchor.evaluate_structured",
    "dawonv3.anchor.evaluate_structured",
):
    try:
        module = importlib.import_module(module_name)
        evaluate_predictions = getattr(module, "evaluate_predictions")
        structured_eval_module_name = module_name
        break
    except Exception as exc:
        last_import_error = exc

if evaluate_predictions is None:
    raise SystemExit(
        "Unable to import structured evaluator from "
        f"{structured_eval_py_root}. Last error: {last_import_error}"
    )

decoded_total_datas = []
for data in total_datas:
    decoded = dict(data)
    generate_response = decoded.get("generate_response")
    if isinstance(generate_response, str):
        stripped = generate_response.strip()
        if stripped.startswith(("{", "[")):
            try:
                decoded["generate_response"] = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    decoded_total_datas.append(decoded)

structured_summary = evaluate_predictions(decoded_total_datas)
structured_eval_output_path.parent.mkdir(parents=True, exist_ok=True)
structured_eval_output_path.write_text(
    json.dumps(structured_summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

used_times = [data.get("used_time") for data in total_datas if isinstance(data.get("used_time"), (int, float))]
print(f"merged_samples={len(total_datas)}")
print(f"include_error_outputs_in_score={include_error_outputs}")
print(f"structured_eval_module={structured_eval_module_name}")
print(f"structured_eval_output_path={structured_eval_output_path}")
print(f"structured_exact_match_rate={structured_summary['exact_match_rate']:.4f}")
if used_times:
    print(f"avg_used_time_min={sum(used_times)/len(used_times):.4f}")
PY

if [[ "$LAMBO_V2_JUDGE" == "1" ]]; then
    echo ""
    echo "Running LAMBO v2-style LLM judge..."
    OPENAI_THINKING_MODULES="$OPENAI_THINKING_MODULES" "$PYTHON_BIN" - <<PY
import json
import sys
import traceback
from pathlib import Path

generate_output_path = Path(r"$GENERATE_OUTPUT_PATH")
judge_output_path = Path(r"$LAMBO_V2_JUDGE_OUTPUT_PATH")
lambo_root = Path(r"$LAMBO_V2_JUDGE_PY_ROOT")
strict = r"$LAMBO_V2_JUDGE_STRICT" == "1"

try:
    if str(lambo_root) not in sys.path:
        sys.path.insert(0, str(lambo_root))

    from lambo_v2.backend import OpenAIClient
    from lambo_v2.eval.llm_judge import run_llm_judge

    prediction_rows = [
        json.loads(line)
        for line in generate_output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    client = OpenAIClient(
        api_key=r"$LAMBO_V2_JUDGE_API_KEY",
        model_name=r"$LAMBO_V2_JUDGE_MODEL",
        base_url=r"$LAMBO_V2_JUDGE_BASE_URL",
    )
    judge_out = run_llm_judge(llm=client, prediction_rows=prediction_rows)
    judge_output_path.parent.mkdir(parents=True, exist_ok=True)
    judge_output_path.write_text(
        json.dumps(judge_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = judge_out.get("summary", {})
    print(f"lambo_v2_judge_output_path={judge_output_path}")
    print(f"lambo_v2_judge_total={summary.get('total')}")
    print(f"lambo_v2_judge_scoring_success_rate={summary.get('scoring_success_rate')}")
    print(f"lambo_v2_judge_avg_score={summary.get('avg_score')}")
    print(f"lambo_v2_judge_perfect_rate={summary.get('perfect_rate')}")
    if strict and float(summary.get("scoring_success_rate") or 0.0) <= 0.0:
        raise SystemExit("LAMBO v2 judge produced 0 scored responses")
except Exception as exc:
    payload = {
        "summary": {"error": str(exc)},
        "verdicts": [],
        "traceback": traceback.format_exc(),
    }
    judge_output_path.parent.mkdir(parents=True, exist_ok=True)
    judge_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"lambo_v2_judge_output_path={judge_output_path}")
    print(f"lambo_v2_judge_error={exc}")
    if strict:
        raise
PY
else
    echo ""
    echo "LAMBO_V2_JUDGE=0, skipping LAMBO v2-style LLM judge."
fi

echo ""
echo "Running evaluator..."
cd "$LOONG_DIR/src"
"$PYTHON_BIN" step3_model_evaluate.py \
    --models "$GEN_MODEL_CONFIG" \
    --eval_model "$EVAL_MODEL_CONFIG_FOR_RUN" \
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
    --eval_model "$EVAL_MODEL_CONFIG_FOR_RUN" \
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
    "eval_model_config_for_run": "$EVAL_MODEL_CONFIG_FOR_RUN",
    "gen_model_config": "$GEN_MODEL_CONFIG",
    "url": "$URL",
    "api_model_name": "$API_MODEL_NAME",
    "structrag_enable_thinking": "$STRUCTRAG_ENABLE_THINKING",
    "openai_thinking_modules": "$OPENAI_THINKING_MODULES",
    "include_error_outputs_in_score": "$INCLUDE_ERROR_OUTPUTS_IN_SCORE",
    "structured_eval_py_root": "$STRUCTURED_EVAL_PY_ROOT",
    "structured_eval_output_path": "$STRUCTURED_EVAL_OUTPUT_PATH",
    "lambo_v2_judge": "$LAMBO_V2_JUDGE",
    "lambo_v2_judge_strict": "$LAMBO_V2_JUDGE_STRICT",
    "lambo_v2_judge_py_root": "$LAMBO_V2_JUDGE_PY_ROOT",
    "lambo_v2_judge_base_url": "$LAMBO_V2_JUDGE_BASE_URL",
    "lambo_v2_judge_model": "$LAMBO_V2_JUDGE_MODEL",
    "lambo_v2_judge_output_path": "$LAMBO_V2_JUDGE_OUTPUT_PATH",
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
