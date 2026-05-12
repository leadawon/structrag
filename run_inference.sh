#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# VENV_PATH="${VENV_PATH:-/workspace/venvs/structrag}"

TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/model/Qwen2.5-32B-Instruct}"
LOONG_DIR="${LOONG_DIR:-$ROOT_DIR/loong/Loong}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1225}"
URL="${URL:-$HOST:$PORT}"
API_MODEL_NAME="${API_MODEL_NAME:-Qwen}"
LLM_NAME="${LLM_NAME:-qwen}"
DATASET_NAME="${DATASET_NAME:-loong}"
ROUTER_URL="${ROUTER_URL:-}"
ROUTER_TOKENIZER_PATH="${ROUTER_TOKENIZER_PATH:-}"
ROUTER_API_MODEL_NAME="${ROUTER_API_MODEL_NAME:-}"
ROUTER_LABEL="${ROUTER_LABEL:-}"
ROUTER_DISABLE_GUIDED_DECODING="${ROUTER_DISABLE_GUIDED_DECODING:-0}"

USER_OUTPUT_LABEL="${OUTPUT_PATH_SUFFIX:-}"
RESUME_OUTPUT_PATH_SUFFIX="${RESUME_OUTPUT_PATH_SUFFIX:-}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
AUTO_SCORE="${AUTO_SCORE:-1}"
AUTO_SCORE_FORCE_OVERWRITE="${AUTO_SCORE_FORCE_OVERWRITE:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
FORCE_NEW_RUN="${FORCE_NEW_RUN:-0}"
WORKER_COUNT="${WORKER_COUNT:-8}"
PROCESS_NUM_EVAL="${PROCESS_NUM_EVAL:-20}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen_local_judge.yaml}"
GEN_MODEL_CONFIG="${GEN_MODEL_CONFIG:-qwen2.yaml}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-$LOONG_DIR/config/models}"
INCLUDE_ERROR_OUTPUTS_IN_SCORE="${INCLUDE_ERROR_OUTPUTS_IN_SCORE:-0}"
STRUCTURED_EVAL_PY_ROOT="${STRUCTURED_EVAL_PY_ROOT:-}"
STRUCTURED_EVAL_OUTPUT_PATH="${STRUCTURED_EVAL_OUTPUT_PATH:-}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"
GUIDED_DECODING_BACKEND="${STRUCTRAG_GUIDED_DECODING_BACKEND:-lm-format-enforcer}"
CLIENT_MAX_INPUT_TOKENS="${STRUCTRAG_MAX_INPUT_TOKENS:-}"
SERVER_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
STRUCTRAG_LOGGING="${STRUCTRAG_LOGGING:-0}"
STRUCTRAG_LOGGING_DIR="${STRUCTRAG_LOGGING_DIR:-}"
STRUCTRAG_LOGGING_RUN_ID="${STRUCTRAG_LOGGING_RUN_ID:-}"
RESTART_EVERY="${RESTART_EVERY:-100}"
RESTART_WAIT_TIMEOUT="${RESTART_WAIT_TIMEOUT:-600}"
RESTART_WAIT_INTERVAL="${RESTART_WAIT_INTERVAL:-5}"
SERVER_RESTART_CMD="${SERVER_RESTART_CMD:-}"
RESTART_STRATEGY="${RESTART_STRATEGY:-auto}"
MANAGE_SERVER="${MANAGE_SERVER:-1}"
SERVER_BOOTSTRAP_CMD="${SERVER_BOOTSTRAP_CMD:-}"
SERVER_SHUTDOWN_CMD="${SERVER_SHUTDOWN_CMD:-}"
SERVER_SCRIPT_PATH="${SERVER_SCRIPT_PATH:-$ROOT_DIR/scripts/7b/run_server.sh}"
SERVER_PID_FILE="${SERVER_PID_FILE:-${PID_FILE:-}}"
SERVER_PGID_FILE="${SERVER_PGID_FILE:-${PGID_FILE:-}}"
SERVER_LOG_PATH="${SERVER_LOG_PATH:-${LOG_PATH:-}}"
SERVER_STARTED_BY_INFERENCE=0
LOCAL_MODEL_DIR="${STRUCTRAG_LOCAL_MODEL_DIR:-}"
if [[ -n "$LOCAL_MODEL_DIR" ]]; then
    MANAGE_SERVER=0
fi

RUN_OUTPUT_PATH_SUFFIX=""
RUN_EVAL_RESULTS_DIR=""
RUN_INTERMEDIATE_DIR=""
RUN_METADATA_PATH=""
ACTION_DESCRIPTOR=""
RUN_RESUME_KEY=""
RUN_RESUME_STATUS=""
RUN_IS_RESUMED=0
RUN_FINALIZED=0

usage() {
    cat <<EOF
Usage:
  bash run_inference.sh sample5
  bash run_inference.sh sample10
  bash run_inference.sh sample100
  bash run_inference.sh exper99 --eval_data_path /path/to/loong_set1_balanced99.jsonl
  bash run_inference.sh sample9999
  bash run_inference.sh single <dataset_id>
  bash run_inference.sh worker <worker_id> [extra main.py args]
  bash run_inference.sh all_workers [extra main.py args]
  OUTPUT_PATH_SUFFIX=<existing_suffix> bash run_inference.sh merge

Behavior:
  - Automatically creates a run-specific result folder name with timestamp and settings
  - Automatically resumes the latest incomplete matching run unless FORCE_NEW_RUN=1
  - Automatically runs scoring after inference unless AUTO_SCORE=0
  - Writes run metadata to eval_results/.../run_manifest.json

Environment overrides:
  TOKENIZER_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  LOONG_DIR=$ROOT_DIR/loong/Loong
  URL=127.0.0.1:1225
  API_MODEL_NAME=Qwen
  ROUTER_URL=127.0.0.1:1226
  ROUTER_TOKENIZER_PATH=$ROOT_DIR/model/Qwen2.5-32B-Instruct
  ROUTER_API_MODEL_NAME=Qwen
  ROUTER_LABEL=learned-router
  ROUTER_DISABLE_GUIDED_DECODING=1
  OUTPUT_PATH_SUFFIX=_mylabel
  AUTO_SCORE=1
  AUTO_RESUME=1
  FORCE_NEW_RUN=0
  RESUME_OUTPUT_PATH_SUFFIX=_ts-...
  PROCESS_NUM_EVAL=20
  EVAL_MODEL_CONFIG=qwen_local_judge.yaml
  GEN_MODEL_CONFIG=qwen2.yaml
  INCLUDE_ERROR_OUTPUTS_IN_SCORE=0
  STRUCTURED_EVAL_PY_ROOT=/path/to/LAMBO
  STRUCTRAG_ENABLE_THINKING=0
  STRUCTRAG_MAX_INPUT_TOKENS=65536
    RESTART_EVERY=100
    RESTART_STRATEGY=manual
    RESTART_WAIT_TIMEOUT=600
    RESTART_WAIT_INTERVAL=5
    MANAGE_SERVER=1
    SERVER_SCRIPT_PATH=$ROOT_DIR/scripts/7b/run_server.sh
    SERVER_PID_FILE=$ROOT_DIR/logs/model.pid
    SERVER_LOG_PATH=$ROOT_DIR/logs/model.log
    SERVER_BOOTSTRAP_CMD='bash scripts/7b/run_server.sh --detach'
    SERVER_SHUTDOWN_CMD='bash scripts/7b/run_server.sh --stop'
    SERVER_RESTART_CMD='bash scripts/7b/run_server.sh --stop && bash scripts/7b/run_server.sh --detach'

Examples:
  bash run_inference.sh sample5
  bash run_inference.sh sample100
  bash run_inference.sh sample9999
  ROUTER_URL=127.0.0.1:1226 ROUTER_LABEL=learned-router ROUTER_DISABLE_GUIDED_DECODING=1 bash run_inference.sh sample5
  OUTPUT_PATH_SUFFIX=_ablationA bash run_inference.sh sample5
  FORCE_NEW_RUN=1 bash run_inference.sh sample9999
  RESUME_OUTPUT_PATH_SUFFIX=_ts-20260327T005213Z_act-sample9999_api-qwen_tok-qwen2-5-7b-instruct_gen-qwen2_eval-qwen-local-judge_guide-lm-format-enforcer_url-127-0-0-1-1225 bash run_inference.sh sample9999
  bash run_inference.sh single 13a4a371-6339-4c9d-82cf-fc9ab2bb017d
  AUTO_SCORE=0 bash run_inference.sh worker 0 --limit 1 --no_shuffle
    AUTO_SCORE=0 RESTART_EVERY=100 bash run_inference.sh sample9999
    AUTO_SCORE=0 RESTART_EVERY=100 RESTART_STRATEGY=manual bash run_inference.sh sample9999
    AUTO_SCORE=1 RESTART_EVERY=100 MANAGE_SERVER=1 bash run_inference.sh sample9999
EOF
}

slugify() {
    local input="${1:-}"
    local max_len="${2:-32}"
    local lowered
    lowered="$(printf '%s' "$input" | tr '[:upper:]' '[:lower:]')"
    local cleaned
    cleaned="$(printf '%s' "$lowered" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
    if [[ -z "$cleaned" ]]; then
        cleaned="na"
    fi
    printf '%s' "${cleaned:0:max_len}"
}

compute_run_suffix() {
    local action_desc="$1"
    local tokenizer_name
    tokenizer_name="$(basename "$TOKENIZER_PATH")"
    local gen_cfg_name
    gen_cfg_name="$(basename "$GEN_MODEL_CONFIG" .yaml)"
    local eval_cfg_name
    eval_cfg_name="$(basename "$EVAL_MODEL_CONFIG" .yaml)"
    local url_slug
    url_slug="$(slugify "$URL" 20)"
    local api_slug
    api_slug="$(slugify "$API_MODEL_NAME" 20)"
    local tok_slug
    tok_slug="$(slugify "$tokenizer_name" 28)"
    local action_slug
    action_slug="$(slugify "$action_desc" 24)"
    local gen_slug
    gen_slug="$(slugify "$gen_cfg_name" 16)"
    local eval_slug
    eval_slug="$(slugify "$eval_cfg_name" 16)"
    local guide_slug
    guide_slug="$(slugify "$GUIDED_DECODING_BACKEND" 18)"
    local label_slug=""
    local router_slug=""
    if [[ -n "$USER_OUTPUT_LABEL" ]]; then
        label_slug="$(slugify "${USER_OUTPUT_LABEL#_}" 24)"
    fi
    if [[ -n "$ROUTER_LABEL" ]]; then
        router_slug="$(slugify "$ROUTER_LABEL" 24)"
    elif [[ -n "$ROUTER_URL" ]]; then
        router_slug="external"
    fi

    local suffix="_ts-${RUN_TIMESTAMP}_act-${action_slug}_api-${api_slug}_tok-${tok_slug}_gen-${gen_slug}_eval-${eval_slug}_guide-${guide_slug}_url-${url_slug}"
    if [[ -n "$router_slug" ]]; then
        suffix="${suffix}_router-${router_slug}"
    fi
    if [[ -n "$label_slug" ]]; then
        suffix="${suffix}_lbl-${label_slug}"
    fi
    printf '%s' "$suffix"
}

compute_run_resume_key() {
    local action_desc="$1"
    local tokenizer_name
    tokenizer_name="$(basename "$TOKENIZER_PATH")"
    local gen_cfg_name
    gen_cfg_name="$(basename "$GEN_MODEL_CONFIG" .yaml)"
    local eval_cfg_name
    eval_cfg_name="$(basename "$EVAL_MODEL_CONFIG" .yaml)"
    local url_slug
    url_slug="$(slugify "$URL" 20)"
    local api_slug
    api_slug="$(slugify "$API_MODEL_NAME" 20)"
    local tok_slug
    tok_slug="$(slugify "$tokenizer_name" 28)"
    local action_slug
    action_slug="$(slugify "$action_desc" 24)"
    local gen_slug
    gen_slug="$(slugify "$gen_cfg_name" 16)"
    local eval_slug
    eval_slug="$(slugify "$eval_cfg_name" 16)"
    local guide_slug
    guide_slug="$(slugify "$GUIDED_DECODING_BACKEND" 18)"
    local label_slug=""
    local router_slug=""
    if [[ -n "$USER_OUTPUT_LABEL" ]]; then
        label_slug="$(slugify "${USER_OUTPUT_LABEL#_}" 24)"
    fi
    if [[ -n "$ROUTER_LABEL" ]]; then
        router_slug="$(slugify "$ROUTER_LABEL" 24)"
    elif [[ -n "$ROUTER_URL" ]]; then
        router_slug="external"
    fi

    local key="act-${action_slug}_api-${api_slug}_tok-${tok_slug}_gen-${gen_slug}_eval-${eval_slug}_guide-${guide_slug}_url-${url_slug}"
    if [[ -n "$router_slug" ]]; then
        key="${key}_router-${router_slug}"
    fi
    if [[ -n "$label_slug" ]]; then
        key="${key}_lbl-${label_slug}"
    fi
    printf '%s' "$key"
}

extract_timestamp_from_suffix() {
    local suffix="${1:-}"
    if [[ "$suffix" =~ _ts-([^_]+)_act- ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    fi
}

set_run_paths_from_suffix() {
    local suffix="$1"
    RUN_OUTPUT_PATH_SUFFIX="$suffix"
    RUN_EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
    RUN_INTERMEDIATE_DIR="$ROOT_DIR/intermediate_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
    RUN_METADATA_PATH="$RUN_EVAL_RESULTS_DIR/run_manifest.json"
}

load_existing_run_context() {
    local manifest_path="$RUN_METADATA_PATH"
    if [[ ! -f "$manifest_path" ]]; then
        local extracted_ts
        extracted_ts="$(extract_timestamp_from_suffix "$RUN_OUTPUT_PATH_SUFFIX")"
        if [[ -n "$extracted_ts" ]]; then
            RUN_TIMESTAMP="$extracted_ts"
        fi
        return 0
    fi

    local loaded
    loaded="$(python - <<PY
import json
from pathlib import Path

path = Path(r"$manifest_path")
data = json.loads(path.read_text(encoding="utf-8"))
print("\t".join([
    str(data.get("run_timestamp", "")),
    str(data.get("status", "")),
    str(data.get("resume_key", "")),
]))
PY
)"
    local loaded_timestamp loaded_status loaded_key
    IFS=$'\t' read -r loaded_timestamp loaded_status loaded_key <<< "$loaded"
    if [[ -n "$loaded_timestamp" ]]; then
        RUN_TIMESTAMP="$loaded_timestamp"
    fi
    if [[ -n "$loaded_status" ]]; then
        RUN_RESUME_STATUS="$loaded_status"
    fi
    if [[ -n "$loaded_key" ]]; then
        RUN_RESUME_KEY="$loaded_key"
    fi
}

find_matching_resumable_run() {
    local eval_root="$ROOT_DIR/eval_results/$LLM_NAME"
    if [[ ! -d "$eval_root" ]]; then
        return 0
    fi

    python - <<PY
import json
from pathlib import Path

base = Path(r"$eval_root")
resume_key = r"$RUN_RESUME_KEY"
dataset_name = r"$DATASET_NAME"

matches = []
for manifest_path in base.glob(f"{dataset_name}_*/run_manifest.json"):
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    if data.get("resume_key") != resume_key:
        continue
    if data.get("status") == "scored":
        continue

    matches.append(
        (
            manifest_path.stat().st_mtime,
            str(data.get("output_path_suffix", "")),
            str(data.get("run_timestamp", "")),
            str(data.get("status", "")),
        )
    )

if matches:
    _, output_path_suffix, run_timestamp, status = max(matches, key=lambda item: item[0])
    print("\t".join([output_path_suffix, run_timestamp, status]))
PY
}

print_existing_progress() {
    python - <<PY
import json
from pathlib import Path

output_dir = Path(r"$RUN_EVAL_RESULTS_DIR")
worker_count = int(r"$WORKER_COUNT")

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
                    f"resume_warning malformed_jsonl path={path} line={lineno} error={exc}"
                )
    return rows

total_completed = 0
total_errors = 0
for worker_id in range(worker_count):
    completed = load_jsonl_records(output_dir / f"final_output_{worker_id}.jsonl")
    errors = load_jsonl_records(output_dir / f"final_output_error_{worker_id}.jsonl")
    if completed or errors:
        print(
            f"resume_progress worker_id={worker_id} completed={len(completed)} errors={len(errors)}"
        )
    total_completed += len(completed)
    total_errors += len(errors)

print(f"resume_progress total_completed={total_completed} total_errors={total_errors}")
PY
}

prepare_run_paths() {
    RUN_RESUME_KEY="$(compute_run_resume_key "$ACTION_DESCRIPTOR")"
    RUN_IS_RESUMED=0
    RUN_RESUME_STATUS=""

    local resume_suffix=""
    local matched_run=""

    if [[ -n "$RESUME_OUTPUT_PATH_SUFFIX" ]]; then
        resume_suffix="$RESUME_OUTPUT_PATH_SUFFIX"
        RUN_IS_RESUMED=1
        RUN_RESUME_STATUS="explicit"
    elif [[ "$AUTO_RESUME" == "1" && "$FORCE_NEW_RUN" != "1" ]]; then
        matched_run="$(find_matching_resumable_run)"
        if [[ -n "$matched_run" ]]; then
            local matched_suffix matched_timestamp matched_status
            IFS=$'\t' read -r matched_suffix matched_timestamp matched_status <<< "$matched_run"
            resume_suffix="$matched_suffix"
            if [[ -n "$matched_timestamp" ]]; then
                RUN_TIMESTAMP="$matched_timestamp"
            fi
            RUN_IS_RESUMED=1
            RUN_RESUME_STATUS="$matched_status"
        fi
    fi

    if [[ -n "$resume_suffix" ]]; then
        set_run_paths_from_suffix "$resume_suffix"
        load_existing_run_context
    else
        set_run_paths_from_suffix "$(compute_run_suffix "$ACTION_DESCRIPTOR")"
    fi

    mkdir -p "$RUN_EVAL_RESULTS_DIR" "$RUN_INTERMEDIATE_DIR"
    if [[ "$RUN_IS_RESUMED" == "1" ]]; then
        echo "Resuming existing run."
        echo "resume_output_path_suffix=$RUN_OUTPUT_PATH_SUFFIX"
        echo "resume_status=${RUN_RESUME_STATUS:-unknown}"
        print_existing_progress
    else
        echo "Starting new run."
        echo "output_path_suffix=$RUN_OUTPUT_PATH_SUFFIX"
    fi
}

write_run_metadata() {
    local status="$1"
    mkdir -p "$RUN_EVAL_RESULTS_DIR"
    python - <<PY
import json
from pathlib import Path

payload = {
    "status": "$status",
    "run_timestamp": "$RUN_TIMESTAMP",
    "action": "$ACTION_DESCRIPTOR",
    "llm_name": "$LLM_NAME",
    "dataset_name": "$DATASET_NAME",
    "url": "$URL",
    "api_model_name": "$API_MODEL_NAME",
    "tokenizer_path": "$TOKENIZER_PATH",
    "router_url": "$ROUTER_URL",
    "router_api_model_name": "$ROUTER_API_MODEL_NAME",
    "router_tokenizer_path": "$ROUTER_TOKENIZER_PATH",
    "router_label": "$ROUTER_LABEL",
    "router_disable_guided_decoding": "$ROUTER_DISABLE_GUIDED_DECODING",
    "loong_dir": "$LOONG_DIR",
    "output_path_suffix": "$RUN_OUTPUT_PATH_SUFFIX",
    "eval_results_dir": "$RUN_EVAL_RESULTS_DIR",
    "intermediate_results_dir": "$RUN_INTERMEDIATE_DIR",
    "auto_score": "$AUTO_SCORE",
    "worker_count": "$WORKER_COUNT",
    "process_num_eval": "$PROCESS_NUM_EVAL",
    "eval_model_config": "$EVAL_MODEL_CONFIG",
    "gen_model_config": "$GEN_MODEL_CONFIG",
    "include_error_outputs_in_score": "$INCLUDE_ERROR_OUTPUTS_IN_SCORE",
    "structured_eval_py_root": "$STRUCTURED_EVAL_PY_ROOT",
    "structured_eval_output_path": "$STRUCTURED_EVAL_OUTPUT_PATH",
    "structrag_enable_thinking": "$STRUCTRAG_ENABLE_THINKING",
    "model_config_dir": "$MODEL_CONFIG_DIR",
    "guided_decoding_backend": "$GUIDED_DECODING_BACKEND",
    "client_max_input_tokens": "$CLIENT_MAX_INPUT_TOKENS",
    "server_max_model_len": "$SERVER_MAX_MODEL_LEN",
    "logging_enabled": "$STRUCTRAG_LOGGING",
    "logging_dir": "$STRUCTRAG_LOGGING_DIR",
    "logging_run_id": "$STRUCTRAG_LOGGING_RUN_ID",
    "server_pid_file": "$SERVER_PID_FILE",
    "server_pgid_file": "$SERVER_PGID_FILE",
    "server_log_path": "$SERVER_LOG_PATH",
    "resume_key": "$RUN_RESUME_KEY",
    "auto_resume": "$AUTO_RESUME",
    "force_new_run": "$FORCE_NEW_RUN",
    "resume_output_path_suffix": "$RESUME_OUTPUT_PATH_SUFFIX",
    "resumed_from_existing_run": "$RUN_IS_RESUMED",
}

Path(r"$RUN_METADATA_PATH").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
}

run_main() {
    local worker_id="$1"
    shift
    local resolved_router_tokenizer_path="${ROUTER_TOKENIZER_PATH:-$TOKENIZER_PATH}"
    local resolved_router_api_model_name="${ROUTER_API_MODEL_NAME:-$API_MODEL_NAME}"
    local extra_router_args=()
    if [[ -n "$ROUTER_URL" ]]; then
        extra_router_args+=(
            --router_url "$ROUTER_URL"
            --router_tokenizer_path "$resolved_router_tokenizer_path"
            --router_api_model_name "$resolved_router_api_model_name"
        )
    fi
    local eval_data_args=()
    if [[ -n "$EVAL_DATA_PATH" ]]; then
        eval_data_args=(--eval_data_path "$EVAL_DATA_PATH")
    fi
    cd "$ROOT_DIR"
    STRUCTRAG_GUIDED_DECODING_BACKEND="$GUIDED_DECODING_BACKEND" \
    STRUCTRAG_ROUTER_DISABLE_GUIDED_DECODING="$ROUTER_DISABLE_GUIDED_DECODING" \
    STRUCTRAG_ENABLE_THINKING="$STRUCTRAG_ENABLE_THINKING" \
    STRUCTRAG_MAX_INPUT_TOKENS="$CLIENT_MAX_INPUT_TOKENS" \
    STRUCTRAG_LOCAL_MODEL_DIR="$LOCAL_MODEL_DIR" \
    STRUCTRAG_LOCAL_DTYPE="${STRUCTRAG_LOCAL_DTYPE:-}" \
    STRUCTRAG_LOCAL_TRUST_REMOTE_CODE="${STRUCTRAG_LOCAL_TRUST_REMOTE_CODE:-}" \
    STRUCTRAG_LOCAL_MAX_INPUT_TOKENS="${STRUCTRAG_LOCAL_MAX_INPUT_TOKENS:-}" \
    STRUCTRAG_LOGGING="$STRUCTRAG_LOGGING" \
    STRUCTRAG_LOGGING_DIR="$STRUCTRAG_LOGGING_DIR" \
    STRUCTRAG_LOGGING_RUN_ID="$STRUCTRAG_LOGGING_RUN_ID" \
    "${PYTHON_BIN:-python}" main.py \
        --url "$URL" \
        --worker_id "$worker_id" \
        --llm_name "$LLM_NAME" \
        --dataset_name "$DATASET_NAME" \
        --loong_dir "$LOONG_DIR" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --api_model_name "$API_MODEL_NAME" \
        --output_path_suffix "$RUN_OUTPUT_PATH_SUFFIX" \
        "${extra_router_args[@]}" \
        "${eval_data_args[@]}" \
        "$@"
}

check_endpoint_health() {
    local endpoint="$1"
    local label="$2"
    python - "$endpoint" "$label" <<'PY'
import sys
import requests

endpoint = sys.argv[1]
label = sys.argv[2]
health_url = f"http://{endpoint}/health"

try:
    response = requests.get(health_url, timeout=5)
except Exception as exc:
    raise SystemExit(f"{label} server is unreachable: {health_url} ({exc})")

if response.status_code != 200:
    raise SystemExit(
        f"{label} server health check failed: {health_url} "
        f"(status={response.status_code}, body={response.text[:200]!r})"
    )
PY
}

server_started_process_exited() {
    if [[ "$SERVER_STARTED_BY_INFERENCE" != "1" ]]; then
        return 1
    fi
    if [[ -z "$SERVER_PID_FILE" || ! -f "$SERVER_PID_FILE" ]]; then
        return 1
    fi

    local server_pid
    server_pid="$(tr -d '[:space:]' < "$SERVER_PID_FILE" 2>/dev/null || true)"
    if [[ -z "$server_pid" ]]; then
        return 1
    fi
    if kill -0 "$server_pid" >/dev/null 2>&1; then
        return 1
    fi

    return 0
}

print_server_log_tail() {
    if [[ -z "$SERVER_LOG_PATH" || ! -f "$SERVER_LOG_PATH" ]]; then
        return 0
    fi

    echo "server_log_path=$SERVER_LOG_PATH"
    echo "server_log_tail_begin"
    tail -n 120 "$SERVER_LOG_PATH" | tr -d '\000' || true
    echo "server_log_tail_end"
}

ensure_required_servers_alive() {
    if [[ -n "$LOCAL_MODEL_DIR" ]]; then
        return 0
    fi
    if [[ "$MANAGE_SERVER" == "1" ]]; then
        start_main_server_if_needed
    fi
    check_endpoint_health "$URL" "main"
    if [[ -n "$ROUTER_URL" ]]; then
        check_endpoint_health "$ROUTER_URL" "router"
    fi
}

start_main_server_if_needed() {
    if check_endpoint_health "$URL" "main" >/dev/null 2>&1; then
        return 0
    fi

    echo "Main server is down. Starting it from inference..."
    if [[ -n "$SERVER_BOOTSTRAP_CMD" ]]; then
        eval "$SERVER_BOOTSTRAP_CMD"
    elif [[ -f "$SERVER_SCRIPT_PATH" ]]; then
        bash "$SERVER_SCRIPT_PATH" --detach
    else
        echo "No server bootstrap command available. Set SERVER_BOOTSTRAP_CMD or SERVER_SCRIPT_PATH."
        return 1
    fi

    SERVER_STARTED_BY_INFERENCE=1
    wait_for_endpoint_health "$URL" "main" "$RESTART_WAIT_TIMEOUT" "$RESTART_WAIT_INTERVAL"
}

stop_main_server_if_owned() {
    if [[ "$MANAGE_SERVER" != "1" || "$SERVER_STARTED_BY_INFERENCE" != "1" ]]; then
        return 0
    fi

    echo "Stopping main server started by inference..."
    if [[ -n "$SERVER_SHUTDOWN_CMD" ]]; then
        eval "$SERVER_SHUTDOWN_CMD" || true
    elif [[ -f "$SERVER_SCRIPT_PATH" ]]; then
        bash "$SERVER_SCRIPT_PATH" --stop || true
    else
        pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" >/dev/null 2>&1 || true
    fi
    SERVER_STARTED_BY_INFERENCE=0
}

wait_for_endpoint_health() {
    local endpoint="$1"
    local label="$2"
    local timeout_s="$3"
    local interval_s="$4"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if check_endpoint_health "$endpoint" "$label" >/dev/null 2>&1; then
            echo "$label server is healthy: http://$endpoint/health"
            return 0
        fi

        if [[ "$label" == "main" ]] && server_started_process_exited; then
            echo "$label server process exited before health became ready: http://$endpoint/health"
            print_server_log_tail
            return 1
        fi

        local now_ts
        now_ts="$(date +%s)"
        if (( now_ts - start_ts >= timeout_s )); then
            echo "Timed out waiting for $label server health after ${timeout_s}s: http://$endpoint/health"
            print_server_log_tail
            return 1
        fi

        local elapsed_s
        elapsed_s=$((now_ts - start_ts))
        echo "Waiting for $label server health... elapsed=${elapsed_s}s timeout=${timeout_s}s"
        sleep "$interval_s"
    done
}

restart_main_server_and_wait() {
    if [[ -n "$LOCAL_MODEL_DIR" ]]; then
        return 0
    fi
    echo "Preparing server restart before next chunk..."
    if [[ "$RESTART_STRATEGY" == "manual" ]]; then
        echo "Manual restart mode: restart your server in the other terminal now."
        echo "Inference will keep waiting until health is back."
    elif [[ -n "$SERVER_RESTART_CMD" ]]; then
        eval "$SERVER_RESTART_CMD"
    elif [[ -f "$SERVER_SCRIPT_PATH" ]]; then
        # Stop detached instance if present.
        bash "$SERVER_SCRIPT_PATH" --stop || true

        # If a foreground vLLM process is still alive on this port, terminate it.
        pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" >/dev/null 2>&1 || true
        sleep 1

        bash "$SERVER_SCRIPT_PATH" --detach
        SERVER_STARTED_BY_INFERENCE=1
    else
        echo "SERVER_RESTART_CMD is not set and default restart script not found."
        return 1
    fi

    wait_for_endpoint_health "$URL" "main" "$RESTART_WAIT_TIMEOUT" "$RESTART_WAIT_INTERVAL"
    if [[ -n "$ROUTER_URL" ]]; then
        wait_for_endpoint_health "$ROUTER_URL" "router" "$RESTART_WAIT_TIMEOUT" "$RESTART_WAIT_INTERVAL"
    fi
}

worker_sequence() {
    seq 0 "$((WORKER_COUNT - 1))"
}

run_auto_score() {
    if [[ "$AUTO_SCORE" != "1" ]]; then
        echo "AUTO_SCORE=0, skipping scoring."
        return 10
    fi

    echo ""
    echo "Starting automatic scoring..."
    cd "$ROOT_DIR"
    local score_status=0
    FORCE_OVERWRITE="$AUTO_SCORE_FORCE_OVERWRITE" \
    INPUT_LLM_NAME="$LLM_NAME" \
    DATASET_NAME="$DATASET_NAME" \
    OUTPUT_PATH_SUFFIX="$RUN_OUTPUT_PATH_SUFFIX" \
    WORKER_COUNT="$WORKER_COUNT" \
    PROCESS_NUM_EVAL="$PROCESS_NUM_EVAL" \
    EVAL_MODEL_CONFIG="$EVAL_MODEL_CONFIG" \
    GEN_MODEL_CONFIG="$GEN_MODEL_CONFIG" \
    MODEL_CONFIG_DIR="$MODEL_CONFIG_DIR" \
    URL="$URL" \
    API_MODEL_NAME="$API_MODEL_NAME" \
    INCLUDE_ERROR_OUTPUTS_IN_SCORE="$INCLUDE_ERROR_OUTPUTS_IN_SCORE" \
    STRUCTURED_EVAL_PY_ROOT="$STRUCTURED_EVAL_PY_ROOT" \
    STRUCTURED_EVAL_OUTPUT_PATH="$STRUCTURED_EVAL_OUTPUT_PATH" \
    STRUCTRAG_ENABLE_THINKING="$STRUCTRAG_ENABLE_THINKING" \
    LOONG_DIR="$LOONG_DIR" \
    RUN_TIMESTAMP="$RUN_TIMESTAMP" \
    bash "$ROOT_DIR/run_score.sh" || score_status=$?
    return "$score_status"
}

show_run_summary() {
    echo ""
    echo "Run completed."
    echo "run_timestamp=$RUN_TIMESTAMP"
    echo "action=$ACTION_DESCRIPTOR"
    echo "output_path_suffix=$RUN_OUTPUT_PATH_SUFFIX"
    echo "eval_results_dir=$RUN_EVAL_RESULTS_DIR"
    echo "intermediate_results_dir=$RUN_INTERMEDIATE_DIR"
    echo "run_metadata_path=$RUN_METADATA_PATH"
    echo "final_output_path=$RUN_EVAL_RESULTS_DIR/final_output_0.jsonl"
    echo "final_error_path=$RUN_EVAL_RESULTS_DIR/final_output_error_0.jsonl"
    if [[ -n "$ROUTER_URL" ]]; then
        echo "router_url=$ROUTER_URL"
        echo "router_api_model_name=${ROUTER_API_MODEL_NAME:-$API_MODEL_NAME}"
        echo "router_tokenizer_path=${ROUTER_TOKENIZER_PATH:-$TOKENIZER_PATH}"
        echo "router_label=${ROUTER_LABEL:-external}"
    fi
    if [[ -n "$CLIENT_MAX_INPUT_TOKENS" ]]; then
        echo "client_max_input_tokens=$CLIENT_MAX_INPUT_TOKENS"
    fi
    if [[ -n "$SERVER_MAX_MODEL_LEN" ]]; then
        echo "server_max_model_len=$SERVER_MAX_MODEL_LEN"
    fi
    if [[ "$STRUCTRAG_LOGGING" == "1" && -n "$STRUCTRAG_LOGGING_DIR" ]]; then
        echo "logging_dir=$STRUCTRAG_LOGGING_DIR"
        echo "logging_run_id=$STRUCTRAG_LOGGING_RUN_ID"
    fi
    if [[ "$AUTO_SCORE" == "1" ]]; then
        echo "score_log_path=$RUN_EVAL_RESULTS_DIR/score.log"
        echo "score_metadata_path=$RUN_EVAL_RESULTS_DIR/score_manifest.json"
    fi
}

finalize_after_inference() {
    write_run_metadata "inference_completed"
    if run_auto_score; then
        write_run_metadata "scored"
    else
        write_run_metadata "inference_completed"
    fi
    show_run_summary
    RUN_FINALIZED=1
}

cleanup_on_exit() {
    local exit_code=$?
    set +e
    if [[ "$RUN_FINALIZED" != "1" && -n "$RUN_METADATA_PATH" && -n "$RUN_OUTPUT_PATH_SUFFIX" ]]; then
        write_run_metadata "interrupted" >/dev/null 2>&1 || true
    fi
    stop_main_server_if_owned || true
    return 0
}

trap cleanup_on_exit EXIT

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

if ! command -v python >/dev/null 2>&1; then
    echo "Python binary not found: python"
    exit 1
fi

if [[ ! -e "$TOKENIZER_PATH" ]]; then
    echo "Tokenizer path not found: $TOKENIZER_PATH"
    exit 1
fi

_eval_data_path="${EVAL_DATA_PATH:-$LOONG_DIR/data/loong_process.jsonl}"
if [[ ! -e "$_eval_data_path" ]]; then
    echo "Loong dataset not found: $_eval_data_path"
    exit 1
fi

case "$ACTION" in
    sample5)
        ACTION_DESCRIPTOR="sample5"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main 0 --limit 5 --no_shuffle "$@"
        finalize_after_inference
        ;;
    sample10)
        ACTION_DESCRIPTOR="sample10"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main 0 --limit 10 --no_shuffle "$@"
        finalize_after_inference
        ;;
    sample100)
        ACTION_DESCRIPTOR="sample100"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main 0 --limit 100 --no_shuffle "$@"
        finalize_after_inference
        ;;
    exper99)
        ACTION_DESCRIPTOR="exper99"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main 0 --limit 99 --no_shuffle "$@"
        finalize_after_inference
        ;;
    sample9999)
        ACTION_DESCRIPTOR="sample9999"
        prepare_run_paths
        write_run_metadata "running"
        if (( RESTART_EVERY > 0 )); then
            for worker_id in $(worker_sequence); do
                local_offset=0
                while (( local_offset < 200 )); do
                    ensure_required_servers_alive
                    run_main "$worker_id" --start_bias "$local_offset" --limit "$RESTART_EVERY" --no_shuffle "$@"
                    local_offset=$((local_offset + RESTART_EVERY))
                    if (( local_offset < 200 )); then
                        restart_main_server_and_wait
                    fi
                done
            done
        else
            for worker_id in $(worker_sequence); do
                ensure_required_servers_alive
                run_main "$worker_id" --no_shuffle "$@"
            done
        fi
        finalize_after_inference
        ;;
    single)
        DATASET_ID="${1:?dataset_id is required}"
        shift
        ACTION_DESCRIPTOR="single-$(slugify "$DATASET_ID" 16)"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main 0 --only_id "$DATASET_ID" --limit 1 --no_shuffle "$@"
        finalize_after_inference
        ;;
    worker)
        WORKER_ID="${1:?worker_id is required}"
        shift
        ACTION_DESCRIPTOR="worker-${WORKER_ID}"
        prepare_run_paths
        write_run_metadata "running"
        ensure_required_servers_alive
        run_main "$WORKER_ID" "$@"
        finalize_after_inference
        ;;
    all_workers)
        ACTION_DESCRIPTOR="all-workers"
        prepare_run_paths
        write_run_metadata "running"
        for worker_id in $(worker_sequence); do
            ensure_required_servers_alive
            run_main "$worker_id" "$@"
        done
        finalize_after_inference
        ;;
    merge)
        if [[ -z "$USER_OUTPUT_LABEL" ]]; then
            echo "merge action requires OUTPUT_PATH_SUFFIX to point to an existing run."
            exit 1
        fi
        ACTION_DESCRIPTOR="merge-only"
        RUN_OUTPUT_PATH_SUFFIX="$USER_OUTPUT_LABEL"
        RUN_EVAL_RESULTS_DIR="$ROOT_DIR/eval_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
        RUN_INTERMEDIATE_DIR="$ROOT_DIR/intermediate_results/$LLM_NAME/${DATASET_NAME}${RUN_OUTPUT_PATH_SUFFIX}"
        RUN_METADATA_PATH="$RUN_EVAL_RESULTS_DIR/run_manifest.json"
        write_run_metadata "merge_only"
        cd "$ROOT_DIR"
        python do_merge_each_batch.py \
            --llm_name "$LLM_NAME" \
            --dataset_name "$DATASET_NAME" \
            --output_path_suffix "$RUN_OUTPUT_PATH_SUFFIX" \
            --loong_dir "$LOONG_DIR" \
            "$@"
        show_run_summary
        RUN_FINALIZED=1
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown action: $ACTION"
        usage
        exit 1
        ;;
esac
