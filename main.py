import os
import json
import copy
import time
import traceback
import tqdm
import random
from pathlib import Path
random.seed(1024)
import argparse

from utils.qwenapi import QwenAPI
from utils.trace_logger import build_trace_logger_from_env

from router import Router
from structurizer import Structurizer
from utilizer import Utilizer


def is_fatal_backend_error(error_message):
    if error_message is None:
        return False
    fatal_markers = [
        "Connection refused",
        "Engine loop has died",
        "Max retries exceeded with url: /v1/chat/completions",
        "server is unreachable",
    ]
    return any(marker in error_message for marker in fatal_markers)


def classify_error_kind(error_message):
    if error_message is None:
        return "error"
    lowered = str(error_message).lower()
    oom_markers = [
        "out of memory",
        "cuda oom",
        "cudnn_status_alloc_failed",
        "torch.cuda.oom",
        "memoryerror",
    ]
    if any(marker in lowered for marker in oom_markers):
        return "oom"
    return "error"


def resolve_loong_dir(loong_dir):
    candidates = []
    if loong_dir is not None:
        candidates.append(Path(loong_dir))
    candidates.extend([
        Path("./Loong"),
        Path("./loong/Loong"),
    ])

    for candidate in candidates:
        if (candidate / "data" / "loong_process.jsonl").exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find Loong data directory. searched={searched}")


def load_jsonl_records(path, label):
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed JSONL line from {label}: "
                    f"path={path}, line={lineno}, error={exc}"
                )
    return records

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="qwen")
    parser.add_argument("--dataset_name", type=str, default="loong")
    parser.add_argument("--url", type=str, default="10.32.15.63:1225")
    parser.add_argument("--router_url", type=str, default=None)
    parser.add_argument("--router_tokenizer_path", type=str, default=None)
    parser.add_argument("--router_api_model_name", type=str, default=None)
    parser.add_argument("--worker_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=0)
    parser.add_argument("--start_bias", type=int, default=0) # used to manually skip last time error data
    parser.add_argument("--output_path_suffix", type=str, default="")
    parser.add_argument("--loong_dir", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None, help="Override Loong processed JSONL path")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--api_model_name", type=str, default="Qwen")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items after filtering")
    parser.add_argument("--only_id", type=str, default=None, help="Only process the sample with this dataset id")
    parser.add_argument("--no_shuffle", action="store_true", help="Keep dataset order for debugging")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print('\nstart...')

    loong_dir = resolve_loong_dir(args.loong_dir)
    print(f"resolved_loong_dir: {loong_dir}")

    router_tokenizer_path = args.router_tokenizer_path or args.tokenizer_path
    router_api_model_name = args.router_api_model_name or args.api_model_name
    router_guided_decoding_backend = "__auto__"
    if os.environ.get("STRUCTRAG_ROUTER_DISABLE_GUIDED_DECODING", "0") == "1":
        router_guided_decoding_backend = None

    trace_logger = build_trace_logger_from_env()
    trace_logger.log_run_manifest(
        {
            "args": vars(args),
            "logging_dir": os.environ.get("STRUCTRAG_LOGGING_DIR"),
            "logging_run_id": os.environ.get("STRUCTRAG_LOGGING_RUN_ID"),
            "router_guided_decoding_backend": router_guided_decoding_backend,
            "resolved_loong_dir": str(loong_dir),
        }
    )
    trace_logger.log_run_event("run_started", {"worker_id": args.worker_id})

    main_llm = QwenAPI(
        url=f"http://{args.url}/v1/chat/completions",
        tokenizer_path=args.tokenizer_path,
        model_name=args.api_model_name,
        trace_logger=trace_logger,
    )
    if args.router_url is None:
        router_llm = QwenAPI(
            url=f"http://{args.url}/v1/chat/completions",
            tokenizer_path=router_tokenizer_path,
            model_name=router_api_model_name,
            guided_decoding_backend=router_guided_decoding_backend,
            trace_logger=trace_logger,
        )
    else:
        router_llm = QwenAPI(
            url=f"http://{args.router_url}/v1/chat/completions",
            tokenizer_path=router_tokenizer_path,
            model_name=router_api_model_name,
            guided_decoding_backend=router_guided_decoding_backend,
            trace_logger=trace_logger,
        )

    eval_data_path = Path(args.eval_data_path).expanduser() if args.eval_data_path else loong_dir / "data" / "loong_process.jsonl"
    print(f"eval_data_path: {eval_data_path}")
    eval_datas = [json.loads(l) for l in open(eval_data_path, encoding="utf-8")]
    if not args.no_shuffle and args.only_id is None:
        random.shuffle(eval_datas)

    if args.only_id is not None:
        eval_datas = [data for data in eval_datas if str(data["id"]) == args.only_id]
    else:
        eval_datas = eval_datas[200*args.worker_id+args.start_bias : 200*(args.worker_id+1)]

    if args.limit is not None:
        eval_datas = eval_datas[:args.limit]

    print(f"len eval_datas: {len(eval_datas)}")
    if args.only_id is not None and len(eval_datas) == 0:
        raise ValueError(f"No sample found for only_id={args.only_id}")
    if len(eval_datas) <= 10:
        print("eval_data_ids:", [data["id"] for data in eval_datas])

    intermediate_results_dir = f"./intermediate_results/{args.llm_name}/{args.dataset_name}{args.output_path_suffix}"
    os.makedirs(intermediate_results_dir) if not os.path.exists(intermediate_results_dir) else None

    chunk_kb_path = f"{intermediate_results_dir}/chunk_kb"
    graph_kb_path = f"{intermediate_results_dir}/graph_kb"
    table_kb_path = f"{intermediate_results_dir}/table_kb"
    algorithm_kb_path = f"{intermediate_results_dir}/algorithm_kb"
    catalogue_kb_path = f"{intermediate_results_dir}/catalogue_kb"
    os.makedirs(chunk_kb_path) if not os.path.exists(chunk_kb_path) else None
    os.makedirs(graph_kb_path) if not os.path.exists(graph_kb_path) else None
    os.makedirs(table_kb_path) if not os.path.exists(table_kb_path) else None
    os.makedirs(algorithm_kb_path) if not os.path.exists(algorithm_kb_path) else None
    os.makedirs(catalogue_kb_path) if not os.path.exists(catalogue_kb_path) else None

    output_dir = f"./eval_results/{args.llm_name}/{args.dataset_name}{args.output_path_suffix}"
    os.makedirs(output_dir) if not os.path.exists(output_dir) else None
    fw = open(f"{output_dir}/final_output_{args.worker_id}.jsonl", "a", encoding="utf-8")
    fw_error = open(f"{output_dir}/final_output_error_{args.worker_id}.jsonl", "a", encoding="utf-8")
    existing_data = load_jsonl_records(
        f"{output_dir}/final_output_{args.worker_id}.jsonl",
        label=f"worker_{args.worker_id}_output",
    )
    existing_data_ids = {d["id"] for d in existing_data if "id" in d}
    print(f"existing_completed_for_worker: {len(existing_data_ids)}")

    router = Router(router_llm)
    structurizer = Structurizer(main_llm, chunk_kb_path, graph_kb_path, table_kb_path, algorithm_kb_path, catalogue_kb_path)
    utilizer = Utilizer(main_llm, chunk_kb_path, graph_kb_path, table_kb_path, algorithm_kb_path, catalogue_kb_path)

    worker_base_index = 0
    if args.only_id is None:
        worker_base_index = 200 * args.worker_id + args.start_bias

    for i, data in enumerate(eval_datas): # data: {"instruction": "", "question": "", "docs": "", "prompt_template": "{},{},{}"}
        global_index = worker_base_index + i
        if data["id"] in existing_data_ids:
            print(
                f"################## Skipping local={i}, global={global_index} existing... ##################"
            )
            trace_logger.log_run_event(
                "sample_skipped_existing",
                {
                    "index": i,
                    "global_index": global_index,
                    "data_id": data["id"],
                },
            )
            continue
        print(
            f"################## Processing local={i}, global={global_index} ##################"
        )

        try:
            current_time = time.time()
            fw_intermediate = open(f"{intermediate_results_dir}/{data['id']}.jsonl", "w")

            query = data['prompt_template'].format(instruction=data['instruction'], question=data['question'], docs="......")
            _, titles = structurizer.split_content_and_tile(data['docs'])
            core_content = "The titles of the docs are: " + "\n".join(list(set(titles)))
            trace_logger.start_sample(
                data["id"],
                {
                    "worker_id": args.worker_id,
                    "dataset_name": args.dataset_name,
                    "llm_name": args.llm_name,
                    "level": data.get("level"),
                    "set": data.get("set"),
                    "type": data.get("type"),
                    "instruction": data.get("instruction"),
                    "question": data.get("question"),
                    "prompt_template": data.get("prompt_template"),
                    "doc_titles": titles,
                },
            )
            trace_logger.log_stage(
                data["id"],
                stage="query_and_core_content",
                title="Query and Core Content",
                payload={
                    "query": query,
                    "core_content": core_content,
                },
            )

            # 1. router
            chosen, router_output = router.do_route(query, core_content, data['id'])
            fw_intermediate.write(json.dumps({"query": query, "chosen": chosen, "router_output": router_output}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            trace_logger.log_stage(
                data["id"],
                stage="router",
                title="Router",
                payload={
                    "chosen": chosen,
                    "router_output": router_output,
                },
            )

            # 2. structurizer
            instruction, kb_info = structurizer.construct(query, chosen, data['docs'], data['id'])
            fw_intermediate.write(json.dumps({"instruction": instruction, "kb_info": kb_info}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            trace_logger.log_stage(
                data["id"],
                stage="structurizer",
                title="Structurizer",
                payload={
                    "chosen": chosen,
                    "instruction": instruction,
                    "kb_info": kb_info,
                },
            )

            # 3. utilizer
            subqueries = utilizer.do_decompose(query, kb_info, data['id'])
            fw_intermediate.write(json.dumps({"subqueries": subqueries}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            trace_logger.log_stage(
                data["id"],
                stage="subqueries",
                title="Subqueries",
                payload={"subqueries": subqueries},
            )
            subknowledges = utilizer.do_extract(query, subqueries, chosen, data['id'])
            fw_intermediate.write(json.dumps({"subknowledges": subknowledges}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            trace_logger.log_stage(
                data["id"],
                stage="retrieval",
                title="Retrieved Knowledge",
                payload={
                    "chosen": chosen,
                    "subknowledges": subknowledges,
                },
            )
            answer, _, _ = utilizer.do_merge(query, subqueries, subknowledges, chosen, data['id'])
            fw_intermediate.write(json.dumps({"answer": answer}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            trace_logger.log_stage(
                data["id"],
                stage="generation",
                title="Generation",
                payload={"answer": answer},
            )
            
            used_time = (time.time() - current_time) / 60
            print(f"level:{data['level']},set:{data['set']},type:{data['type']}")
            print(f"used time: {used_time:.2f} min")
            trace_logger.log_stage(
                data["id"],
                stage="completion",
                title="Completion",
                payload={"used_time_min": round(used_time, 4)},
            )

            data['generate_response'] = answer
            data['used_time'] = used_time
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw.flush()
            trace_logger.log_run_event(
                "sample_completed",
                {
                    "index": i,
                    "global_index": global_index,
                    "data_id": data["id"],
                    "used_time_min": round(used_time, 4),
                },
            )

        except Exception as e:
            print(f"(print in main.py) Error: {e}")
            traceback.print_exc()
            data['generate_response'] = "meet error"
            data['used_time'] = -100
            data['error_message'] = str(e)
            data['error_kind'] = classify_error_kind(f"{type(e).__name__} {e} {traceback.format_exc()}")
            fw_error.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw_error.flush()
            trace_logger.log_error(
                data["id"],
                {
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            trace_logger.log_run_event(
                "sample_failed",
                {
                    "index": i,
                    "global_index": global_index,
                    "data_id": data["id"],
                    "error_message": str(e),
                },
            )
            if is_fatal_backend_error(data['error_message']):
                raise SystemExit(f"Aborting worker due to fatal backend error: {data['error_message']}")


    print("all done")
