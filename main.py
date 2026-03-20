import os
import json
import copy
import time
import tqdm
import random
from pathlib import Path
random.seed(1024)
import argparse

from utils.qwenapi import QwenAPI

from router import Router
from structurizer import Structurizer
from utilizer import Utilizer


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="qwen")
    parser.add_argument("--dataset_name", type=str, default="loong")
    parser.add_argument("--url", type=str, default="10.32.15.63:1225")
    parser.add_argument("--router_url", type=str, default=None)
    parser.add_argument("--worker_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=0)
    parser.add_argument("--start_bias", type=int, default=0) # used to manually skip last time error data
    parser.add_argument("--output_path_suffix", type=str, default="")
    parser.add_argument("--loong_dir", type=str, default=None)
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

    main_llm = QwenAPI(url=f"http://{args.url}/v1/chat/completions", tokenizer_path=args.tokenizer_path, model_name=args.api_model_name)
    if args.router_url is None:
        router_llm = QwenAPI(url=f"http://{args.url}/v1/chat/completions", tokenizer_path=args.tokenizer_path, model_name=args.api_model_name)
    else:
        router_llm = QwenAPI(url=f"http://{args.router_url}/v1/chat/completions", tokenizer_path=args.tokenizer_path, model_name=args.api_model_name)

    eval_data_path = loong_dir / "data" / "loong_process.jsonl"
    eval_datas = [json.loads(l) for l in open(eval_data_path)]
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
    fw = open(f"{output_dir}/final_output_{args.worker_id}.jsonl", "a")
    fw_error = open(f"{output_dir}/final_output_error_{args.worker_id}.jsonl", "a")
    exiting_data = [json.loads(l) for l in open(f"{output_dir}/final_output_{args.worker_id}.jsonl")]
    exiting_data_ids = [d["id"] for d in exiting_data]    

    router = Router(router_llm)
    structurizer = Structurizer(main_llm, chunk_kb_path, graph_kb_path, table_kb_path, algorithm_kb_path, catalogue_kb_path)
    utilizer = Utilizer(main_llm, chunk_kb_path, graph_kb_path, table_kb_path, algorithm_kb_path, catalogue_kb_path)

    for i, data in enumerate(eval_datas): # data: {"instruction": "", "question": "", "docs": "", "prompt_template": "{},{},{}"}
        if data["id"] in exiting_data_ids:
            print(f"################## Skipping {i}th data existing... ##################")
            continue
        print(f"################## Processing {i}th data... ##################")

        try:
            current_time = time.time()
            fw_intermediate = open(f"{intermediate_results_dir}/{data['id']}.jsonl", "w")

            query = data['prompt_template'].format(instruction=data['instruction'], question=data['question'], docs="......")
            _, titles = structurizer.split_content_and_tile(data['docs'])
            core_content = "The titles of the docs are: " + "\n".join(list(set(titles)))

            # 1. router
            chosen = router.do_route(query, core_content, data['id'])  
            fw_intermediate.write(json.dumps({"query": query, "chosen": chosen}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()

            # 2. structurizer
            instruction, kb_info = structurizer.construct(query, chosen, data['docs'], data['id'])
            fw_intermediate.write(json.dumps({"instruction": instruction, "kb_info": kb_info}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()

            # 3. utilizer
            subqueries = utilizer.do_decompose(query, kb_info, data['id'])
            fw_intermediate.write(json.dumps({"subqueries": subqueries}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            subknowledges = utilizer.do_extract(query, subqueries, chosen, data['id'])
            fw_intermediate.write(json.dumps({"subknowledges": subknowledges}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            answer, _, _ = utilizer.do_merge(query, subqueries, subknowledges, chosen, data['id'])
            fw_intermediate.write(json.dumps({"answer": answer}, ensure_ascii=False) + "\n")
            fw_intermediate.flush()
            
            used_time = (time.time() - current_time) / 60
            print(f"level:{data['level']},set:{data['set']},type:{data['type']}")
            print(f"used time: {used_time:.2f} min")

            data['generate_response'] = answer
            data['used_time'] = used_time
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw.flush()

        except Exception as e:
            print(f"(print in main.py) Error: {e}")
            data['generate_response'] = "meet error"
            data['used_time'] = -100
            fw_error.write(json.dumps(data, ensure_ascii=False) + "\n")
            fw_error.flush()

    print("all done")
