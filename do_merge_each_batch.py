import os
import json
import argparse
from pathlib import Path


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
    raise FileNotFoundError(f"Could not find Loong directory. searched={searched}")


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
                    f"Warning: skipping malformed JSONL line: "
                    f"path={path}, line={lineno}, error={exc}"
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="qwen")
    parser.add_argument("--dataset_name", type=str, default="loong")
    parser.add_argument("--output_path_suffix", type=str, default="")
    parser.add_argument("--git_hash", type=str, default="")
    parser.add_argument("--worker_count", type=int, default=8)
    parser.add_argument("--loong_dir", type=str, default=None)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    loong_dir = resolve_loong_dir(args.loong_dir)
    output_dir = loong_dir / "output" / args.llm_name
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_output_path = output_dir / "loong_generate.jsonl"
    evaluate_output_path = output_dir / "loong_evaluate.jsonl"

    if not args.allow_overwrite and generate_output_path.exists():
        raise ValueError(f"File already exists: {generate_output_path}")
    if not args.allow_overwrite and evaluate_output_path.exists():
        raise ValueError(f"File already exists: {evaluate_output_path}")

    total_datas = []
    seen_ids = set()
    dir_path = Path(f"./eval_results{args.git_hash}/{args.llm_name}/{args.dataset_name}{args.output_path_suffix}")

    for worker_id in range(args.worker_count):
        worker_output_path = dir_path / f"final_output_{worker_id}.jsonl"
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
            total_datas += deduped_worker_datas

    print("len(total_datas)", len(total_datas))

    with open(generate_output_path, "w") as fw:
        for data in total_datas:
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
