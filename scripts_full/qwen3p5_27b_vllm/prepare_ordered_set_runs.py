#!/usr/bin/env python3
"""Prepare ordered Loong set runs by reusing successful prior outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SETS = (1, 2, 3, 4)
DEFAULT_CHUNK_SIZE = 200


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_set_list(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SETS)
    out: list[int] = []
    for item in raw.replace(",", " ").split():
        value = int(item)
        if value not in DEFAULT_SETS:
            raise ValueError(f"unsupported set id: {value}")
        out.append(value)
    return sorted(dict.fromkeys(out))


def split_paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning malformed_jsonl path={path} line={lineno} error={exc}", file=sys.stderr)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_error_row(row: dict[str, Any]) -> bool:
    response = row.get("generate_response")
    return (
        row.get("error_message") is not None
        or row.get("error_kind") is not None
        or row.get("used_time") == -100
        or response == "meet error"
    )


def result_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    files: list[Path] = []
    for candidate in sorted(path.glob("final_output_*.jsonl")):
        if candidate.name.startswith("final_output_error_"):
            continue
        files.append(candidate)
    generate_path = path / "loong_generate.jsonl"
    if generate_path.exists():
        files.append(generate_path)
    return files


def discover_default_sources(eval_results_root: Path, llm_name: str) -> dict[str, list[Path]]:
    llm_root = eval_results_root / llm_name
    if not llm_root.exists():
        return {"pulled": [], "local": []}

    pulled = sorted(path for path in llm_root.glob("loong_full_*_lbl-qwen35-think-off-vllm") if path.is_dir())
    local = sorted(path for path in llm_root.glob("loong_full_*_lbl-qwen35-think-off-vllm-re*") if path.is_dir())
    for path in sorted(llm_root.glob("loong_full_*reverse*")):
        if path.is_dir() and path not in local:
            local.append(path)
    for pattern in ("loong_set*_server*", "loong_set*server*_tail*", "loong_set*_tail*_server*"):
        for path in sorted(llm_root.glob(pattern)):
            if path.is_dir() and path not in local:
                local.append(path)
    return {"pulled": pulled, "local": local}


def read_full_index(full_data_path: Path) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
    id_meta: dict[str, dict[str, Any]] = {}
    ids_by_set: dict[int, list[str]] = defaultdict(list)
    with full_data_path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            row = json.loads(line)
            data_id = str(row["id"])
            set_id = int(row["set"])
            id_meta[data_id] = {
                "order": index,
                "set": set_id,
                "length": row.get("length"),
            }
            ids_by_set[set_id].append(data_id)
    return id_meta, ids_by_set


def load_source_rows(
    sources: dict[str, list[Path]],
    id_meta: dict[str, dict[str, Any]],
    include_error_results: bool,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for group, paths in sources.items():
        for source_path in paths:
            for jsonl_path in result_jsonl_files(source_path):
                for row in load_jsonl(jsonl_path):
                    data_id = row.get("id")
                    if data_id is None:
                        continue
                    data_id = str(data_id)
                    if data_id not in id_meta:
                        continue
                    if not include_error_results and is_error_row(row):
                        continue
                    set_id = int(row.get("set") or id_meta[data_id]["set"])
                    item = {
                        "id": data_id,
                        "set": set_id,
                        "order": int(id_meta[data_id]["order"]),
                        "source_group": group,
                        "source_path": str(source_path),
                        "jsonl_path": str(jsonl_path),
                        "mtime": jsonl_path.stat().st_mtime if jsonl_path.exists() else 0.0,
                        "row": row,
                    }
                    grouped[group][data_id].append(item)
    return grouped


def pick_latest(items: list[dict[str, Any]]) -> dict[str, Any]:
    return max(items, key=lambda item: (float(item["mtime"]), str(item["jsonl_path"])))


def existing_eval_rows(target_eval_dir: Path) -> tuple[dict[str, dict[str, Any]], int]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    max_worker_id = -1
    if not target_eval_dir.exists():
        return rows_by_id, max_worker_id
    for path in sorted(target_eval_dir.glob("final_output_*.jsonl")):
        if path.name.startswith("final_output_error_"):
            continue
        stem = path.stem.replace("final_output_", "")
        if stem.isdigit():
            max_worker_id = max(max_worker_id, int(stem))
        for row in load_jsonl(path):
            data_id = row.get("id")
            if data_id is None or is_error_row(row):
                continue
            rows_by_id.setdefault(str(data_id), row)
    return rows_by_id, max_worker_id


def create_or_replace_symlink(link_path: Path, target: Path) -> None:
    try:
        if link_path.is_symlink() or not link_path.exists():
            if link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(target)
    except OSError as exc:
        print(f"warning symlink_failed link={link_path} target={target} error={exc}", file=sys.stderr)


def format_suffix(template: str, set_id: int) -> str:
    return template.format(set_id=set_id, set_name=f"set{set_id}")


def load_reserved_external_ids(output_root: Path, set_id: int) -> tuple[set[str], list[str]]:
    reserved_ids: set[str] = set()
    reservation_paths: list[str] = []
    for shard_dir in sorted(output_root.glob(f"set{set_id}_server*")):
        data_path = shard_dir / "data" / "loong_process.jsonl"
        if not data_path.exists():
            continue
        reservation_paths.append(str(data_path))
        for row in load_jsonl(data_path):
            data_id = row.get("id")
            if data_id is not None and int(row.get("set", -1)) == set_id:
                reserved_ids.add(str(data_id))
    return reserved_ids, reservation_paths


def prepare_set(
    *,
    set_id: int,
    args: argparse.Namespace,
    id_meta: dict[str, dict[str, Any]],
    ids_by_set: dict[int, list[str]],
    grouped_source_rows: dict[str, dict[str, list[dict[str, Any]]]],
    source_groups: list[str],
    source_priority: list[str],
) -> dict[str, Any]:
    set_name = f"set{set_id}"
    output_suffix = format_suffix(args.output_suffix_template, set_id)
    dataset_name = format_suffix(args.dataset_name_template, set_id)
    set_dir = args.output_root / set_name
    pending_data_path = set_dir / "data" / "loong_process.jsonl"
    target_eval_dir = args.eval_results_root / args.llm_name / f"{dataset_name}{output_suffix}"

    set_dir.mkdir(parents=True, exist_ok=True)
    (set_dir / "reused").mkdir(parents=True, exist_ok=True)
    pending_data_path.parent.mkdir(parents=True, exist_ok=True)
    target_eval_dir.mkdir(parents=True, exist_ok=True)

    per_group_rows: dict[str, list[dict[str, Any]]] = {}
    per_group_sources: dict[str, Counter[str]] = {}
    for group in source_groups:
        chosen = []
        source_counts: Counter[str] = Counter()
        for data_id, items in grouped_source_rows.get(group, {}).items():
            item = pick_latest(items)
            if int(item["set"]) != set_id:
                continue
            chosen.append(item)
            source_counts[str(item["source_path"])] += 1
        chosen.sort(key=lambda item: item["order"])
        rows = [item["row"] for item in chosen]
        per_group_rows[group] = rows
        per_group_sources[group] = source_counts
        write_jsonl(set_dir / "reused" / group / "final_output.jsonl", rows)
        write_json(
            set_dir / "reused" / group / "manifest.json",
            {
                "source_group": group,
                "set": set_id,
                "count": len(rows),
                "sources": dict(sorted(source_counts.items())),
                "generated_at": utc_now(),
            },
        )

    selected_by_id: dict[str, dict[str, Any]] = {}
    selected_source_group: dict[str, str] = {}
    for group in source_priority:
        for row in per_group_rows.get(group, []):
            data_id = str(row["id"])
            if data_id not in selected_by_id:
                selected_by_id[data_id] = row
                selected_source_group[data_id] = group

    selected_rows = sorted(
        selected_by_id.values(),
        key=lambda row: int(id_meta[str(row["id"])]["order"]),
    )
    write_jsonl(set_dir / "reused" / "final_output.jsonl", selected_rows)

    existing_rows, max_worker_id = existing_eval_rows(target_eval_dir)
    missing_reused_rows = [row for row in selected_rows if str(row["id"]) not in existing_rows]
    append_jsonl(target_eval_dir / "final_output_0.jsonl", missing_reused_rows)
    if missing_reused_rows:
        for row in missing_reused_rows:
            existing_rows[str(row["id"])] = row
        max_worker_id = max(max_worker_id, 0)

    reserved_external_ids, reservation_paths = load_reserved_external_ids(args.output_root, set_id)
    pending_count = 0
    reserved_pending_count = 0
    with args.full_data_path.open(encoding="utf-8") as src, pending_data_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if int(row["set"]) != set_id:
                continue
            data_id = str(row["id"])
            if data_id in existing_rows:
                continue
            if data_id in reserved_external_ids:
                reserved_pending_count += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            pending_count += 1

    full_count = len(ids_by_set.get(set_id, []))
    target_completed_count = len(existing_rows)
    missing_count = max(full_count - target_completed_count, 0)
    worker_count = max(1, math.ceil(pending_count / args.worker_chunk_size)) if pending_count else 0
    score_worker_count = max(max_worker_id + 1, worker_count, 1)

    source_overlap_count = sum(
        1
        for data_id in selected_by_id
        if sum(1 for group in source_groups if any(str(row["id"]) == data_id for row in per_group_rows.get(group, []))) > 1
    )
    selected_counts = Counter(selected_source_group.values())

    manifest = {
        "set": set_id,
        "set_name": set_name,
        "generated_at": utc_now(),
        "full_data_path": str(args.full_data_path),
        "full_count": full_count,
        "dataset_name": dataset_name,
        "output_path_suffix": output_suffix,
        "target_eval_dir": str(target_eval_dir),
        "result_set_dir": str(set_dir),
        "pending_data_path": str(pending_data_path),
        "worker_chunk_size": args.worker_chunk_size,
        "pending_count": pending_count,
        "worker_count": worker_count,
        "score_worker_count": score_worker_count,
        "target_completed_count": target_completed_count,
        "missing_count": missing_count,
        "reserved_external_count": reserved_pending_count,
        "reserved_external_paths": reservation_paths,
        "reused_count": len(selected_rows),
        "missing_reused_appended_to_eval": len(missing_reused_rows),
        "source_overlap_count": source_overlap_count,
        "source_priority": source_priority,
        "selected_reused_by_source": dict(sorted(selected_counts.items())),
        "available_reused_by_source": {
            group: len(rows) for group, rows in sorted(per_group_rows.items())
        },
        "source_dirs": {
            group: [str(path) for path in args.sources[group]] for group in source_groups
        },
    }
    write_json(set_dir / "manifest.json", manifest)
    write_json(
        target_eval_dir / "ordered_set_reuse_manifest.json",
        {
            key: value
            for key, value in manifest.items()
            if key not in {"source_dirs"}
        }
        | {"source_dirs": manifest["source_dirs"]},
    )
    create_or_replace_symlink(set_dir / "eval_results", target_eval_dir)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data-path", type=Path, required=True)
    parser.add_argument("--eval-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--llm-name", default="qwen35-27b-vllm")
    parser.add_argument("--sets", default="1 2 3 4")
    parser.add_argument("--dataset-name-template", default="loong_set{set_id}")
    parser.add_argument("--output-suffix-template", default="_ordered_set{set_id}")
    parser.add_argument("--worker-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--pulled-result-dir", action="append", type=Path, default=[])
    parser.add_argument("--local-result-dir", action="append", type=Path, default=[])
    parser.add_argument("--prefer-source", default="pulled,local")
    parser.add_argument("--include-error-results", action="store_true")
    args = parser.parse_args()

    args.full_data_path = args.full_data_path.expanduser().resolve()
    args.eval_results_root = args.eval_results_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    if not args.full_data_path.exists():
        raise SystemExit(f"full data path not found: {args.full_data_path}")

    env_pulled = split_paths(os.environ.get("PULLED_RESULT_DIRS"))
    env_local = split_paths(os.environ.get("LOCAL_RESULT_DIRS"))
    default_sources = discover_default_sources(args.eval_results_root, args.llm_name)
    sources = {
        "pulled": [path.expanduser().resolve() for path in (args.pulled_result_dir or env_pulled or default_sources["pulled"])],
        "local": [path.expanduser().resolve() for path in (args.local_result_dir or env_local or default_sources["local"])],
    }
    args.sources = sources

    source_groups = [group for group in ("pulled", "local") if sources[group]]
    if not source_groups:
        print("warning no prior result sources found; all set items will be pending", file=sys.stderr)
        source_groups = ["pulled", "local"]

    source_priority = [item.strip() for item in args.prefer_source.split(",") if item.strip()]
    unknown_priority = [item for item in source_priority if item not in {"pulled", "local"}]
    if unknown_priority:
        raise SystemExit(f"unknown source priority: {unknown_priority}")
    for group in ("pulled", "local"):
        if group not in source_priority:
            source_priority.append(group)

    id_meta, ids_by_set = read_full_index(args.full_data_path)
    grouped_source_rows = load_source_rows(sources, id_meta, args.include_error_results)
    set_ids = parse_set_list(args.sets)

    manifests = []
    for set_id in set_ids:
        manifests.append(
            prepare_set(
                set_id=set_id,
                args=args,
                id_meta=id_meta,
                ids_by_set=ids_by_set,
                grouped_source_rows=grouped_source_rows,
                source_groups=source_groups,
                source_priority=source_priority,
            )
        )

    summary = {
        "generated_at": utc_now(),
        "sets": set_ids,
        "llm_name": args.llm_name,
        "full_data_path": str(args.full_data_path),
        "eval_results_root": str(args.eval_results_root),
        "output_root": str(args.output_root),
        "source_dirs": {group: [str(path) for path in paths] for group, paths in sources.items()},
        "source_priority": source_priority,
        "manifests": manifests,
    }
    write_json(args.output_root / "summary.json", summary)

    print("set\tfull\treused\tcompleted\tpending\tworkers\teval_dir")
    for manifest in manifests:
        print(
            "\t".join(
                [
                    str(manifest["set"]),
                    str(manifest["full_count"]),
                    str(manifest["reused_count"]),
                    str(manifest["target_completed_count"]),
                    str(manifest["pending_count"]),
                    str(manifest["worker_count"]),
                    str(manifest["target_eval_dir"]),
                ]
            )
        )
    print(f"summary={args.output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
