# StructRAG Set1 Server0 Tail Run

## Goal

`run_inference_full_set1_server0.sh` runs only the last 100 unfinished `set1` samples on another server. It writes results to a clearly named server0 folder, separate from the main `set1` run.

## Important: Code Vs Results

`git pull` only brings code changes such as:

```text
scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh
scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh
scripts_full/qwen3p5_27b_vllm/prepare_ordered_set_runs.py
```

`git pull` does not bring generated result folders because `eval_results/` and `result_full/` are ignored runtime outputs.

So the workflow is:

1. Use git pull, or manually copy these scripts, so server0 has the code.
2. Copy the prepared server0 input folder from the main server to server0.
3. Run server0.
4. Copy the server0 result folder back to the main server.

## Main Server: Prepare Server0 Input

On the main server, prepare the exact 100 IDs assigned to server0:

```bash
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh --prepare-only
```

This creates:

```text
result_full/qwen35_27b_vllm_sets/set1_server0/
```

Copy this whole folder to server0 at the same relative path:

```text
/workspace/StructRAG/result_full/qwen35_27b_vllm_sets/set1_server0/
```

This folder contains the exact input file:

```text
result_full/qwen35_27b_vllm_sets/set1_server0/data/loong_process.jsonl
```

Do not rely on git for this folder.

## Main Server: Continue Main Run

After the server0 input folder exists on the main server, you can continue the normal set run:

```bash
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh
```

The normal set runner treats `result_full/qwen35_27b_vllm_sets/set1_server0/data/loong_process.jsonl` as a reserved external shard. Therefore the main server skips those 100 IDs even before the server0 results are copied back.

## Server0: Run The Assigned 100

```bash
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh
```

Useful checks:

```bash
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh --prepare-only
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_set1_server0.sh --dry-run
```

## Output Folders

Server0 inference output:

```text
eval_results/qwen35-27b-vllm/loong_set1_server0_tail100/
```

Server0 prepared input and manifest:

```text
result_full/qwen35_27b_vllm_sets/set1_server0/
```

## Copy Back To Main Server

After server0 finishes, copy this result folder back to the same relative path on the main server:

```text
eval_results/qwen35-27b-vllm/loong_set1_server0_tail100/
```

The full destination should be:

```text
/workspace/StructRAG/eval_results/qwen35-27b-vllm/loong_set1_server0_tail100/
```

Then rerun the normal set wrapper on the main server:

```bash
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_sets.sh
```

The prepare step detects `loong_set*_server*` result folders and reuses those rows. After the copy-back, those 100 IDs become normal completed set1 rows in the main set workflow.
