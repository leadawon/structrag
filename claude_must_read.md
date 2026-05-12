# StructRAG Setup — Full Walkthrough (Claude Must Read)

This document records every step taken to get StructRAG running from a clean workspace,
including all errors hit and how they were fixed. Future Claude sessions and other servers
should read this before touching any scripts.

---

## Environment

| Item | Value |
|---|---|
| GPU | 4× NVIDIA RTX A5000 (GPUs 4,5,6,7; GPUs 0-3 occupied by another container) |
| CUDA driver | 570.133.07 → supports CUDA ≤ 12.8 |
| OS Python | /opt/conda/bin/python (do NOT use for inference) |
| Project root | /workspace/structrag |
| Venv | /workspace/venvs/structrag |
| Model | Qwen3.5-27B (downloaded to /workspace/structrag/model/Qwen3.5-27B) |

---

## Step 1 — Create the virtual environment

```bash
python3 -m venv /workspace/venvs/structrag
```

---

## Step 2 — Install huggingface_hub (needed by download_model.sh)

```bash
/workspace/venvs/structrag/bin/pip install -U huggingface_hub
```

**Why**: The download script imports `huggingface_hub` and it was missing from the base venv.

---

## Step 3 — Download the model

```bash
cd /workspace/structrag
bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh --dry-run
# Triggers auto-download (AUTO_DOWNLOAD_MODEL=1)
# Downloads ~20 minutes, 24 files, no HF token needed (unauthenticated OK)
```

Model lands at: `/workspace/structrag/model/Qwen3.5-27B`

---

## Step 4 — Install vLLM with correct CUDA version

**Critical constraint**: CUDA driver 12.8 → max CUDA toolkit 12.8 → use `+cu128` wheels.

**FAILED attempts** (document so you don't repeat):
- `vllm==0.20.2` → pulls `torch 2.11.0+cu130` → needs CUDA 13.0 → driver too old → FAIL
- `torchvision==0.21.0+cu128` → does not exist → use `0.22.0+cu128`

**Working versions** (first vllm release with Qwen3_5ForConditionalGeneration):
```bash
/workspace/venvs/structrag/bin/pip install \
    torch==2.10.0+cu128 torchvision==0.22.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

/workspace/venvs/structrag/bin/pip install vllm==0.17.1
```

If vllm 0.20.x was previously installed, uninstall it first or torch reinstall will conflict:
```bash
/workspace/venvs/structrag/bin/pip uninstall -y vllm
```

---

## Step 5 — Fix flashinfer version mismatch

After installing vllm 0.17.1, flashinfer's cubin wheel version must match exactly:

```bash
# Check what version flashinfer itself is:
/workspace/venvs/structrag/bin/python -c "import flashinfer; print(flashinfer.__version__)"
# Should print 0.6.4

# The cubin wheel must match:
/workspace/venvs/structrag/bin/pip install flashinfer-cubin==0.6.4
```

**Why**: `flashinfer==0.6.4` uses `flashinfer-cubin` for CUDA kernels. If cubin is a different
version (e.g., 0.6.8.post1), you get a version mismatch error at runtime.

**Final verified versions**:
| Package | Version |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.22.0+cu128 |
| vllm | 0.17.1 |
| flashinfer | 0.6.4 |
| flashinfer-cubin | 0.6.4 |
| transformers | 4.57.6 |

---

## Step 6 — Install inference dependencies

```bash
/workspace/venvs/structrag/bin/pip install \
    transformers tokenizers sentencepiece tqdm requests openai lm-format-enforcer
```

**Why**: `transformers` and the others were not in the venv; inference imports them directly.

---

## Step 7 — Critical patch: run_inference.sh must use venv python

**Bug**: `run_inference.sh` called bare `python` which resolved to `/opt/conda/bin/python`
(system Python without any of the packages above). The `PYTHON_BIN` env var was only
forwarded to the vLLM server, not to the inference worker.

**Fix applied** (line ~508 of `run_inference.sh`):
```bash
# OLD:
    STRUCTRAG_LOGGING_RUN_ID="$STRUCTRAG_LOGGING_RUN_ID" \
    python main.py \

# NEW:
    STRUCTRAG_LOGGING_RUN_ID="$STRUCTRAG_LOGGING_RUN_ID" \
    "${PYTHON_BIN:-python}" main.py \
```

This fix is already committed. Do not revert it.

---

## Step 8 — Run a smoke test

```bash
cd /workspace/structrag
PYTHON_BIN=/workspace/venvs/structrag/bin/python \
SERVER_PYTHON_BIN=/workspace/venvs/structrag/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh --smoke
```

Expected output: `inference_completed`, 1–2 samples processed, no errors.

---

## Full forward run (samples 0 → 1599)

```bash
cd /workspace/structrag
PYTHON_BIN=/workspace/venvs/structrag/bin/python \
SERVER_PYTHON_BIN=/workspace/venvs/structrag/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts_full/qwen3p5_27b_vllm/run_inference_full.sh
```

- Dataset: 1600 samples, 8 workers × 200 samples each
- No shuffle (`--no_shuffle`)
- Thinking OFF (`STRUCTRAG_ENABLE_THINKING=0`)
- Output suffix: `qwen35-think-off-vllm`
- Results dir: `result_full/qwen35_27b_vllm/`
- Throughput: ~5–6 seconds/sample, ~2.5 hours total

---

## Full reverse run (samples 1599 → 0) — for second server

Run this on a **second server** in parallel with the forward run so both ends converge
toward the middle. Stop whichever one reaches the meeting point first (monitor
`reverse_global` printed in terminal; stop when it overlaps with the forward run's `global`).

```bash
cd /workspace/structrag
git pull  # get the latest with --reverse support

PYTHON_BIN=/workspace/venvs/structrag/bin/python \
SERVER_PYTHON_BIN=/workspace/venvs/structrag/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_reverse.sh
```

- Output suffix: `qwen35-think-off-vllm-reverse`
- Results dir: `result_full/qwen35_27b_vllm_reverse/`
- Terminal shows: `Processing local=0, global=0, reverse_global=1600` (starts from end)

### How the reverse logic works

`main.py --reverse` reverses `eval_datas` **before** the worker slice.
So worker 0 gets the last 200 samples (indices 1400–1599), worker 7 gets indices 0–199.

Terminal output format:
```
Processing local=0, global=0, reverse_global=1600
Processing local=1, global=1, reverse_global=1599
...
```

`reverse_global = 1600 - global_index` — tells you how many samples remain from the tail end.

---

## GPU assignment

| GPUs | Role |
|---|---|
| 0, 1, 2, 3 | Occupied by another container (invisible via `ps`, different namespace) |
| 4, 5, 6, 7 | This vLLM server (CUDA_VISIBLE_DEVICES=4,5,6,7, tensor_parallel_size=4) |

Do not change `CUDA_VISIBLE_DEVICES` unless you know the other container has released GPUs 0-3.

---

## Checking run progress

```bash
# Tail vLLM server log:
tail -f /workspace/structrag/logs/qwen35_27b_vllm_gpu4567.log

# Count completed samples (all workers combined):
ls /workspace/structrag/eval_results/qwen35-27b-vllm/loong_full_*/final_output_*.jsonl \
  | xargs wc -l 2>/dev/null

# Watch latest intermediate results:
ls -lt /workspace/structrag/intermediate_results/ | head -20
```

---

## Total setup time estimate

From a clean venv (no packages) to first successful smoke run: **~35–45 minutes**
- Model download: ~20 min
- Package installs + troubleshooting CUDA/flashinfer versions: ~15–20 min
- Smoke run itself: ~5 min

On a second server where the model is already downloaded (or shared NFS): **~15–20 min**
(package installs + smoke test only).
