# 다른 서버에서 Claude가 할 일 (2026-05-13 기준)

이 파일은 **다른 서버**에서 StructRAG 추론을 이어달리기 할 때 Claude가 참고할 지침입니다.

---

## 현재 상황 (서버 1 — 원본 서버)

서버 1은 **순방향 run** (sample 0 → 1599, worker 0 → 7 순서)을 진행 중입니다.

| Worker | 상태 | 완료 수 |
|--------|------|---------|
| 0 | ✅ 완료 | 155/200 (45개 에러) |
| 1 | ✅ 완료 | 79/200 (182개 에러) |
| 2 | 🔄 진행중 | 43/200 (계속 증가) |
| 3~7 | ⏳ 대기 | 0/200 |

- Resume 파일 위치: `eval_results/qwen35-27b-vllm/loong_full_ts-20260511T114337Z_act-all-workers_.../`
- Workers 0~1의 성공 결과(final_output_0.jsonl, final_output_1.jsonl)는 git에 포함되어 있음
- **에러 파일(final_output_error_*.jsonl)은 GitHub 100MB 제한으로 push 불가** → 다른 서버에서는 에러 샘플을 재시도하게 됨 (문제없음)

---

## 이 서버(서버 2)에서 할 일: 역방향 run

서버 1이 순방향(0→1599)을 처리 중이므로, 이 서버는 **역방향 run** (sample 1599 → 0)을 실행합니다.
중간 어딘가에서 두 run이 만나면, 먼저 끝나는 쪽을 중지하면 됩니다.

---

## Step 1 — 환경 세팅

`claude_must_read.md` 를 전체 읽고 순서대로 따라하세요.
주요 내용:
- venv: `/workspace/venvs/structrag`
- 필수 패키지: `torch==2.10.0+cu128`, `vllm==0.17.1`, `flashinfer==0.6.4`, `flashinfer-cubin==0.6.4`
- CUDA 드라이버 12.8 이하인 경우 반드시 `+cu128` 버전 사용
- `run_inference.sh`의 `${PYTHON_BIN:-python}` 패치는 이미 적용되어 있음

환경 세팅 완료 후 smoke test:
```bash
cd /workspace/structrag
PYTHON_BIN=/workspace/venvs/structrag/bin/python \
SERVER_PYTHON_BIN=/workspace/venvs/structrag/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_reverse.sh --smoke
```

---

## Step 2 — 역방향 full run 실행

```bash
cd /workspace/structrag
PYTHON_BIN=/workspace/venvs/structrag/bin/python \
SERVER_PYTHON_BIN=/workspace/venvs/structrag/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts_full/qwen3p5_27b_vllm/run_inference_full_reverse.sh
```

- 출력 suffix: `qwen35-think-off-vllm-reverse`
- 결과 디렉토리: `eval_results/qwen35-27b-vllm/loong_full_..._lbl-qwen35-think-off-vllm-reverse/`
- 터미널에서 `reverse_global=1600`, `1599`, `1598`... 순으로 출력되면 정상

---

## Step 3 — 진행 상황 모니터링

서버 1의 `global` 값과 서버 2의 `reverse_global` 값이 만나는 지점을 확인하세요.

```bash
# 완료된 샘플 수 확인
EVAL_DIR=$(ls -td eval_results/qwen35-27b-vllm/loong_full_ts-*lbl-qwen35-think-off-vllm-reverse* 2>/dev/null | head -1)
wc -l "$EVAL_DIR"/final_output_*.jsonl 2>/dev/null | grep -v error | grep total
```

두 서버의 처리 범위가 겹치기 시작하면 (e.g., 서버1 global ≈ 서버2 reverse_global),
이 서버(서버 2)의 프로세스를 Ctrl+C로 중지합니다.

---

## 역방향 run 메커니즘 설명

`main.py --reverse` 동작 방식:
1. 전체 1600개 샘플을 로딩 후 **list(reversed(eval_datas))** 처리
2. 그 후 worker slice 적용: worker 0 → 역순 0~199 (=원래 1400~1599번 샘플)
3. `global_index = worker_base_index + i`
4. `reverse_global_index = 1600 - global_index` 로 출력

즉 `--reverse` 상태에서 worker 0은 원본 데이터의 1599번 샘플부터 시작합니다.

---

## 주의사항

- **CUDA_VISIBLE_DEVICES**: 이 서버의 사용 가능한 GPU 번호로 바꿔야 할 수 있음
  - 확인: `nvidia-smi`로 점유되지 않은 GPU 번호 확인
- **모델 자동 다운로드**: `AUTO_DOWNLOAD_MODEL=1` (기본값)이므로 모델이 없으면 자동 다운로드
  - 약 20분 소요, 디스크 약 60GB 필요
- 서버 1의 순방향 결과 파일(final_output_0.jsonl 등)은 이 서버에도 git clone으로 받아짐
  - 역방향 run과는 별개 디렉토리(suffix가 다름)이므로 충돌 없음
