# StructRAG Scripts 실행 가이드

## 디렉토리 구조

| 폴더 | 모델 | GPU 수 | 정밀도 | 데이터셋 |
|------|------|:------:|--------|----------|
| `27b/` | Qwen3.5-27B | 4 | bfloat16 | 99-sample 실험 (thinking 선택 가능) |
| `32b/` | Qwen2.5-32B-Instruct | 4 | bfloat16 | 99-sample 실험 |
| `72b/` | Qwen2-72B-Instruct | 8 | **bfloat16** | **전체 1600-sample 실험** |
| `72b_4bit/` | Qwen2-72B-Instruct-AWQ | 4 | **int4 (AWQ)** | **전체 1600-sample 실험** |
| `72b_16bit/` | Qwen2-72B-Instruct | 8 | **float16** | **전체 1600-sample 실험** |
| `7b/` | Qwen2.5-7B-Instruct | 1–2 | bfloat16 | 소규모 테스트 |
| `router/` | — | — | — | Router 모델 학습 |

> **72b / 72b_4bit / 72b_16bit** 는 전체 Loong 데이터셋(1,600개)을 대상으로 하며,
> 동일 모델이 추론(inference)과 LLM judge 역할을 모두 수행합니다.

---

## 실험 실행 순서

### 공통 흐름

```
① 모델 다운로드          download_model.sh
        ↓
② 전체 실험 실행          run_inference_full.sh   ← 서버 자동 기동·추론·채점 일괄 처리
        ↓
③ (선택) 채점만 재실행    run_score_existing.sh
```

> `run_inference_full.sh` 하나로 아래 과정이 자동으로 진행됩니다:
> vLLM 서버 기동 → 1600개 전체 추론 → 동일 모델로 LLM judge → Loong 메트릭 계산 → 결과 저장

---

## 72b (bfloat16, 8 GPU) — 권장

```bash
# 1. 모델 다운로드 (최초 1회)
bash scripts/72b/download_model.sh

# 2. 전체 실험 실행
bash scripts/72b/run_inference_full.sh

# 2-a. 로깅 포함 실행 (72b_logging/ 에 상세 추적 기록)
bash scripts/72b/run_inference_full.sh --logging

# 3. (선택) 이미 완료된 추론 결과로 채점만 재실행
#    서버가 꺼져 있으면 START_SERVER=1 을 붙여 자동 기동
START_SERVER=1 bash scripts/72b/run_score_existing.sh --latest
```

결과 위치:
- 추론 결과: `eval_results/qwen/loong_<suffix>/final_output_*.jsonl`
- LLM judge: `eval_results/qwen/loong_<suffix>/lambo_v2_llm_judge.json`
- 메트릭:    `eval_results/qwen/loong_<suffix>/score.log`

---

## 72b_4bit (AWQ int4, 4 GPU) — GPU 메모리 절약

AWQ 양자화로 4개 GPU에서 실행 가능합니다. 모델 파일이 72b와 다르므로 별도 다운로드가 필요합니다.

```bash
# 1. AWQ 양자화 모델 다운로드 (model/Qwen2-72B-Instruct-AWQ)
bash scripts/72b_4bit/download_model.sh

# 2. 전체 실험 실행
bash scripts/72b_4bit/run_inference_full.sh

# 2-a. 로깅 포함
bash scripts/72b_4bit/run_inference_full.sh --logging

# 3. (선택) 채점 재실행
START_SERVER=1 bash scripts/72b_4bit/run_score_existing.sh --latest
```

결과 위치:
- 추론 결과: `eval_results/qwen/loong_<suffix>/` (suffix 기본값: `qwen2-72b-4bit-full`)

---

## 72b_16bit (float16, 8 GPU)

`72b`와 동일한 모델 파일을 사용하므로, 이미 `scripts/72b/download_model.sh`를 실행했다면
다운로드를 **건너뛰어도 됩니다**.

```bash
# 1. 모델 다운로드 — scripts/72b/ 에서 이미 받았으면 생략 가능
bash scripts/72b_16bit/download_model.sh

# 2. 전체 실험 실행 (float16)
bash scripts/72b_16bit/run_inference_full.sh

# 2-a. 로깅 포함
bash scripts/72b_16bit/run_inference_full.sh --logging

# 3. (선택) 채점 재실행
START_SERVER=1 bash scripts/72b_16bit/run_score_existing.sh --latest
```

결과 위치:
- 추론 결과: `eval_results/qwen/loong_<suffix>/` (suffix 기본값: `qwen2-72b-fp16-full`)

---

## 27b (bfloat16, 4 GPU) — 99-sample 실험

```bash
# 1. 모델 다운로드 (최초 1회)
bash scripts/27b/download_model.sh

# 2. 99-sample 실험 실행
bash scripts/27b/run_inference_exper99.sh

# 2-a. 로깅 포함
bash scripts/27b/run_inference_exper99.sh --logging

# 3. (선택) 채점 재실행
START_SERVER=1 bash scripts/27b/run_score_existing.sh --latest
```

---

## 유용한 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `FORCE_NEW_RUN=1` | 0 | 이전 실행을 이어받지 않고 새 실행 강제 시작 |
| `WORKER_COUNT` | 8 (72b 계열) | 병렬 워커 수 (각 200개 처리) |
| `MAX_MODEL_LEN` | 32768 | vLLM 최대 컨텍스트 길이 |
| `OUTPUT_PATH_SUFFIX` | 모델별 기본값 | 결과 폴더 이름에 붙는 접미사 |
| `CUDA_VISIBLE_DEVICES` | 모델별 기본값 | 사용할 GPU 번호 |
| `START_SERVER=1` | 0 | `run_score_existing.sh` 실행 시 서버 자동 기동 |
| `AUTO_RESUME=1` | 1 | 중단된 실행 자동 재개 |

---

## 로그 / 결과 파일 위치 요약

```
StructRAG/
├── eval_results/qwen/<dataset>_<suffix>/
│   ├── final_output_0.jsonl        # 워커 0 추론 결과
│   ├── final_output_error_0.jsonl  # 워커 0 에러 샘플
│   ├── run_manifest.json           # 실행 메타데이터
│   ├── structured_eval.json        # EM-style 구조적 평가
│   ├── lambo_v2_llm_judge.json     # LLM judge 결과
│   └── score.log                   # 채점 전체 로그
├── intermediate_results/qwen/<dataset>_<suffix>/
│   └── <sample_id>.jsonl           # 샘플별 중간 결과
├── 72b_logging/                    # --logging 모드 상세 추적
├── 72b_4bit_logging/
└── 72b_16bit_logging/
```
