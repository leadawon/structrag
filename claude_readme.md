# StructRAG 72B Inference — 실행 가이드

> 작성일: 2026-05-05  
> 작성자: Claude (Sonnet 4.6)  
> 실행 환경: A100 80GB PCIe × 2

---

## 환경 구성

### GPU 정보
| GPU | 모델 | VRAM | Bus ID |
|-----|------|------|--------|
| 0 | NVIDIA A100 80GB PCIe | 80 GB | 00000000:65:00.0 |
| 1 | NVIDIA A100 80GB PCIe | 80 GB | 00000000:CA:00.0 |

- Driver Version: 535.183.06
- CUDA Driver 지원 버전: 12.2
- 실제 CUDA Toolkit (nvcc): 11.8 (cuda-11.8)

### 패키지 버전 (최종 확정)

| 패키지 | 버전 |
|--------|------|
| Python | 3.10.13 |
| torch | 2.4.0+cu121 |
| vllm | 0.5.5 |
| transformers | 4.44.2 |
| tokenizers | 0.19.1 |
| outlines | 0.0.44 |

---

## 에러슈팅 기록

### 1. vllm 미설치
- **증상**: `ModuleNotFoundError: No module named 'vllm'`
- **원인**: 기본 시스템 Python에 vllm이 없었음
- **해결**: pip로 직접 설치
```bash
pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.5.5
```
> **주의**: vllm 설치 시 torch가 2.4.0으로 자동 업그레이드되어 cu121 빌드가 설치됨 (정상)

---

### 2. pyairports 모듈 누락 (outlines 의존성 버그)
- **증상**: `ModuleNotFoundError: No module named 'pyairports'`
- **원인**: `outlines` 라이브러리가 PyPI의 `pyairports` 패키지에 의존하는데, PyPI에 올라온 `pyairports==0.0.1`은 실제 모듈 파일이 없는 stub 패키지임
- **해결**: `pyairports.airports.AIRPORT_LIST`를 제공하는 stub 모듈을 직접 생성
```bash
mkdir -p /home/elicer/.local/lib/python3.10/site-packages/pyairports
# airports.py에 최소한의 AIRPORT_LIST 튜플 정의
```
> 참고: outlines==0.0.44로 다운그레이드해도 동일 문제 발생 → stub 생성이 필요

---

### 3. transformers 버전 충돌 (rope_scaling AssertionError)
- **증상**: `AssertionError` at `vllm/config.py:1650: assert "factor" in rope_scaling`
- **원인**: vllm 0.5.5가 설치한 transformers 5.7.0이 `rope_scaling` 형식을 다르게 처리함
- **해결**: transformers를 vllm 0.5.5 호환 버전으로 다운그레이드
```bash
pip3 install "transformers==4.44.2"
```

---

### 4. CustomAllreduce + expandable_segments 충돌
- **증상**: `RuntimeError: Tensors allocated with expandable_segments:True cannot be shared between processes.`
- **원인**: `run_server.sh`에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 설정하는데, vllm의 CustomAllreduce가 이를 지원하지 않음
- **해결**: `DISABLE_CUSTOM_ALL_REDUCE=1` 환경 변수 설정
```bash
DISABLE_CUSTOM_ALL_REDUCE=1 bash scripts/72b/run_inference_full.sh
```

---

### 5. GPU KV 캐시 부족 → AWQ 4bit 양자화로 해결
- **증상**: `ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache`
- **원인**:
  - Qwen2-72B-Instruct bf16: GPU당 ~72GB → KV캐시 여유 없음 → 13K 토큰 한도
  - Loong 평균 문서 길이 ~110K chars (~50K 토큰) → 13K면 심각한 성능 저하
- **최종 해결**: AWQ 4bit 양자화 모델 사용

| 형식 | 총 크기 | GPU당 모델 | KV캐시 여유 | 최대 컨텍스트 |
|------|---------|-----------|------------|-------------|
| bf16 | 144 GB | 72 GB | ~0 GB | ❌ 13K (임시) |
| AWQ 4bit | ~38 GB | ~19 GB | ~57 GB/GPU | ✅ 32K+ |

```bash
# AWQ 모델 다운로드 (약 38GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-72B-Instruct-AWQ',
    local_dir='/home/elicer/structrag/model/Qwen2-72B-Instruct-AWQ')
"
```

> **참고**: fp16과 bf16은 동일한 메모리 사용(2 bytes/param). 양자화가 필요하면 반드시 INT4(AWQ/GPTQ) 사용

---

### 6. loong/Loong 디렉토리 미존재
- **증상**: `FileNotFoundError: Could not find Loong data directory`
- **원인**: `loong/` 폴더에 macOS 메타데이터(`__MACOSX`)만 있고 실제 `Loong/` 디렉토리가 없음
- **해결**: `loong/Loong/data/` 구조를 생성하고 실제 데이터 심볼릭 링크
```bash
mkdir -p /home/elicer/structrag/loong/Loong/data
ln -sf /home/elicer/structrag/all_data/loong_process.jsonl \
       /home/elicer/structrag/loong/Loong/data/loong_process.jsonl
```

---

### 7. LAMBO 없음 (scoring 의존성)
- **원인**: scoring 단계에서 `STRUCTURED_EVAL_PY_ROOT=$(dirname "$ROOT_DIR")/LAMBO` 경로 필요
- **해결**: 공개 저장소에서 클론
```bash
git clone https://github.com/leadawon/LAMBO.git /home/elicer/LAMBO
```

---

## 최종 실행 명령어

```bash
cd /home/elicer/structrag

CUDA_VISIBLE_DEVICES=0,1 \
TENSOR_PARALLEL_SIZE=2 \
TOKENIZER_PATH=/home/elicer/structrag/model/Qwen2-72B-Instruct \
MODEL_DIR=/home/elicer/structrag/model/Qwen2-72B-Instruct \
LOONG_DIR=/home/elicer/structrag/loong/Loong \
DISABLE_CUSTOM_ALL_REDUCE=1 \
GPU_MEMORY_UTILIZATION=0.95 \
MAX_NUM_SEQS=8 \
MAX_MODEL_LEN=13312 \
STRUCTRAG_MAX_INPUT_TOKENS=13312 \
bash scripts/72b/run_inference_full.sh
```

### 주요 환경 변수 설명

| 변수 | 값 | 이유 |
|------|----|------|
| `CUDA_VISIBLE_DEVICES` | `0,1` | A100 2개 사용 |
| `TENSOR_PARALLEL_SIZE` | `2` | GPU 2개로 모델 분산 |
| `TOKENIZER_PATH` | `model/Qwen2-72B-Instruct` | 기본값이 32B 경로이므로 명시 필요 |
| `MODEL_DIR` | `model/Qwen2-72B-Instruct` | 72B 모델 경로 명시 |
| `DISABLE_CUSTOM_ALL_REDUCE` | `1` | expandable_segments 충돌 회피 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | KV 캐시 확보 (기본값 0.9 → 0 블록) |
| `MAX_NUM_SEQS` | `8` | 배치 크기 제한으로 메모리 안정성 확보 |
| `MAX_MODEL_LEN` | `13312` | 2 A100×80GB에서 지원 가능한 최대 컨텍스트 |
| `STRUCTRAG_MAX_INPUT_TOKENS` | `13312` | 클라이언트 입력 토큰 한도와 서버 동기화 |
| `AUTO_SCORE` | `1` (기본값) | 추론 완료 후 자동 채점 실행 |

---

## 실행 결과 요약

- **vLLM 서버**: `http://127.0.0.1:1225` (포트 기본값)
- **GPU blocks**: 2126개 (13312 토큰 컨텍스트 충분히 지원)
- **모델 로딩 시간**: 약 4분 20초 (37개 safetensors 샤드)
- **데이터셋**: `all_data/loong_process.jsonl` (1600개 샘플)
- **워커**: 8개 (워커당 200개 샘플, `--no_shuffle`)
- **출력 경로**: `eval_results/qwen/loong<suffix>/`
- **서버 로그**: `logs/qwen2_72b_vllm.log`
- **추론 로그**: `logs/inference_full.log`

---

## 디렉토리 구조 (실행 전 준비 필요)

```
/home/elicer/
├── structrag/
│   ├── model/
│   │   └── Qwen2-72B-Instruct/      # 72B 모델 (136GB)
│   ├── all_data/
│   │   └── loong_process.jsonl       # 1600 샘플 데이터
│   ├── loong/
│   │   └── Loong/
│   │       └── data/
│   │           └── loong_process.jsonl  # symlink to all_data/
│   └── logs/
│       ├── qwen2_72b_vllm.log
│       └── inference_full.log
└── LAMBO/                             # git clone https://github.com/leadawon/LAMBO.git
```

---

---

## 명령어 for Dawon — 처음부터 다시 재현하는 전체 셋업

> 이 섹션만 보고 위에서 아래로 순서대로 실행하면 현재 실험 세팅을 완전히 재현할 수 있습니다.

### Step 1. Python 패키지 설치

```bash
# PyTorch + CUDA 12.1 빌드
pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# vLLM (설치 시 torch가 2.4.0으로 자동 업그레이드됨 — 정상)
pip3 install vllm==0.5.5

# vLLM이 설치한 transformers가 너무 최신이라 충돌 → 다운그레이드 필수
pip3 install "transformers==4.44.2"
```

### Step 2. pyairports stub 생성 (outlines 의존성 버그 우회)

```bash
mkdir -p /home/elicer/.local/lib/python3.10/site-packages/pyairports

cat > /home/elicer/.local/lib/python3.10/site-packages/pyairports/__init__.py << 'EOF'
EOF

cat > /home/elicer/.local/lib/python3.10/site-packages/pyairports/airports.py << 'EOF'
AIRPORT_LIST = [
    ("John F Kennedy International Airport", "New York", "United States", "JFK", "KJFK", 40.63972, -73.77889),
    ("Los Angeles International Airport", "Los Angeles", "United States", "LAX", "KLAX", 33.94250, -118.40806),
    ("Incheon International Airport", "Seoul", "South Korea", "ICN", "RKSI", 37.46910, 126.45080),
]
EOF
```

### Step 3. AWQ 4bit 모델 다운로드 (~37GB)

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2-72B-Instruct-AWQ',
    local_dir='/home/elicer/structrag/model/Qwen2-72B-Instruct-AWQ'
)
"
```

> 원본 bf16 모델(136GB)도 tokenizer 용도로 필요하면 추가 다운로드.  
> 현재 세팅은 AWQ 모델을 tokenizer로도 사용하므로 bf16은 불필요.

### Step 4. Loong 데이터 디렉토리 구조 생성

```bash
mkdir -p /home/elicer/structrag/loong/Loong/data

ln -sf /home/elicer/structrag/all_data/loong_process.jsonl \
       /home/elicer/structrag/loong/Loong/data/loong_process.jsonl
```

> `all_data/loong_process.jsonl` (1600 샘플, 1.1GB)은 미리 준비되어 있어야 함.

### Step 5. LAMBO 클론 (scoring 프레임워크)

```bash
git clone https://github.com/leadawon/LAMBO.git /home/elicer/LAMBO
```

### Step 6. Loong config/src stub 파일 확인

아래 파일들이 없으면 scoring 단계에서 오류 발생. 이미 repo에 포함되어 있으므로 별도 생성 불필요:

- `loong/Loong/config/models/qwen2.yaml`
- `loong/Loong/config/models/qwen_local_judge.yaml`
- `loong/Loong/src/step3_model_evaluate.py`
- `loong/Loong/src/step4_cal_metric.py`

### Step 7. 추론 실행

```bash
cd /home/elicer/structrag

CUDA_VISIBLE_DEVICES=0,1 \
TENSOR_PARALLEL_SIZE=2 \
MODEL_DIR=/home/elicer/structrag/model/Qwen2-72B-Instruct-AWQ \
TOKENIZER_PATH=/home/elicer/structrag/model/Qwen2-72B-Instruct-AWQ \
QUANTIZATION=awq_marlin \
DTYPE=float16 \
MAX_MODEL_LEN=32768 \
STRUCTRAG_MAX_INPUT_TOKENS=32768 \
GPU_MEMORY_UTILIZATION=0.90 \
DISABLE_CUSTOM_ALL_REDUCE=1 \
MAX_NUM_SEQS=8 \
LOONG_DIR=/home/elicer/structrag/loong/Loong \
EVAL_DATA_PATH=/home/elicer/structrag/all_data/loong_process.jsonl \
STRUCTURED_EVAL_PY_ROOT=/home/elicer/LAMBO \
FORCE_NEW_RUN=1 \
bash scripts/72b/run_inference_full.sh > logs/inference_full.log 2>&1 &

echo "PID: $!"
```

> `FORCE_NEW_RUN=1` — 새로 시작할 때만. 이어서 실행할 때는 제거하면 자동 resume.

### Step 8. 진행 상황 모니터링

```bash
# 실시간 로그
tail -f /home/elicer/structrag/logs/inference_full.log

# 완료된 샘플 수 확인 (OUTDIR은 실제 경로로 교체)
OUTDIR="eval_results/qwen/loong_<timestamp>..."
cat "$OUTDIR"/final_output_[0-9]*.jsonl | wc -l

# vLLM 서버 상태
curl -s http://127.0.0.1:1225/health
```

### 주요 환경 변수 요약

| 변수 | 값 | 이유 |
|------|----|------|
| `CUDA_VISIBLE_DEVICES` | `0,1` | A100 2개 사용 |
| `TENSOR_PARALLEL_SIZE` | `2` | GPU 2개 텐서 병렬 |
| `MODEL_DIR` / `TOKENIZER_PATH` | `model/Qwen2-72B-Instruct-AWQ` | AWQ 4bit 모델 |
| `QUANTIZATION` | `awq_marlin` | AWQ + Marlin 커널 (더 빠름) |
| `DTYPE` | `float16` | AWQ는 bfloat16 미지원 |
| `MAX_MODEL_LEN` | `32768` | Qwen2 네이티브 최대 컨텍스트 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | KV 캐시 충분히 확보 |
| `DISABLE_CUSTOM_ALL_REDUCE` | `1` | expandable_segments 충돌 방지 |
| `MAX_NUM_SEQS` | `8` | 배치 크기 제한 (메모리 안정성) |
| `FORCE_NEW_RUN` | `1` | 새 실험 시작 (resume 시 제거) |

### 트러블슈팅 빠른 참조

| 증상 | 해결책 |
|------|--------|
| `ModuleNotFoundError: vllm` | Step 1 패키지 설치 |
| `ModuleNotFoundError: pyairports` | Step 2 stub 생성 |
| `AssertionError: factor in rope_scaling` | `pip3 install transformers==4.44.2` |
| `expandable_segments cannot be shared` | `DISABLE_CUSTOM_ALL_REDUCE=1` |
| `bfloat16 not supported for awq` | `DTYPE=float16` |
| `Could not find Loong data directory` | Step 4 심볼릭 링크 재생성 |
| GPU 메모리 미해제 (nvidia-smi에 PID 남음) | `sudo fuser /dev/nvidia* \| xargs kill -9` |

---

## 향후 개선 사항

### 32K 풀 컨텍스트 지원
현재 설정으로는 최대 13,312 토큰만 처리 가능합니다.  
Loong 벤치마크는 긴 문서 이해가 중요하므로, 풀 컨텍스트를 위해서는:

1. **AWQ INT4 양자화 모델 사용** (권장)
   - 모델 크기: ~36GB (기존 136GB의 1/4)
   - GPU당 18GB 사용 → KV 캐시에 ~120GB 여유
   - `vllm-awq` 형식 모델 필요

2. **더 많은 GPU 사용**
   - 4 × A100 80GB: 모델 34GB/GPU + KV 캐시 46GB/GPU → 32K 충분

### 스코어링 (AUTO_SCORE=1)
LAMBO 클론 후 자동으로 실행됨. 필요한 Python 패키지는 별도 확인 필요.
