# PyTorch MPS에서 Qwen3-4B 실행

Apple Silicon의 MPS 백엔드에서 `Qwen/Qwen3-4B`를 실행하고 모델 로딩 시간, 생성 시간,
토큰 수와 MPS 메모리를 기록하는 기준 실험이다.

## 고정한 모델

- Model ID: `Qwen/Qwen3-4B`
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Parameters: 4.0B
- Weight dtype: BF16
- Hugging Face 저장 용량: 약 8.06 GB

모델 카드: <https://huggingface.co/Qwen/Qwen3-4B>

## 1. 다운로드 전 점검

저장소 루트에서 실행한다.

```bash
uv run --no-project --python 3.12 \
  experiments/pytorch-qwen/src/pytorch_qwen/preflight.py
```

이 명령은 모델이나 Python 패키지를 다운로드하지 않는다.

## 2. 환경 설치

모델과 패키지를 위한 디스크 공간을 확보한 뒤 실행한다.

```bash
uv sync --project experiments/pytorch-qwen
```

## 3. 첫 추론

```bash
uv run --project experiments/pytorch-qwen qwen-generate \
  --prompt "LLM 추론에서 KV cache가 필요한 이유를 두 문장으로 설명해줘." \
  --max-new-tokens 64
```

첫 실행은 Hugging Face에서 모델을 다운로드하므로 이후 실행보다 오래 걸린다. 결과의
`model_load_seconds`와 `generation_seconds`를 구분해서 해석한다.

## 캐시 위치 변경

내장 디스크가 부족하면 충분한 여유 공간이 있는 볼륨을 지정할 수 있다.

```bash
HF_HOME=/absolute/path/to/huggingface-cache \
uv run --project experiments/pytorch-qwen qwen-generate
```

경로는 모델을 내려받기 전에 결정한다. 동일 모델을 여러 위치에 중복 다운로드하지 않는다.
