# LLM Inference From Scratch

**LLM 추론/서빙 최적화**

작은 모델을 직접 실행하고 내부 동작을 관찰하면서 LLM 추론과 서빙 최적화를 단계적으로 학습하는 프로젝트입니다.

이 저장소는 다음 콘텐츠의 원본을 함께 관리합니다.

- 재현 가능한 실습 코드와 개발환경
- 구조와 데이터 흐름을 설명하는 그림
- 성능 측정 과정과 결과

## 현재 실험

- [PyTorch MPS에서 Qwen3-4B 실행](experiments/pytorch-qwen/README.md)

Python 실행 환경은 실험별로 분리된 uv 프로젝트로 관리합니다. 구조를 선택한 이유는
[Python 프로젝트 구성 결정](docs/decisions/0001-python-project-layout.md)에 기록합니다.
