# 0001. 실험별 독립 uv 프로젝트를 사용한다

- 상태: 채택
- 결정일: 2026-10-25

## 배경

이 저장소는 PyTorch MPS, vLLM CPU, LMCache처럼 설치 방법과 의존성이 서로 다른
실험을 함께 다룬다. 콘텐츠와 코드는 한 저장소에서 관리하되 각 실험은 재현 가능한
독립 환경이 필요하다.

## 결정

Git 저장소는 모노레포로 사용하고, 실행 환경은 `experiments/<name>` 아래의 독립 uv
프로젝트로 관리한다. 각 프로젝트는 자체 `pyproject.toml`, `uv.lock`, `.venv`를 가진다.

초기 프로젝트는 `experiments/pytorch-qwen`이다. vLLM 환경은 실제 Mac 지원 범위와
빌드 방법을 확인한 뒤 별도 프로젝트로 추가한다.

## 단일 uv workspace를 사용하지 않는 이유

uv workspace는 여러 패키지를 하나의 lockfile과 일관된 의존성 집합으로 관리한다.
공식 문서에서도 멤버가 상충하는 요구사항이나 별도 가상환경을 필요로 하면 workspace가
적합하지 않다고 설명한다.

PyTorch MPS와 vLLM CPU는 Python 버전, PyTorch 빌드, 설치 방식이 달라질 수 있으므로
처음부터 환경을 분리한다.

참고: <https://docs.astral.sh/uv/concepts/projects/workspaces/>
