# Project State

## 현재 단계

시리즈 0: Mac 개발환경과 추론 실험환경 구축

## 확인 완료

- Apple M3 Pro, 통합 메모리 18GB, macOS 15.7.4
- Git, uv, Xcode Command Line Tools, VS Code 설치 상태
- PyTorch MPS 백엔드 사용 가능
- 첫 대상 모델: `Qwen/Qwen3-4B`
- 모델 revision: `1cfa9a7208912126459214e8b04321603b3df60c`

## 준비됨

- PyTorch Qwen 실험용 독립 uv 프로젝트
- 다운로드 전 환경 점검 스크립트
- MPS 추론과 기본 측정 스크립트

## 아직 완료하지 않음

- 프로젝트 가상환경 설치
- Qwen3-4B 다운로드
- PyTorch MPS 첫 추론
- 기준 성능 측정

## 현재 차단 조건

2026-10-25 확인 당시 내장 디스크 여유 공간은 8.7GiB였다. Qwen3-4B 모델
저장소만 약 8.06GB이므로 패키지, 캐시, 임시 공간을 포함하기에 부족하다.

첫 다운로드 전에 최소 20GiB의 여유 공간을 확보하거나 Hugging Face 캐시를 여유 있는
외부 볼륨으로 지정한다.

## 다음 행동

디스크 공간 또는 외부 캐시 위치를 결정한 뒤 preflight를 다시 실행한다.
