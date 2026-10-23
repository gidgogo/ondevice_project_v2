# 변경 이력

## 2026-10-23 — 분할 실행·중간 저장·복구 보완

실행 코드: [2e689f0](https://github.com/gidgogo/ondevice_project_v2/commit/2e689f075c8fd5029fa604b7b5630793e50c2220). 노트북 3개는 이 커밋을 사용한다.

- 노트북 기본값 `EPOCHS_PER_SESSION=10`: 같은 본 학습 셀을 다시 누르면 다음 10 epoch를 실행. 전체 `EPOCHS=120`, 학습률 일정, 실험 이름은 유지.
- 매 epoch 종료 후 Drive 저장. `progress.json`은 마지막 저장 epoch·남은 횟수·다음 epoch를 표시하고 `session.json`은 이번 실행 구간·코드 해시·환경을 기록.
- 학습 전 epoch 0 checkpoint도 저장해 첫 epoch 도중 끊겨도 재개 가능.
- 최신 checkpoint 외에 `last.prev.pt`를 보존. 최신 파일이 없거나 읽히지 않으면 직전 파일에서 복구. 미완료 epoch 또는 백업 이후 작업은 다시 계산.
- checkpoint 안에 로그 이력과 최적 모델을 포함해, 저장 직후 프로세스가 종료되더라도 다음 재개 시 파생 파일을 복원.
- 기존 random 실험의 epoch checkpoint는 동일 설정이면 재개 가능. center 실험과는 혼합하지 않음.
- 신규 분할 실행 코드 커밋으로 노트북 3개를 다시 생성. 옛 Drive 사본 대신 최신 GitHub 링크를 열 것.
- 로컬 테스트 **43개 통과**: 세 모델 분할/연속 실행 일치, 초기 checkpoint 재개, 읽히지 않는 최신 파일의 백업 복구, 저장 직후 강제 종료와 로그 복원, 기존 checkpoint 호환 포함. 실제 Colab GPU·Drive 장애 상황을 직접 검증한 것은 아님.

## 2026-10-23 — Colab 실행 준비와 계획 정리

### 팀원이 모델 하나씩 실행할 수 있도록 분리

- `colab/01_mini`, `02_max`, `03_maxplus`: 모델별 노트북·안내, README의 Colab 바로 실행 링크 추가.
- `configs/colab_gt.json`: 세 모델의 공통 학습 설정 분리.
- `scripts/package_colab.py`: GitHub 고정 커밋 방식과 로컬 코드 ZIP 방식으로 노트북 생성.
- 학습 데이터는 Drive에서 런타임 내부로 복사하고, 결과·환경 기록은 모델별 Drive 폴더로 분리. 고정 train/validation manifest를 사용하고 시험셋은 학습 중 평가하지 않음.

### 공정한 비교와 중단 후 재개 보완

- `scripts/train.py`: 학생을 먼저 생성하고 교사 생성은 `fork_rng` 안에서 수행해 교사 유무로 학생 초기값이 바뀌는 문제 수정.
- `initial.pt` 저장 및 `--init-checkpoint` 추가. 일반 학습과 증류가 동일한 학생 초기 가중치에서 시작하도록 지원.
- CUDA RNG 저장·복원, pilot/full 설정 및 manifest 일치 검사, 완료 epoch·상태 기록 보완.
- 세 모델의 학습·재개 및 GT/KD 초기값 일치 테스트 추가. 작은 테스트 파일만 실행할 때 `vendor/train.py`가 잘못 선택되는 테스트 import 순서도 수정.

### center에서 random으로 변경

- 세 모델의 공통 학습 입력을 **random 180×180**으로 통일. 동일 seed·epoch·이미지는 같은 crop 위치를 사용.
- 실험 이름을 **`gt_random_v1`**으로 분리. 기존 `gt_v1` center 결과는 삭제하거나 덮어쓰지 않음.
- 노트북 3개의 CODE_REF를 center 코드 `3a3602c`에서 random 코드 **`25be7c8`**로 갱신.
- 실행 준비 셀에서 실제 설정이 random / 180인지 검사하고 출력. 옛 Drive 사본은 최신 링크로 다시 열도록 안내.

### 계획·역할·상태 문서 갱신

- [실행 계획표](docs/execution_plan.md): 선행 조건, 병렬 가능한 일반 학습 3종, 담당·산출물·상태, 후속 증류와 최종 평가 순서 정리.
- [역할 분담](docs/team_roles.md): 128 패치 대신 180, LOL-v2-real 검증 69장 분석, 저장된 `best_validation_per_image.json` 활용, 기본 교사는 Max+-GT로 정리.
- [설계 결정 기록](docs/experiment_decisions.md): 규칙 기반 선택기와 620/69/100 분할 유지, Oracle 제외, 937개 Max+ 전체 학습, 동일 초기값부터 GT/KD 비교를 기록.
- README에서 실행 계획표·변경 이력으로 연결. 코드 준비 완료와 실제 GPU 학습 완료를 구분.
- 로컬 PROJECT_PLAN / PROJECT_CONTEXT 및 기존 설계서 사본도 계획 v3.2로 동기화. Canva·PDF·Notion 원본은 이번 작업에서 변경하지 않음.

### 관련 코드 커밋

| 커밋 | 변경 |
|---|---|
| [3a3602c](https://github.com/gidgogo/ondevice_project_v2/commit/3a3602cbf5308fc37d7d5f7fd56cf4afe6142180) | Colab 준비, 초기값·재개 수정, 공통 설정·안내 |
| [2e82ce2](https://github.com/gidgogo/ondevice_project_v2/commit/2e82ce2ee669012bbd8bc19c795a9485badbf0e8) | 모델별 노트북 3개 게시(center 버전) |
| [25be7c8](https://github.com/gidgogo/ondevice_project_v2/commit/25be7c8) | random 설정, 새 실험 이름, 역할·설계 문서 갱신 |
| [1fda4f2](https://github.com/gidgogo/ondevice_project_v2/commit/1fda4f2993bb7b9f8cdfc74c4089dc445374e55e) | 노트북 3개를 random 코드 커밋에 고정 |

### 검증과 한계

2026-10-23 로컬 테스트 **39개 통과**. 노트북 문법·CODE_REF·해당 커밋의 random 설정과 GitHub 게시 파일을 확인했다. 실제 Colab GPU 학습·처리시간·품질 개선은 아직 확인하지 않았다. 이번 후속 계획표·변경 이력 정리는 문서만 변경하므로 학습 코드와 고정 커밋은 유지한다.
