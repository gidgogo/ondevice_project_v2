# 변경 이력

## 2026-10-27 — 선택 비용 측정 범위 분리

- `scripts/selector.py time`: `compute`(이미 읽은 사진에서 썸네일·밝기 통계·규칙 비교)와 `with_decode`(PNG 읽기 포함)를 따로 측정하고 `--output`으로 JSON 저장. 모델 end_to_end에 PNG 읽기가 이미 들어 있으므로 선택 시스템 시간은 end_to_end + compute로 계산하고, `fit --selector-ms`에는 compute 중앙값을 쓴다(읽기 시간 이중 계산 방지). 특징 값 계산 결과는 수정 전과 같음을 확인.

## 2026-10-27 — Retinexformer 교사 증류 3개 결과 추가

- `results/midterm_lolv2/`: Mini·Max·Max+ ← Retinexformer(9000회 best, 21.99 dB) 증류 결과(β=0.3, 120 epoch, Colab T4). 세 실행 모두 일반 학습 `initial.pt`·설정 일치, 같은 교사 SHA-256(`e5c2f0a2…`) 확인. 일반 학습 대비 PSNR −0.006 / +0.002 / +0.009 dB. 비교표·품질 CSV·학습 기록·그림·7장 초안 갱신, 증류 손실 그림 추가.

## 2026-10-26 — CPU 측정 원자료 CSV, 사진 보정 스크립트

- `scripts/benchmark_cpu.py`: `--csv` 추가. 반복마다 `model_id, checkpoint_status, width, height, threads, repeat_id, inference_ms(model), total_ms(end_to_end)` 한 줄(역할 문서 B 결과 필드). JSON 보고에 표준편차와 반복별 값, 사용한 checkpoint 추가. 측정 방식은 그대로.
- `scripts/enhance_image.py`: 학습된 best.pt로 사진 한 장 또는 폴더를 보정해 PNG 저장. 모델과 checkpoint가 다르면 거부.
- `tests/test_benchmark_csv.py` 추가.

## 2026-10-26 — 서버 일괄 실행기 (학교 4090)

기존 학습 코드(`train.py`, `train_teacher.py` 등)는 바꾸지 않았다. 실행 순서를 묶는 파일만 추가했다.

- `scripts/run_remaining.py`: Retinexformer 교사를 저장된 `last.pt`부터 30000회까지 이어서 학습 → best.pt를 고정 사본으로 복사 → Max+·Max·Mini ← Retinexformer 증류(β=0.3, 일반 학습 `initial.pt`·설정 일치 검사) → 일반 학습·Max+ 교사 증류와 비교표, A 형식 품질 CSV, 결과 zip. 다시 실행하면 끝난 단계는 건너뛰고 끊긴 단계는 이어서 한다. 시작 전에 필요한 파일·설정·데이터 해시를 모두 확인한다. `--smoke`는 짧은 점검으로 예상 시간을 계산하고 테스트 알림을 보낸다.
- 알림: 단계 완료·실패·중단 시 Gmail(앱 비밀번호, 환경변수) 또는 ntfy. 알림이 실패해도 학습은 계속된다. 중단(`kill`)되면 학습 프로세스도 같이 끝낸다.
- `scripts/run_4090.sh`: 가상환경 설치, `smoke` / `full`(백그라운드) / `status` / `stop`.
- `tests/test_server_runner.py`: 작은 데이터·CPU로 교사 이어서 학습 → 증류 3개 → 결과 정리, 재실행 시 다시 학습하지 않음을 확인. 전체 로컬 테스트 **55개 통과**. 중단 후 재실행 결과가 중단 없이 돌린 결과와 같은지도 따로 확인. 실제 4090 실행은 아직 하지 않음.

## 2026-10-26 — 중간발표 결과와 A 기준 묶음 수정본 올림

코드 변경 없음. 결과 파일과 문서만 추가했다.

- `results/midterm_lolv2/`: Colab T4에서 끝난 일반 학습 3개(Mini 10.55 / Max 13.53 / Max+ 17.13 dB), Max+ 교사 증류 2개(β=0.3, Mini −0.01 / Max +0.01 dB), 학습 중인 Retinexformer 교사(18000/30000회, 최고 21.99 dB @9000회)의 검증 69장 결과. 실행 폴더의 `log.csv`·`summary.json`·`best_validation_per_image.json`을 옮기거나 합쳤고, 품질 CSV 평균이 각 `summary.json`의 best 값과 같음을 확인. 시험 100장은 사용하지 않음.
- `baseline_a/`: 팀원 A의 LOL-v1 기준 묶음 2차 수정본(검사기 경로, LOL-v2 명부·변환 도구, 측정 범위 문구 등 19개 항목)과 3차 메모. `check_handoff.py` PASS.

## 2026-10-24 — 교사·증류 노트북 설정 반영, 교사 복구, 선택기 누락 검사

실행 코드: [1c2ad21](https://github.com/gidgogo/ondevice_project_v2/commit/1c2ad2128f75242559859b8ddc1eced21c3c88bf). `colab/04_teacher`, `colab/05_kd`는 이 커밋을 사용한다. 일반 학습 노트북 3개는 바꾸지 않았다. **이전 `a0a07aa` 노트북으로 교사 본 학습을 시작하지 말 것.**

- 교사·증류 노트북: 점검 셀에서 만든 명령을 본 학습 셀이 다시 쓰던 문제 수정. 각 실행 셀이 `teacher_plan()` / `kd_plan()`으로 **그 순간의 설정 셀 값**에서 경로·명령을 새로 만들고 준비 파일을 다시 검사한다. 점검 후 `TOTAL_ITERS`나 `BETA`를 바꾸면 설정 셀을 다시 실행한 뒤 본 학습 셀을 실행한다.
- `scripts/train_teacher.py`: 마지막 checkpoint 저장 직후 `best.pt`·로그·완료 기록을 쓰기 전에 끊긴 경우, 재개 시 checkpoint에서 결과 파일과 `summary.json`을 복원한다.
- `scripts/selector.py`: 검증 manifest의 모든 이미지가 세 결과 파일에 빠짐없이 있어야 한다(세 파일에서 같은 이미지가 함께 빠져도 거부, 추가·중복 이미지도 거부). 다른 분할의 평가 보고서도 거부한다.
- 위 세 경우의 테스트 5개 추가(이전 커밋에서는 실패). 전체 로컬 테스트 **54개 통과**. 점검 뒤 `TOTAL_ITERS`·`BETA`를 바꿔 본 학습에 반영되는지 노트북 셀 흐름을 CPU·작은 데이터로 대체 실행해 확인. 실제 Colab GPU 실행은 아직 확인하지 않음.

## 2026-10-23 — 고성능 교사(Retinexformer)·증류·선택기 실행 코드

실행 코드: [a0a07aa](https://github.com/gidgogo/ondevice_project_v2/commit/a0a07aafa9f3bcb6f87842521cdc839a4876e88d). 새 노트북 2개(`colab/04_teacher`, `colab/05_kd`)는 이 커밋을 사용한다. 일반 학습 노트북 3개와 그 고정 커밋은 바꾸지 않았다.

- 고성능 교사를 **Retinexformer**(학습 파라미터 1,605,701개, MIT)로 선정. HVI-CIDNet은 이번 범위에서 제외. 구조 파일은 수정 없이 `vendor_retinexformer/`에 두고 해시를 `provenance.json`에 고정.
- `scripts/train_teacher.py`: 우리 train 620쌍으로 처음부터 학습, validation 69쌍으로 best 선택. 원 설정(L1, Adam 2e-4, grad clip 0.01, 256 패치, batch 8, 뒤집기·회전)을 따르되 반복 수는 30,000(시작안)으로 줄이고 단일 cosine 사용, mixup 제외. 반복 번호로 배치를 정해 재개해도 같은 순서. 점검 실행이 반복당 시간과 전체 예상 시간을 기록.
- `scripts/train.py`: 외부 교사도 학생과 같은 random 패치에서 온라인으로 계산(`teacher={"model": "retinexformer", ...}`). 교사 checkpoint 해시를 `run.json`에 기록하고 다른 교사로의 재개를 거부. 일반 학습 동작은 그대로.
- `scripts/eval_checkpoint.py`: 학생·교사의 이미지별 PSNR·SSIM(·LPIPS) 평가. 시험 분할은 `--allow-test` 필요.
- `scripts/selector.py`: 64×64 썸네일 밝기 규칙 선택기. 검증에서 임계값 고정(Max+* 대비 PSNR 0.2 dB 이내 중 가장 빠른 규칙), 같은 호출 비율의 무작위 선택 대비 이득·심한 저하 비율 기록, 시험에는 그대로 적용. 선택 비용 측정 하위 명령 포함.
- 노트북: 교사 학습(점검·예상 시간·본 학습·Max+ 대비 비교), 증류(학생·교사·β 선택, 일반 학습 initial.pt·설정 일치 검사, 일반 학습 대비 차이 출력).
- 로컬 테스트 **49개 통과**(교사 구조 해시·크기, 교사 학습 분할 재개 일치, 온라인 교사 증류, 시험 평가 거부, 선택기 격자·무작위 기준·시험 적용 포함). 두 노트북은 CPU·작은 데이터로 셀 흐름을 대체 실행해 확인. **실제 Colab GPU 학습·교사 성능·증류 효과는 아직 확인하지 않음.**

## 2026-10-23 — 기본 실행 분량 120으로 조정

실행 코드: [4ecc0ba](https://github.com/gidgogo/ondevice_project_v2/commit/4ecc0ba3c8977d52e22f51bd9733ec1814c54be3). 최신 노트북 3개는 이 커밋을 사용한다.

- EPOCHS_PER_SESSION 기본값을 10에서 **120**으로 변경해 남은 학습을 끝까지 진행하도록 함. 총 EPOCHS=120은 그대로이며 초과 학습하지 않음.
- 매 epoch 저장·직전 백업·복구는 유지. 짧게 나누는 기능은 시간 측정·제한된 실행용 선택 사항으로 남김.
- 10 epoch마다 수동 재실행하고 유휴 런타임을 방치하는 부담을 줄이기 위한 변경. 짧은 구간 종료 후에는 직접 재실행해야 하며 연속 실행 완료도 보장되지 않음을 안내.
- 노트북·계획표·역할 문서·로컬 사본을 동기화. 학습 엔진은 변경하지 않았고 생성된 셀의 문법·설정·고정 커밋을 검사.

## 2026-10-23 — 분할 실행·중간 저장·복구 보완

실행 코드: [2e689f0](https://github.com/gidgogo/ondevice_project_v2/commit/2e689f075c8fd5029fa604b7b5630793e50c2220). 당시 노트북 3개는 이 커밋을 사용했다.

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
