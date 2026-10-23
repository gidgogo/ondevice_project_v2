# ondevice_project_v2

**지식 증류와 적응형 모델 선택을 결합한 저조도 이미지 복원** — 2026-2 온디바이스AI 2조 프로젝트

UltraFast-LiNET 기반 Mini·Max·Max+에서 일반 학습과 출력 기반 지식 증류를 비교하고, 크기별로 선발한 모델을 사진 상태에 따라 선택했을 때 품질과 전체 실행 비용(노트북 CPU)이 어떻게 달라지는지 검증한다.

## 팀원별 Colab 실행

담당 모델의 링크를 열어 **GPU 런타임**을 선택한 뒤 위에서부터 실행한다. 세 작업은 서로 기다리지 않고 독립적으로 실행할 수 있다. 무료 GPU 배정 여부는 각 실행 환경에 따라 다르다.

| 담당 작업 | 실행 파일 | 바로 실행 | 결과 폴더 |
|---|---|---|---|
| Mini 일반 학습 (36개 파라미터) | [01_mini](colab/01_mini/) | [Colab에서 열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/01_mini/train.ipynb) | `mini_gt_s42/` |
| Max 일반 학습 (180개 파라미터) | [02_max](colab/02_max/) | [Colab에서 열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/02_max/train.ipynb) | `max_gt_s42/` |
| Max+ 일반 학습 (937개 파라미터, 프로젝트 확장 모델) | [03_maxplus](colab/03_maxplus/) | [Colab에서 열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/03_maxplus/train.ipynb) | `maxplus_gt_s42/` |

**먼저 준비할 것:** 각자의 Google Drive `내 드라이브/ondevice/LOLv2.zip`. 기존 프로젝트에서 확인한 원본 압축파일을 사용한다. 코드는 노트북이 GitHub의 고정 커밋에서 자동으로 가져오며 코드 ZIP 업로드는 필요 없다.

**기본 설정:** train 620 / validation 69, seed 42, 120 epoch, batch 40, **random crop 180×180**. 학습 예산은 공통 시작 설정이며 예비 실행에서 확인한다. 결과는 `gt_random_v1/`에 저장하고 이전 center 결과(`gt_v1/`)는 보존한다. **옛 Colab 사본은 자동 갱신되지 않으므로 위 링크에서 최신 사본을 열 것.** [팀 실행 안내](colab/README.md) · [저장·재개·증류 연결 안내](docs/colab.md) · [역할 분담](docs/team_roles.md)

**기본은 끝까지 실행:** `EPOCHS_PER_SESSION = 120`으로 남은 학습을 진행하면서 매 epoch Drive에 저장한다. 짧게 점검할 때만 5~10으로 줄인다. 중단되면 같은 본 학습 셀을 다시 실행해 저장 지점부터 이어간다. 총 `EPOCHS=120`·실험 이름을 바꾸지 않는다. 최신 checkpoint 읽기 실패 시 직전 백업으로 복구하고, 마지막 저장 이후 미완료 작업은 다시 계산한다.

## 연구 질문

실험 조건의 변경 이유와 이전 설계의 차이는 [실행 기준 기록](docs/experiment_decisions.md)에 정리했다.

1. 초경량 모델에서도 증류가 일반 학습보다 복원 품질을 높이는가? 교사 종류에 따라 효과가 다른가?
2. 사진별 모델 선택이 한 모델을 고정해서 쓰는 것보다 품질·전체 처리시간·메모리 면에서 유리한가?

## 진행 단계

**2026-10-23 기준:** 실행 코드·노트북 준비 완료, 실제 Colab GPU 학습 결과는 아직 확인하지 않았다. [담당·순서·완료 기준 계획표](docs/execution_plan.md) · [GitHub 변경 이력](CHANGELOG.md)

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 실험 기준 고정 | LOL-v2-real 점검·검증 분할, 평가 코드(PSNR·SSIM·LPIPS), CPU 시간 측정 규약 | 완료 ([data](docs/data.md), [측정 규약](docs/measurement_protocol.md)) |
| 2. 모델·교사 준비 | Mini·Max·Max+ 일반 학습, 고성능 교사(Retinexformer) 학습 | 일반 학습 노트북 3개, [교사 학습 노트북](colab/04_teacher/) 준비. 교사는 [Retinexformer로 선정](docs/teacher_candidates.md) |
| 3. 증류와 선발 | 8개 구성 비교 → Mini*·Max*·Max+* | [증류 노트북](colab/05_kd/)·이미지별 평가 스크립트 준비, 실행 예정 ([2단계 안내](docs/stage2.md)) |
| 4. 사진별 모델 선택 | 사진 상태별 결과 비교, 선택 기준 결정 | 규칙 기반 선택기 `scripts/selector.py` 준비, 학습 결과 대기 |
| 5. 최종 시험·시제품 | 설정 고정 후 시험, 로컬 사진 보정 | 예정 |

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `colab/01_mini`, `colab/02_max`, `colab/03_maxplus` | 담당 모델별 독립 실행 노트북과 결과 전달 안내 |
| `configs/colab_gt.json` | 세 일반 학습 모델의 공통 설정 |
| `scripts/models.py` | Mini(36) / Max(180) / Max+(937 = Max + 원본 해상도 보정망, 처음엔 Max와 같은 출력) |
| `scripts/audit_lolv2.py`, `scripts/prepare_lolv2.py` | LOL-v2-real 점검, 장면 단위 검증 분할 |
| `manifests/lolv2_real/` | 고정된 train 620 / validation 69 / test 100 목록 |
| `scripts/train.py` | 일반 학습과 출력 기반 증류(`beta`, `teacher`) |
| `scripts/teacher_models.py`, `scripts/train_teacher.py`, `configs/teacher_retinexformer.json` | 고성능 교사 Retinexformer(1,605,701개) 구조·학습. 구조 파일은 `vendor_retinexformer/`(MIT, 해시 고정) |
| `scripts/eval_checkpoint.py` | 학생·교사 checkpoint의 이미지별 PSNR·SSIM(·LPIPS) 평가. 시험은 `--allow-test` 명시 |
| `scripts/selector.py` | 규칙 기반 모델 선택기: 밝기 특징, 검증으로 임계값 고정, 무작위 선택 대비 이득, 시험 적용 |
| `colab/04_teacher`, `colab/05_kd` | 교사 학습·증류 실행 노트북 ([2단계 안내](docs/stage2.md)) |
| `scripts/evaluation.py` | PSNR·SSIM·LPIPS 평가(시험 분할은 명시적 허용 필요) |
| `scripts/benchmark_cpu.py` | CPU 시간(model / end-to-end, 평균·p95)·메모리 측정 |
| `docs/` | 실행 계획표, 역할 분담, 설계 결정, 데이터, 측정 규약, 교사 후보 |
| `CHANGELOG.md` | 변경 파일·이유·관련 커밋·검증 범위 |
| `vendor/` | 공식 UltraFast-LiNET 구현(커밋 `12e8c79`, Apache-2.0). 해시는 `provenance.json`에 고정 |
| `scripts/*_a.py`, `prepare_data.py`, `evaluate.py`, `reproduce.py` | LOL-v1 재현 저장소 [gidgogo/ultrafast-linet-baseline](https://github.com/gidgogo/ultrafast-linet-baseline) `3ead18e`에서 가져온 기존 코드(LOL-v1 전용, 기록용) |
| `weights/official_max.pkl` | 공식 공개 Max 가중치(LOL-v1) |

LOL-v1 재현 결과(시험 PSNR 17.9173 dB / SSIM 0.527209)는 기존 저장소에 기록되어 있으며, 새 LOL-v2-real 결과와 직접 비교하지 않는다.

## 실행

**Colab에서 Mini·Max·Max+를 각각 학습:** [노트북 준비·실행·재개 안내](docs/colab.md).
`scripts/package_colab.py`로 노트북 3개와 현재 코드 ZIP을 생성한다. 초기 가중치도 저장하므로 후속 증류와 출발점을 맞출 수 있다.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q
```

```bash
# 데이터: docs/data.md의 LOLv2.zip을 data/raw/에 풀기
cd scripts
../.venv/bin/python prepare_lolv2.py --data-root ../data/raw/LOLv2/Real_captured --output ../manifests/lolv2_real
# 일반 학습
../.venv/bin/python train.py --data-root ../data/raw/LOLv2/Real_captured --config ../configs/colab_gt.json --output ../runs/gt_random_v1/max_gt_s42 --set model='"max"' device='"cpu"'
# 증류 (교사 = 학습된 Max+)
../.venv/bin/python train.py --data-root ../data/raw/LOLv2/Real_captured --config ../configs/colab_gt.json --output ../runs/gt_random_v1/mini_basekd_s42 \
  --init-checkpoint ../runs/gt_random_v1/mini_gt_s42/initial.pt \
  --set model='"mini"' device='"cpu"' beta=0.5 'teacher={"model":"maxplus","checkpoint":"../runs/gt_random_v1/maxplus_gt_s42/best.pt"}'
# CPU 시간
../.venv/bin/python benchmark_cpu.py --image ../data/raw/LOLv2/Real_captured/Train/Low/00011.png
```

GPU에서는 `--set device='"cuda"'`를 붙인다. 데이터(`data/`)와 학습 결과(`runs/`)는 저장소에 올리지 않는다.

## 팀

김경서 · 백가람 · 이성수 · 정해민 · 허인혜

## 라이선스

Apache-2.0. `vendor/`는 원 저작자의 Apache-2.0 구현을 따른다.
