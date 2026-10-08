# ondevice_project_v2

**지식 증류와 적응형 모델 선택을 결합한 저조도 이미지 복원** — 2026-2 온디바이스AI 2조 프로젝트

UltraFast-LiNET 기반 Mini·Max·Max+에서 일반 학습과 출력 기반 지식 증류를 비교하고, 크기별로 선발한 모델을 사진 상태에 따라 선택했을 때 품질과 전체 실행 비용(노트북 CPU)이 어떻게 달라지는지 검증한다.

## 연구 질문

1. 초경량 모델에서도 증류가 일반 학습보다 복원 품질을 높이는가? 교사 종류에 따라 효과가 다른가?
2. 사진별 모델 선택이 한 모델을 고정해서 쓰는 것보다 품질·전체 처리시간·메모리 면에서 유리한가?

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 실험 기준 고정 | LOL-v2-real 점검·검증 분할, 평가 코드(PSNR·SSIM·LPIPS), CPU 시간 측정 규약 | 완료 ([data](docs/data.md), [측정 규약](docs/measurement_protocol.md)) |
| 2. 모델·교사 준비 | Mini·Max·Max+ 일반 학습, 고성능 교사 후보 검증 | 진행 중: 모델·학습·증류 코드 완료, CPU 예비 학습 중, [교사 후보](docs/teacher_candidates.md) 조사 |
| 3. 증류와 선발 | 8개 구성 비교 → Mini*·Max*·Max+* | 예정 |
| 4. 사진별 모델 선택 | 사진 상태별 결과 비교, 선택 기준 결정 | 예정 |
| 5. 최종 시험·시제품 | 설정 고정 후 시험, 로컬 사진 보정 | 예정 |

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `scripts/models.py` | Mini(36) / Max(180) / Max+(937 = Max + 원본 해상도 보정망, 처음엔 Max와 같은 출력) |
| `scripts/audit_lolv2.py`, `scripts/prepare_lolv2.py` | LOL-v2-real 점검, 장면 단위 검증 분할 |
| `manifests/lolv2_real/` | 고정된 train 620 / validation 69 / test 100 목록 |
| `scripts/train.py` | 일반 학습과 출력 기반 증류(`beta`, `teacher`) |
| `scripts/evaluation.py` | PSNR·SSIM·LPIPS 평가(시험 분할은 명시적 허용 필요) |
| `scripts/benchmark_cpu.py` | CPU 시간(model / end-to-end, 평균·p95)·메모리 측정 |
| `docs/` | 데이터, 측정 규약, 교사 후보 |
| `vendor/` | 공식 UltraFast-LiNET 구현(커밋 `12e8c79`, Apache-2.0). 해시는 `provenance.json`에 고정 |
| `scripts/*_a.py`, `prepare_data.py`, `evaluate.py`, `reproduce.py` | LOL-v1 재현 저장소 [gidgogo/ultrafast-linet-baseline](https://github.com/gidgogo/ultrafast-linet-baseline) `3ead18e`에서 가져온 기존 코드(LOL-v1 전용, 기록용) |
| `weights/official_max.pkl` | 공식 공개 Max 가중치(LOL-v1) |

LOL-v1 재현 결과(시험 PSNR 17.9173 dB / SSIM 0.527209)는 기존 저장소에 기록되어 있으며, 새 LOL-v2-real 결과와 직접 비교하지 않는다.

## 실행

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
../.venv/bin/python train.py --data-root ../data/raw/LOLv2/Real_captured --output ../runs/max_gt_s42 --set model='"max"'
# 증류 (교사 = 학습된 Max+)
../.venv/bin/python train.py --data-root ../data/raw/LOLv2/Real_captured --output ../runs/mini_basekd_s42 \
  --set model='"mini"' beta=0.5 'teacher={"model":"maxplus","checkpoint":"../runs/maxplus_gt_s42/best.pt"}'
# CPU 시간
../.venv/bin/python benchmark_cpu.py --image ../data/raw/LOLv2/Real_captured/Train/Low/00011.png
```

GPU에서는 `--set device='"cuda"'`를 붙인다. 데이터(`data/`)와 학습 결과(`runs/`)는 저장소에 올리지 않는다.

## 팀

김경서 · 백가람 · 이성수 · 정해민 · 허인혜

## 라이선스

Apache-2.0. `vendor/`는 원 저작자의 Apache-2.0 구현을 따른다.
