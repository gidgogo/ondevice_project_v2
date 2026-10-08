# ondevice_project_v2

**지식 증류와 적응형 모델 선택을 결합한 저조도 이미지 복원** — 2026-2 온디바이스AI 2조 프로젝트

UltraFast-LiNET 기반 Mini·Max·Max+에서 일반 학습과 출력 기반 지식 증류를 비교하고, 크기별로 선발한 모델을 사진 상태에 따라 선택했을 때 품질과 전체 실행 비용(노트북 CPU)이 어떻게 달라지는지 검증한다.

## 연구 질문

1. 초경량 모델에서도 증류가 일반 학습보다 복원 품질을 높이는가? 교사 종류에 따라 효과가 다른가?
2. 사진별 모델 선택이 한 모델을 고정해서 쓰는 것보다 품질·전체 처리시간·메모리 면에서 유리한가?

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 실험 기준 고정 | LOL-v2-real 점검·검증 분할, 평가 코드(PSNR·SSIM·LPIPS), CPU 시간 측정 규약 | 진행 예정 |
| 2. 모델·교사 준비 | Mini·Max·Max+ 일반 학습, 고성능 교사 후보 검증 | 예정 |
| 3. 증류와 선발 | 8개 구성 비교 → Mini*·Max*·Max+* | 예정 |
| 4. 사진별 모델 선택 | 사진 상태별 결과 비교, 선택 기준 결정 | 예정 |
| 5. 최종 시험·시제품 | 설정 고정 후 시험, 로컬 사진 보정 | 예정 |

## 현재 저장소 구성

LOL-v1 재현 저장소 [gidgogo/ultrafast-linet-baseline](https://github.com/gidgogo/ultrafast-linet-baseline) 커밋 `3ead18e`에서 재사용할 코드를 가져왔다. 아직 LOL-v1 기준으로 작성되어 있으며, 1단계에서 LOL-v2-real에 맞게 수정한다.

| 경로 | 내용 |
|---|---|
| `vendor/` | 공식 UltraFast-LiNET 구현(커밋 `12e8c79`, Apache-2.0). 해시는 `provenance.json`에 고정 |
| `scripts/` | 데이터 분할(`prepare_data.py`), 학습(`train_a.py`), 평가(`evaluate.py`), 공개 가중치 재현·CPU 측정(`reproduce.py`) |
| `tests/` | 재현성·평가·재개 테스트 |
| `configs/baseline_a.json` | 기존 LOL-v1 Max 학습 설정 |
| `weights/official_max.pkl` | 공식 공개 Max 가중치(LOL-v1) |

LOL-v1 재현 결과(시험 PSNR 17.9173 dB / SSIM 0.527209)는 기존 저장소에 기록되어 있으며, 새 LOL-v2-real 결과와 직접 비교하지 않는다.

## 실행

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q
```

데이터(`data/`)와 학습 결과(`runs/`)는 저장소에 올리지 않는다.

## 팀

김경서 · 백가람 · 이성수 · 정해민 · 허인혜

## 라이선스

Apache-2.0. `vendor/`는 원 저작자의 Apache-2.0 구현을 따른다.
