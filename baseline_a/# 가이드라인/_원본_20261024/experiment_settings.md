# Max 베이스라인 실험 설정

이 파일은 팀 전체가 유지할 데이터·모델·평가 기준이다. 파일 사용법, 데이터 준비, 재실행 명령, 발표 초안은 `README.md`에 있다.

## 1. 모델과 가중치

| 항목 | 값 |
|---|---|
| 모델 ID | `UltraFastLiNET_Max` |
| 모델 생성 함수 | 고정 저장소 `vendor/model.py`의 `ultrafast_linet_max()` |
| 파라미터 수 | 180개 |
| checkpoint ID | `a_max_seed42_epoch20` |
| 전달 파일 | 이 폴더의 `best.pt` |
| 저장소 원래 경로 | `reports/a_full_seed42/best.pt` |
| 가중치 출처 | 저장소에 기록된 직접 학습 결과. 공식 공개 `weights/official_max.pkl`과 구분 |
| 초기화 | 저장소 학습 기록상 seed 42의 새 초기값. 공식 checkpoint를 초기값으로 사용하지 않음 |
| 완료된 학습량 | 저장소 기록상 360 epoch |
| 평가용 선택 시점 | validation 평균 PSNR이 가장 높은 epoch 20 |
| 시험 자료의 checkpoint 선택 사용 | 사용하지 않은 것으로 평가 보고서에 기록됨 |
| 추가 합성 잡음 | 없음. 원래 사진의 촬영 잡음까지 없다는 뜻은 아님 |

`best.pt` SHA-256:

`1e2df6a4e420bf8335790cd14c8102ca58f9d804126c7a31f3275f648d445fe6`


## 2. 코드 버전

- 저장소: https://github.com/7b7hom/ultrafast-linet-baseline
- 고정 commit: `3ead18e7b34444aa19eb39c815da86d2dc024d08`
- upstream 코드 commit: `12e8c79566235fecc952d9a148011cf5fbb2c8ba`
- 공통 지표: `common_metrics.py`는 고정 저장소 `vendor/metrics.py`의 동일 복사본이다.
- 전체 모델·학습·시험 평가 코드는 고정 저장소에서 받는다. 이 ZIP은 전달용 결과 묶음이며 전체 저장소를 포함하지 않는다.

## 3. 데이터

| 분할 | 이미지 쌍 수 | 용도 |
|---|---:|---|
| train | 435 | 모델 가중치 학습 |
| validation | 50 | checkpoint 선택, 후속 설정 및 분석 기준 결정 |
| test | 15 | 고정한 모델의 시험 결과 보고 |

- 데이터셋: LOL-v1. 제안서의 LOL-v2-real 최종 실험과 구분한다.
- `our485`에서 train 435쌍과 validation 50쌍을 나눈다. `eval15`는 test 15쌍이다.
- `split_manifest.csv`의 고정 목록을 유지한다. 일반적인 무작위 분할로 다시 나누지 않는다.
- 이미지 ID: `LOL-v1/` + 원래 파일명. 예: `LOL-v1/1.png`.
- low/high SHA-256을 공유하는 중복 그룹이 학습·검증 양쪽으로 나뉘지 않게 한 분할이다. 비슷한 장면까지 모두 독립이라고 보장하지는 않는다.
- 원본 입력·정답은 ZIP에 포함하지 않았다. 데이터 접근 방법은 `README.md`의 ‘원본 데이터 준비’를 참고한다.
- 시험 결과는 이미 공개된 상태다. 후속 설정은 검증 자료에서 정한다.

## 4. 전처리와 품질 지표

| 항목 | 고정 기준 |
|---|---|
| 입력/정답 채널 | RGB |
| 자료형·범위 | float32, [0,1] |
| 평가 크기 | 원본 전체. 이 결과는 가로 600 × 세로 400 |
| 평가 텐서 | 한 장씩 `[1,3,400,600]`, batch size 1 |
| 모델 실행 | `eval()` 및 gradient를 계산하지 않는 평가 모드 |
| 사용할 출력 | 원 모델의 `model(x)[-1]` |
| 예측 처리 | 예측만 [0,1]로 clamp |
| 추가 처리 | Y 변환·정답 밝기 맞추기·테두리 제거·평가 전 uint8 양자화 없음 |
| PSNR | `common_metrics.psnr`; 완전히 같은 영상은 원 구현상 100 dB |
| SSIM | `common_metrics.ssim`; `pytorch-msssim==1.0.0`의 기본 단일 SSIM |
| 집계 | 이미지별 PSNR·SSIM을 구한 후 집합별 산술평균 |
| PNG | float 텐서로 먼저 채점한 뒤 시각화용으로 저장. 저장 PNG로 점수를 재계산하지 않음 |

`common_metrics.py`는 RGB 변환·정규화·clamp를 자동으로 모두 수행하지 않는다. 호출부가 위 조건을 맞춘다. 학습 손실의 MS-SSIM과 보고용 SSIM은 별개다. LPIPS는 이번 결과에 포함되지 않는다.

## 5. 환경 기록

| 항목 | 이번 제출 평가 보고서에 기록된 값 |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.14.0 |
| 운영체제 | macOS-27.0.1-arm64-arm-64bit-Mach-O |
| CPU | Apple M3 |
| torch threads / interop threads | 1 / 1 |
| 패키지 버전 | `requirements.lock.txt`의 30개 고정 버전 |

고정 저장소의 기존 기준 평가 기록은 Python 3.12.6 / Apple M2 Pro 환경이다. 이번 제출 보고서의 환경과 구분한다. Colab·다른 OS·다른 장치를 쓰면 실제 환경을 별도로 기록하고 결과를 비교한다. 서로 다른 환경에서의 실행 성공 및 비트 단위 수치 일치는 이 정리 작업에서 검증하지 않았다.

원래 `environment_packages.txt`의 패키지명 대소문자와 `-`/`_` 차이를 정규화하면 `requirements.lock.txt`와 30개 버전이 일치했다. 중복된 패키지 목록 파일은 제외하고 Python·OS·CPU 기록은 이 문서와 원 평가 JSON에 보존했다.

## 6. 학습 설정 전체

원래 전달물의 `baseline_a.json` 내용을 아래에 그대로 보존했다. 기계가 읽을 원본 설정 파일은 고정 저장소의 `configs/baseline_a.json`에도 있다. 이 설정은 Max의 기존 실행 조건이며 Mini의 자동 실행 설정을 뜻하지 않는다.

```json
{
  "experiment": "A",
  "model": "official-checkpoint-compatible UltraFast-LiNET-Max",
  "seed": 42,
  "epochs": 360,
  "batch_size": 40,
  "crop": 180,
  "lr": 0.01,
  "lr_step": 40,
  "lr_gamma": 0.1,
  "device": "cpu",
  "threads": 1,
  "num_workers": 0,
  "grad_weights": [1.0, 1.0, 0.04],
  "selection": "validation_psnr_sigma0",
  "additional_noise_probability": 0.0
}
```

학습은 중앙 crop 180×180, Adam을 사용한다. 기존 기본 MS-SSIM 손실을 유지하면 짧은 변이 160보다 커야 한다. D는 역할 문서의 128×128 예시를 현재 코드에 그대로 적용하지 말고 기존 180×180에서 시작하거나 변경한 손실을 명시한다. 평가에는 이 학습 crop을 적용하지 않는다.

## 7. 결과와 출처

| split | N | 평균 PSNR(dB) | 평균 SSIM |
|---|---:|---:|---:|
| validation | 50 | 16.061188888549804 | 0.5156236219406128 |
| test | 15 | 17.917277908325197 | 0.5272092560927073 |

- 시험 상세 보고서: `evidence/test_metrics.json`, 출처 라벨 `provided_recheck_report`.
- 검증 상세 보고서: `evidence/validation_metrics.json`, 출처 라벨 `validation_recheck`.
- 원 평가 환경·코드 버전·가중치 해시: `evidence/recheck_status.json`.
- JSON에 남은 `runs/a_member_a_*` 경로는 원 실행의 출처 경로다. 이 ZIP 안에서 그 경로를 찾는 것이 아니다.
- 원 문서의 평가 실행자 기재명: 해민. 정확한 추론 시작·종료 시각은 제출물에 별도 기록되어 있지 않다.
- `evidence/recheck_status.json`의 `generated_at`은 원래 전달물 생성 시각 `2026-10-23T18:37:36.567427+09:00`이다. 실제 추론 시각이나 이번 정리본 생성 시각을 뜻하지 않는다.
- 이번 정리 작업은 제출된 수치·가중치·이미지·평가 JSON을 보존했다. 원본 low/high로 모델을 새로 추론한 결과는 아니다.

## 8. 후속 모델

현재 결과는 Max 180개 파라미터 모델 한 조건이다. Mini 36개 모델은 학습 가중치가 나오면 같은 ID·목록·지표로 평가한다. Max+ 약 950개는 계획상의 확장 모델이며 실제 구현에서 파라미터 수를 확인해야 한다. 두 모델의 품질 결과는 이 묶음에 없다.

파라미터 수는 모델이 가진 값의 개수이고 epoch는 학습 반복 횟수다. 후속 모델은 동일한 품질 CSV 8개 열을 유지하고 `model_id`, `checkpoint_id`를 구분한다. 총 파라미터 수와 실제 업데이트한 파라미터 수가 다르면 설정에 둘 다 기록한다.
