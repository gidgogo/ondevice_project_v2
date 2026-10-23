# 고성능 교사: Retinexformer (2026-10-23 선정)

계획: 우리 학습 분할(620쌍)로 직접 학습한 큰 모델을 고성능 교사로 쓴다. 공개 사전학습 가중치는 공식 Train 689쌍 전체(우리 검증 69쌍 포함)나 LOL-v1으로 학습되었을 수 있어, 그대로 쓰면 검증 기반 선택이 오염된다.

## 선정

| 후보 | 코드·라이선스 | 학습 파라미터(직접 계산) | 원 학습 설정 | 결정 |
|---|---|---:|---|---|
| **Retinexformer** (ICCV 2023) | [GitHub](https://github.com/caiyuanhao1998/Retinexformer) `1e9a0ef`, MIT | **1,605,701** | `Options/RetinexFormer_LOL_v2_real.yml`: L1, Adam 2e-4, grad clip 0.01, 256 패치, batch 8, 150k iter, 뒤집기·회전, mixup | **선정** |
| HVI-CIDNet (CVPR 2025) | [GitHub](https://github.com/Fediory/HVI-CIDNet), MIT | 1,975,569 | 기본 1000 epoch, lr 1e-4 | 이번 범위에서 제외 |

선정 이유: 정답 쌍으로 학습하는 지도학습 모델이라 출력이 정답 분포와 같은 방향이고, 원 저장소에 LOL-v2-real 학습 설정이 있으며, 구조 파일 하나(`einops`만 추가 의존)로 우리 학습 코드에 붙일 수 있다. README의 성능 수치는 GT mean 등 측정 조건이 달라 선정 근거로 쓰지 않았다.

## 우리 실행 설정 (`configs/teacher_retinexformer.json`, `scripts/train_teacher.py`)

| 항목 | 원 설정 | 우리 설정 | 이유 |
|---|---|---|---|
| 데이터 | 공식 Train 689 | 고정 train 620, checkpoint 선택은 validation 69 | 검증·시험 분리 |
| 반복 수 | 150,000 | **30,000 (시작안)** | 무료 Colab 시간. 노트북 점검 셀의 예상 시간을 보고 본 학습 전에 확정 |
| 학습률 | 2e-4, cyclic cosine restart | 2e-4 → 1e-6 단일 cosine | 줄인 반복 수에 맞춘 단순화 |
| 손실·최적화 | L1, Adam(0.9, 0.999), grad clip 0.01 | 동일 | |
| 입력 | 256 패치, batch 8, 뒤집기·회전 | 동일 (반복 번호로 위치·회전 고정) | 재개해도 같은 배치 |
| mixup | 사용 | 사용 안 함 | 단순화. 필요하면 별도 실험 |
| 검증 | — | 1,000회마다 검증 69장 PSNR로 best 선택 | 학생과 같은 평가 규약 |

## 교사 사용 조건

1. 우리 검증 분할에서 기본 교사(Max+-GT)보다 PSNR·SSIM이 확실히 높을 것. 노트북 5단계가 차이를 출력한다. 못 넘으면 "더 좋은 교사를 확보하지 못함"으로 기록하고 반복 수·설정을 재검토한다.
2. 증류할 때는 교사 checkpoint를 고정하고, **학생과 같은 random 패치에서 매 배치 온라인으로** 교사 출력을 계산한다(`teacher={"model": "retinexformer", "checkpoint": ...}`). PNG로 저장한 출력을 자르는 방식(`outputs_dir`)은 8bit 양자화와 전체 이미지 추론·잘라내기 차이가 생기므로 기본으로 쓰지 않는다.
3. 교사 학습 시간(GPU 종류 포함)은 증류의 추가 비용으로 따로 보고한다.
