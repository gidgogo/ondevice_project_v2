# Mini·Max·Max+ 일반 학습: Colab 3개로 분리 실행

각 런타임은 독립적인 모델 하나를 학습한다. Max+도 Max 학습 완료를 기다리지 않는다.
무료 Colab의 GPU 배정과 동시 실행 수는 보장되지 않는다. 확보한 런타임만 사용하며 제한을 우회하지 않는다.

## GitHub에서 바로 실행 (팀원용)

[모델별 실행 폴더](../colab/README.md)에서 담당 모델의 **Colab에서 열기**를 선택한다. 각자의 Google Drive `내 드라이브/ondevice/`에는 기존 `LOLv2.zip`만 준비하면 된다. 코드 ZIP은 필요 없다. 노트북은 기록된 Git 커밋을 내려받으므로 세 모델이 같은 코드로 실행된다.

노트북을 Drive에 사본으로 저장하고 GPU 런타임을 선택한 뒤 위에서 아래로 실행한다. 모델마다 독립 런타임을 사용한다. 2배치 실행 점검과 검증이 성공하면 본 학습을 실행한다. 결과는 각자의 Drive에 남으므로 완료 후 해당 결과 폴더를 팀에 전달한다.

## 로컬 코드 ZIP 방식 (대안)

패키지 생성(데이터와 기존 결과는 포함하지 않음):

```bash
.venv/bin/python scripts/package_colab.py --output ../outputs/colab_gt
```

1. 생성된 `colab_code.zip`과 기존 `data/raw/LOLv2.zip`을 Google Drive의 `내 드라이브/ondevice/`에 올린다.
2. Colab의 **파일 → 노트북 업로드**로 `train_mini_gt.ipynb`, `train_max_gt.ipynb`, `train_maxplus_gt.ipynb`를 각각 연다.
3. 각 노트북에서 **런타임 → 런타임 유형 변경 → GPU**를 선택한다.
4. 각 노트북을 위에서 아래로 실행한다. Drive 연결 승인이 필요하다.
5. 2배치 실행 점검과 검증이 성공한 다음 본 학습을 실행한다.

노트북은 함께 배포된 코드 ZIP의 SHA-256을 검사한다. GitHub에 아직 올라가지 않은 수정도 ZIP에 들어 있으므로 원격의 이전 train.py를 실행하지 않는다. 코드 변경 후에는 패키지를 다시 생성하고 ZIP·노트북을 함께 교체한다.

## 공통 설정

`configs/colab_gt.json`의 공통 시작 설정은 다음과 같다: seed 42, 120 epoch, batch 40, 180×180 random crop, Adam lr 0.01, 40 epoch마다 lr ×0.1, FP32. 최종 학습 예산 확정은 별도이며 세 모델을 같은 조건으로 비교하기 위한 시작 설정이다.

고정 manifest의 학습 620쌍과 검증 69쌍만 읽는다. 데이터 분할을 새로 만들지 않으며 시험 100장은 평가하지 않는다. 학습·검증 이미지의 해시를 검사한다. 원래 LOLv2.zip의 해시도 확인하므로 다른 형태로 다시 압축한 파일은 거절한다.

메모리가 부족하면 **세 노트북 모두** 배치 크기를 동일하게 줄인다. 설정 변경 시 EXPERIMENT를 새 이름으로 지정한다. 같은 모델·실험 폴더에 두 작업을 동시에 실행하지 않는다.

2026-10-23: 세 모델의 학습 패치를 **random 180×180**으로 통일했다. 동일한 seed·epoch·이미지에서는 같은 위치를 잘라 모델 간 조건을 맞춘다. 검증은 원본 해상도 전체 이미지로 수행한다. 이전 `gt_v1/`의 center 학습 결과는 예비 실험으로 보존하고 새 `gt_random_v1/`에서 처음부터 학습한다. 기존 center checkpoint를 random 학습의 초기값으로 사용하거나 이어서 학습하지 않는다.

과거에 Drive에 저장한 노트북 사본은 자동 갱신되지 않는다. GitHub의 최신 노트북을 다시 열고 `EXPERIMENT = "gt_random_v1"` 및 실행 준비 셀의 `학습 crop: random 180` 출력을 확인한다. 설정만 변경한 뒤 예전 CODE_REF를 유지하면 옛 코드가 실행되므로, 코드 커밋 후 새 SHA로 노트북 3개를 함께 재생성한다.

기본 결과 폴더:

```text
내 드라이브/ondevice/runs/gt_random_v1/
  mini_gt_s42/
  max_gt_s42/
  maxplus_gt_s42/
```

- `initial.pt`: 학습 전 학생 가중치. 나중에 같은 모델의 증류 출발점으로 사용.
- `best.pt`: 검증 평균 PSNR이 가장 높은 checkpoint.
- `last.pt`: 마지막 완료 epoch의 가중치·optimizer·scheduler·CPU/CUDA RNG 상태. 학습 전 epoch 0에도 저장.
- `last.prev.pt`: 직전 checkpoint 백업. last.pt가 없거나 읽히지 않으면 사용.
- `progress.json`: 저장된 epoch / 전체 목표 / 다음 epoch / 저장 시각. 실시간 실행 상태 표시는 아님.
- `session.json`: 이번 실행 구간·코드 해시·환경. 총 학습 설정과 구간 길이를 구분.
- `log.csv`: epoch별 손실, 검증 PSNR/SSIM, 학습·검증 시간.
- `run.json`: 설정, 데이터 목록·코드 해시, 초기값 해시, 실행 환경.
- `summary.json`: 진행/완료 상태와 최적 검증 성능.

모든 결과는 Drive에 직접 저장한다. 데이터는 Drive에서 로컬 `/content`로 복사해 학습한다. 세 런타임이 서로 학습 중인 파일을 공유하지 않는다. GPU와 설치 패키지 목록도 `environments/`에 기록한다. 다른 GPU/PyTorch 버전 사이의 비트 단위 재현성을 보장하지는 않는다.

## 중단 후 재개

기본값은 **총 EPOCHS=120, 한 번 실행할 분량 EPOCHS_PER_SESSION=10**이다. 처음에는 1~10 epoch, 본 학습 셀을 다시 누르면 11~20 epoch를 실행한다. 구간 길이를 5나 20으로 바꿔도 전체 목표·학습률 일정·optimizer 상태는 유지된다. 이어갈 때 EPOCHS나 EXPERIMENT를 바꾸지 않는다.

같은 런타임에서는 **4번 본 학습 셀만 다시 실행**한다. 새 런타임에서는 Drive 연결·코드·데이터 준비 셀부터 실행하고 본 학습 셀로 이어간다. `last.pt` 또는 `last.prev.pt`가 있으면 자동으로 `--resume`을 붙인다. 예를 들어 37번째 epoch 도중 끊기면 36까지 저장된 상태에서 37부터 다시 시작한다. 배치 중간의 정확한 위치를 이어가는 방식은 아니다.

각 epoch의 학습·검증이 끝나면 checkpoint를 먼저 저장하고 로그·최적 모델·summary·progress를 갱신한다. checkpoint에 이력과 최적 모델도 포함하므로 저장 직후 끊겨 파생 파일 갱신이 덜 됐어도 다음 재개 시 복원한다. 최신 checkpoint를 읽을 수 없으면 직전 백업에서 복구하며, 이 경우 저장된 직전 epoch 이후 작업은 다시 계산한다.

학습 전 epoch 0도 저장하므로 첫 epoch 중단도 재개할 수 있다. 단, 최초 checkpoint 자체가 저장되기 전에 끊겼거나 두 checkpoint가 모두 없거나 손상됐으면 자동 복구할 수 없다. 기존 폴더를 보존하고 상태를 점검한다. 설정이나 데이터 목록이 달라지면 재개를 거절한다. 기존 random 버전의 epoch checkpoint도 같은 설정이면 이어갈 수 있다. center 실험은 섞지 않는다.

이미 120 epoch가 완료된 폴더를 다시 실행하면 학습을 반복하지 않는다. 파일을 매번 GitHub에 올릴 필요는 없고 학습 상태는 Drive에 저장된다. 저장과 GPU 재연결이 항상 성공함을 보장하지 않으므로 종료 전 `progress.json` 또는 노트북의 저장 완료 출력을 확인한다.

## 이후 증류와 연결

Max+ 일반 학습의 `best.pt`를 Mini·Max의 기본 교사 후보로 사용한다. 학생은 각 일반 학습 실험의 **initial.pt**에서 시작한다(일반 학습이 끝난 best.pt에서 시작하는 것이 아님).

```bash
python scripts/train.py --data-root /path/to/Real_captured \
  --config configs/colab_gt.json --output /path/to/mini_basekd_s42 \
  --init-checkpoint /path/to/mini_gt_s42/initial.pt \
  --set model='"mini"' beta=0.5 \
  'teacher={"model":"maxplus","checkpoint":"/path/to/maxplus_gt_s42/best.pt"}'
```

β=0.5는 실행 예시이며 검증으로 결정한다. 후속 증류도 일반 학습과 동일한 학습 설정을 사용한다. 온디바이스 처리시간은 Colab 학습 시간과 구분하고, 동일한 목표 노트북 CPU에서 따로 측정한다.
