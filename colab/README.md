# 담당 모델 하나씩 실행하기

| 담당 | 모델 | 노트북 | Colab |
|---|---|---|---|
| Mini 담당 | Mini, 36 parameters | [01_mini/train.ipynb](01_mini/train.ipynb) | [열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/01_mini/train.ipynb) |
| Max 담당 | Max, 180 parameters | [02_max/train.ipynb](02_max/train.ipynb) | [열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/02_max/train.ipynb) |
| Max+ 담당 | Max + 원본 해상도 보정망, 937 parameters | [03_maxplus/train.ipynb](03_maxplus/train.ipynb) | [열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/03_maxplus/train.ipynb) |

## 실행 순서

1. 팀에서 사용한 `LOLv2.zip`을 각자의 Drive `내 드라이브/ondevice/`에 준비한다. 공유 폴더를 사용하면 노트북의 DATA_ZIP 경로를 해당 파일로 바꾼다. [데이터 설명](../docs/data.md)
2. 담당 링크를 열고 Drive에 노트북 사본을 저장한다.
3. **런타임 → 런타임 유형 변경 → GPU**를 선택한다.
4. 기본 설정을 확인하고 위에서부터 실행한다. Drive 연결 승인은 직접 진행한다.
5. 데이터 확인 → 짧은 실행 점검 → 본 학습 순서로 진행한다.
6. 완료 후 아래 결과 폴더를 팀에 전달한다. 런타임이 끊기면 같은 노트북을 다시 실행해 마지막 완료 epoch부터 재개한다.

코드는 자동 다운로드된다. **코드 ZIP 업로드·Git 설치·수동 데이터 분할은 필요 없다.** 데이터와 결과는 GitHub에 올리지 않는다.

## 공통 조건과 분담

- 일반 학습 3종만 실행한다. 이번 단계에는 증류·선택기·시험 평가가 포함되지 않는다.
- Max+는 Max의 학습 완료를 기다리지 않고 처음부터 독립적으로 학습한다. Max+는 원 논문의 모델명이 아니라 프로젝트에서 추가한 구성이다.
- 공통 시작 설정: seed 42, 120 epoch, batch 40, center crop 180, train 620 / validation 69.
- 설정은 [공통 JSON](../configs/colab_gt.json)에 있다. 학습 횟수·배치 등을 변경할 때는 팀에서 합의해 세 모델에 똑같이 적용한다.
- 메모리 부족이면 배치를 세 모델 모두 동일하게 줄이고 새 실험 이름을 사용한다.
- 각 노트북의 코드는 특정 커밋에 고정된다. 코드 변경을 반영하려면 새 커밋으로 세 노트북을 함께 다시 생성한다.
- GPU가 다르면 학습 시간만으로 모델 우열을 판단하지 않는다. 최종 속도 비교는 동일 노트북 CPU에서 한다.
- 각자의 Drive에 결과가 생긴다. 같은 모델·실험 경로에 여러 런타임이 동시에 쓰지 않도록 한다.

## 결과 전달

각 담당자는 `내 드라이브/ondevice/runs/gt_v1/<모델>_gt_s42/` 폴더 전체를 전달한다.

| 파일 | 의미 |
|---|---|
| `initial.pt` | 증류와 출발점을 맞출 초기 가중치 |
| `best.pt` | 검증 PSNR로 선발한 모델 |
| `last.pt` | 중단 후 재개용 checkpoint |
| `log.csv` | 학습 곡선·검증 품질·시간 |
| `run.json`, `summary.json` | 조건·진행 상태·최적 검증 결과 |
| `best_validation_per_image.json` | 이미지별 검증 결과 |

추가로 `environments/gt_v1/<모델>_s42/`의 GPU·패키지 기록도 함께 전달한다. Max+의 `best.pt`는 다음 단계 Mini·Max 증류의 기본 교사 후보다.

자세한 재개·오류 대처: [Colab 안내](../docs/colab.md).
