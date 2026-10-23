# Mini 일반 학습 담당

[Colab에서 열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/01_mini/train.ipynb) · [팀 공통 안내](../README.md)

- 모델: **Mini**, 학습 파라미터 **36개**
- 노트북: [train.ipynb](train.ipynb)
- 준비 데이터: Drive의 `내 드라이브/ondevice/LOLv2.zip`
- 결과: `내 드라이브/ondevice/runs/gt_random_v1/mini_gt_s42/`
- 환경 기록: `내 드라이브/ondevice/environments/gt_random_v1/mini_s42/`
- 공통 설정: seed 42 / 120 epoch / batch 40 / 학습 620쌍 / 검증 69쌍

다른 모델의 학습 결과를 기다리지 않고 시작합니다.

완료 후 결과 폴더와 환경 기록을 팀에 전달하세요. `initial.pt`는 이후 같은 모델의 증류 초기값으로 사용하므로 보관하세요. 시험 평가는 이번 노트북에서 하지 않습니다.
