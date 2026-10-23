# Max+ 일반 학습 담당

[Colab에서 열기](https://colab.research.google.com/github/gidgogo/ondevice_project_v2/blob/main/colab/03_maxplus/train.ipynb) · [팀 공통 안내](../README.md)

- 모델: **Max+**, 학습 파라미터 **937개**
- 노트북: [train.ipynb](train.ipynb)
- 준비 데이터: Drive의 `내 드라이브/ondevice/LOLv2.zip`
- 결과: `내 드라이브/ondevice/runs/gt_random_v1/maxplus_gt_s42/`
- 환경 기록: `내 드라이브/ondevice/environments/gt_random_v1/maxplus_s42/`
- 공통 설정: seed 42 / 120 epoch / batch 40 / 학습 620쌍 / 검증 69쌍

Max+는 프로젝트에서 추가한 모델이며 Max를 먼저 학습할 필요가 없습니다.

완료 후 결과 폴더와 환경 기록을 팀에 전달하세요. `initial.pt`는 이후 같은 모델의 증류 초기값으로 사용하므로 보관하세요. 시험 평가는 이번 노트북에서 하지 않습니다.

기본값 EPOCHS_PER_SESSION=120으로 남은 학습을 끝까지 진행하며 매 epoch 저장합니다. 짧은 점검 때만 5~10으로 줄이세요. 중단 후 같은 본 학습 셀을 실행하면 저장 지점부터 이어집니다. 전체 목표 120 epoch와 실험 이름은 그대로 두세요. 자세한 복구 방법은 [공통 안내](../../docs/colab.md)를 참고하세요.
