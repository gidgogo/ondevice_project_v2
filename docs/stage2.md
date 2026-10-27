# 2단계 실행 안내: 고성능 교사 → 증류 → 모델 선택

일반 학습 3종(`gt_random_v1`)이 끝난 뒤의 순서다. 코드는 준비되어 있고, **실제 GPU 학습 결과는 아직 없다.**

```text
① Retinexformer 교사 학습 (colab/04_teacher)        ── 일반 학습과 병행 가능
② 증류 학습 (colab/05_kd)  학생 × 교사 × β            ── Max+-GT 또는 ① 완료 후
③ 검증으로 크기별 선발 → Mini* / Max* / Max+*
④ 규칙 기반 선택기 맞추기 (scripts/selector.py fit)  ── 검증 69장
⑤ 설정 고정 → 시험 100장 평가 → selector.py apply    ── 한 번만
```

## ① 고성능 교사 학습 — `colab/04_teacher/train.ipynb`

- 우리 train 620쌍으로 Retinexformer를 처음부터 학습한다. 설정과 원 논문 설정의 차이는 [교사 문서](teacher_candidates.md).
- 점검 셀이 **반복당 시간과 TOTAL_ITERS 전체의 예상 시간**을 출력한다. 예상 시간을 보고 설정 셀의 `TOTAL_ITERS`를 고친 뒤 **설정 셀을 다시 실행**하고 본 학습을 시작한다(실행 셀마다 현재 설정으로 명령을 새로 만든다). 시작 후 바꾸면 새 EXPERIMENT.
- 1,000회마다 검증·저장, 끊기면 같은 셀로 이어감. 결과: `내 드라이브/ondevice/teachers/teacher_v1/retinexformer_s42/`
- 마지막 셀이 검증 69장 점수를 Max+ 일반 학습과 비교한다. 확실히 높지 않으면 고성능 교사로 쓰지 않는다.

## ② 증류 학습 — `colab/05_kd/train.ipynb`

설정 셀에서 `STUDENT`, `TEACHER_KIND`, `BETA`만 바꿔 실행한다. 한 실행 = 학생 1 × 교사 1 × β 1.

| 구성 | STUDENT | TEACHER_KIND |
|---|---|---|
| Mini-BaseKD / Max-BaseKD | mini / max | base (Max+-GT) |
| Mini-StrongKD / Max-StrongKD / Max+-StrongKD | mini / max / maxplus | strong (Retinexformer) |

- 학생은 **같은 모델의 일반 학습 `initial.pt`**에서 시작하고, 일반 학습의 `run.json` 설정과 같은지 노트북이 검사한다. 다르면 실행을 멈춘다.
- β 후보 0.1 / 0.3 / 1.0은 한 seed에서 검증으로 고르고, 고른 β와 일반 학습을 비교한다. 모든 증류 구성에 같은 탐색 기회를 준다.
- 교사 checkpoint 해시를 `run.json`에 기록하고, 다른 교사로 이어 학습하려 하면 거부한다.
- 결과: `runs/kd_random_v1/<학생>_<base|strong>kd_b<β>_s42/`. 마지막 셀이 같은 학생의 일반 학습과 검증 점수 차이를 출력한다.
- 다른 팀원이 학습한 파일을 쓰면 노트북에 적힌 Drive 경로로 복사한다(`initial.pt`, `run.json`, `summary.json`, 교사 `best.pt`).

## ③ 크기별 선발과 이미지별 결과

```bash
python scripts/eval_checkpoint.py --model mini --checkpoint runs/.../best.pt \
    --data-root data/raw/LOLv2/Real_captured --output reports/val/mini_star.json
```

학생·교사 모두 같은 평가 규약으로 이미지별 PSNR·SSIM을 저장한다(`--lpips` 선택). 시험 분할은 `--split test --allow-test`를 명시해야만 실행되며 ⑤에서 한 번만 쓴다.

## ④ 규칙 기반 선택기 — `scripts/selector.py`

어두운 사진일수록 비싼 모델을 쓰는 규칙(가설)을 검증 69장에서 맞춘다: 64×64 썸네일 평균 휘도 `f`가 `f < t1 → Max+*`, `t1 ≤ f < t2 → Max*`, 그 외 `Mini*`.

```bash
# 선택 비용 측정 — B가 M5에서. compute(읽은 사진에서 썸네일+통계+규칙)와 with_decode(PNG 읽기 포함)를 따로 출력
# 선택 시스템 시간 = 모델 end_to_end + compute (읽기는 end_to_end에 이미 포함). fit의 --selector-ms = compute 중앙값
python scripts/selector.py time --image <검증 저조도 PNG> --output reports/selector_time_m5.json
# 밝기 특징표 (C의 밝기 분석용)
python scripts/selector.py features --manifest manifests/lolv2_real/validation.json --data-root ... --output reports/val_features.csv
# 규칙 맞추기: 분위수 격자(기본 10단계)에서 Max+* 대비 PSNR 0.2 dB 이내인 가장 빠른 규칙
python scripts/selector.py fit --manifest manifests/lolv2_real/validation.json --data-root ... \
    --mini reports/val/mini_star.json --max reports/val/max_star.json --maxplus reports/val/maxplus_star.json \
    --latency reports/cpu_m5.json --selector-ms <time 결과> --output reports/selector_val
```

출력: `rule.json`(고정 규칙), `candidates.csv`(모든 후보의 품질·예상 시간·호출 비율 — 품질–시간 그래프용), `per_image.csv`(이미지별 특징·모델별 PSNR·선택 결과).
각 후보에는 **같은 호출 비율로 무작위로 골랐을 때의 기대 PSNR**(`random_mix_psnr`)과 그 대비 이득(`gain_over_random_db`)이 함께 기록된다. 이득이 0 이하이면 사진을 보고 고른 효과가 없다는 뜻이다.
`--reverse`(밝은 사진에 비싼 모델)나 다른 특징(`--feature`)을 시험하면 탐색 횟수가 늘어나므로 몇 개를 시험했는지 함께 보고한다.

## ⑤ 최종 시험

모든 설정(선발 모델, β, 규칙, 허용치)을 고정한 뒤 시험 100장을 각 모델로 한 번 평가하고, 같은 규칙을 그대로 적용한다.

```bash
python scripts/selector.py apply --manifest manifests/lolv2_real/test.json --data-root ... \
    --mini reports/test/mini_star.json --max ... --maxplus ... --rule reports/selector_val/rule.json --output reports/selector_test.json
```

시험 결과를 보고 규칙이나 모델을 바꾸지 않는다. 기준 미달이면 그대로 보고한다. 선택 시스템의 실제 시간은 세 모델을 메모리에 올린 상태로 목표 M5에서 따로 측정한다([측정 규약](measurement_protocol.md)).
