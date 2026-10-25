# 중간발표용 LOL-v2-real 결과 (2026-10-26)

Colab T4에서 끝난 일반 학습 3개, Max+ 교사 증류 2개, 진행 중인 Retinexformer 교사(18000/30000회)의 **검증 69장** 결과다. 시험 100장은 쓰지 않았다.
원자료는 각 실행 폴더(`내 드라이브/ondevice/runs/...`)의 `log.csv`, `summary.json`, `best_validation_per_image.json`이고, 이 폴더의 표는 그것을 그대로 옮기거나 합친 것이다.

| 실행 | 최고 검증 PSNR / SSIM | 최고 시점 |
|---|---|---|
| Mini 일반 (`gt_random_v1/mini_gt_s42`) | 10.5542 dB / 0.3213 | 7 epoch |
| Max 일반 (`gt_random_v1/max_gt_s42`) | 13.5340 dB / 0.3822 | 4 epoch |
| Max+ 일반 (`gt_random_v1/maxplus_gt_s42`) | 17.1252 dB / 0.6199 | 78 epoch |
| Mini 증류, 교사 Max+, β=0.3 (`kd_random_v1/mini_basekd_b0.3_s42`) | 10.5455 dB / 0.3195 | 16 epoch |
| Max 증류, 교사 Max+, β=0.3 (`kd_random_v1/max_basekd_b0.3_s42`) | 13.5411 dB / 0.3807 | 4 epoch |
| Retinexformer 교사, 학습 중 (`teachers/teacher_v1/retinexformer_s42`) | 21.9929 dB / 0.7827 | 9000회 |

증류 2개는 같은 모델 일반 학습의 `initial.pt`(SHA-256 일치)와 같은 설정에서 시작했고, 교사 checkpoint SHA-256은 `59b626ec…`(Max+ 일반 학습 best.pt)다.

## 누가 무엇을 쓰나

| 받는 사람 | 파일 | 내용 |
|---|---|---|
| A | `quality/*_validation_quality_lolv2.csv` (6개), `quality_summary_lolv2.csv` | A 묶음과 같은 8개 열 품질 CSV, 모델당 69행. `baseline_a/team_a_tools/lolv2_tables.py quality`로 생성 |
| B | `training_time.csv` | 실행별 GPU, 학습·검증 시간(분), epoch당 시간 → `training_budget.md`용 |
| C | `validation_per_image_all_models.csv` | 사진 1장당 1행: 모든 모델의 PSNR·SSIM, `max_minus_mini_psnr`, `maxplus_minus_max_psnr`. 입력 밝기 특징은 `scripts/selector.py features`로 따로 계산해 `name`으로 연결 |
| D / 팀장 | `training_history.csv`, `distillation_comparison.csv`, `fig_*.png`, `slide7_draft.md` | 학습 곡선 원자료, 증류 비교표, 그림 2장, 7장 초안 |
| 팀장 · B | `weights/*.pt` | 학생 5개 best.pt (로컬 시연·M5 속도 측정용) |

## 다시 만들기

```bash
python3 baseline_a/team_a_tools/lolv2_tables.py quality --manifest-csv baseline_a/split_manifest_lolv2.csv --split validation \
    --results <Drive>/runs/gt_random_v1/max_gt_s42/best_validation_per_image.json \
    --model-id UltraFastLiNET_Max --checkpoint-id max_gt_random_v1_s42_best --output max_gt_validation_quality_lolv2.csv
```

## 주의

- 모든 값은 seed 42 한 번의 결과다. 증류는 β=0.3 하나만 해 봤다.
- Retinexformer는 학습이 끝나지 않았다. 표의 값은 9000회 시점 best.pt이며, 이후 학습을 이어가면 바뀔 수 있다.
- `training_time.csv`의 시간은 로그의 학습·검증 시간 합계다. Colab 연결, 데이터 압축 해제, 모델 불러오기 시간은 빠져 있다.
