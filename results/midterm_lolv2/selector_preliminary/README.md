# 선택기 예비 계산 (확정 결과 아님)

- 입력: C의 `brightness_analysis.csv`의 `in_mean`(selector.py와 같은 64×64 썸네일 평균 휘도), 크기별 선발 모델(Mini 일반 / Max 증류(교사 Max+) / Max+ 증류(교사 Retinexformer))의 검증 69장 사진별 PSNR (`../validation_per_image_all_models.csv`)
- 방식: `selector.py fit`과 같은 분위수 격자(10단계), 계획 방향과 반대 방향 모두
- **시간은 임시 값**: 테스트 서버 CPU의 end_to_end 중앙값(Mini 18 / Max 49.6 / Max+ 62 ms), 선택 비용 0.5 ms. M5가 아니다.
- **같은 69장으로 규칙을 맞추고 평가**했으므로 실제보다 좋게 나올 수 있다.

| 품질 허용치 | 고른 규칙 | PSNR (Max+ 17.134 대비) | 예상 시간 | 같은 비율 무작위 대비 |
|---|---|---|---|---|
| 0.2 dB | in_mean 상위 약 10%(7장, 장면 3개) → Mini | 17.067 | 58.0 ms (약 6%↓) | +0.60 dB |

전체 후보: `selector_preliminary_candidates.csv`, 그림: `fig_selector_preliminary.png`. 다시 만들기: `python selector_prelim.py <brightness_analysis.csv>`
