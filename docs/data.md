# 데이터: LOL-v2-real

## 출처

- 파일: `LOLv2.zip` (1,047,872,805 bytes, SHA-256 `298e00fece107b64fb54769ae8bc5175b4f84a3f7e8134e651a0a4b5eb61198d`)
- 배포처: [Retinexformer README](https://github.com/caiyuanhao1998/Retinexformer)의 LOL-v2 Google Drive 링크. 원 저자 저장소([SGM-Low-Light](https://github.com/flyywh/SGM-Low-Light))에는 데이터셋 링크가 없다.
- 사용 부분: `LOLv2/Real_captured` (Synthetic은 사용하지 않음). 압축 해제 위치 `data/raw/LOLv2/` (git 제외)

## 점검 결과 (`scripts/audit_lolv2.py`)

| 항목 | 결과 |
|---|---|
| 쌍 개수 | Train 689 / Test 100, 저조도·정상 파일명 1:1 대응 |
| 형식 | 전부 600×400 RGB PNG |
| 정확히 같은 파일을 공유하는 그룹 | 104개 (모두 Train 내부). 노출만 다른 저조도 사진들이 같은 정상 사진을 공유 |
| 같은 장면 그룹 (정상 사진 16×16 평균 해시) | 기준 6 / 12 / 20 / 28에서 233 / 237 / 241 / 239개. 그룹은 모두 연속 번호(같은 장면의 여러 노출) |
| Train–Test 장면 겹침 | 모든 기준에서 0 |

원본 리포트: `reports/data/lolv2_real_audit.json`

참고: LOL-v2-real에는 LOL-v1 장면(예: 책장·고양이 그림)이 포함되어 있다. LOL-v1으로 사전학습된 외부 가중치를 교사로 쓰면 장면을 미리 봤을 수 있으므로 교사는 우리 학습 분할로 직접 학습한다.

## 분할 (`scripts/prepare_lolv2.py`, `manifests/lolv2_real/`)

| 분할 | 쌍 | 장면 그룹 | 용도 |
|---|---:|---:|---|
| train | 620 | 227 | 학습 |
| validation | 69 | 29 | checkpoint·β·교사·선택 기준 결정 |
| test | 100 | — | 설정 고정 후 최종 평가 1회 |

- 공식 Train 689쌍에서 검증 69쌍(10%)을 분리했다. seed 42.
- 정확히 같은 파일을 공유하거나 정상 사진의 해시 거리가 20 이하인 쌍은 같은 장면 그룹으로 묶고, 그룹 단위로만 나눴다.
- manifest SHA-256: train `88503f98…`, validation `ec2a263c…`, test `fe28cc6e…` (`manifests/lolv2_real/summary.json`)
- 분할 스크립트는 시험 이미지를 디코딩하지 않는다. 시험 이미지는 데이터 점검(형식·장면 겹침)에만 디코딩했고 모델을 실행하지 않았다.
