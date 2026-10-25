# 팀원 A 전달 가이드

**이 묶음은 LOL-v1에서 Max 베이스라인을 평가한 결과와, 다음 팀원이 같은 기준으로 작업하기 위한 자료입니다.**

> **2026-10-24 확인 사항 (D 검토로 추가)** — 이 묶음의 결과는 **LOL-v1 예비 기록**입니다(train 435 / validation 50 / test 15, 중앙 crop, 기존 Max 재현).
> 현재 팀 실험 기준은 GitHub [`gidgogo/ondevice_project_v2`](https://github.com/gidgogo/ondevice_project_v2)의 **LOL-v2-real**(train 620 / validation 69 / test 100, random crop 180, 교사 = Max+ 일반 학습·Retinexformer)입니다.
> 평가 함수(`common_metrics.py`)와 8개 열 CSV 형식은 그대로 이어서 씁니다. 바뀐 위치와 이유는 [`수정내역.md`](수정내역.md)에 있습니다.

현재 포함된 결과는 validation 50장·test 15장의 PSNR·SSIM, 시험 복원 PNG 15장입니다. 학습된 모델은 Max 한 조건이며 Mini·Max+ 결과는 아직 포함되어 있지 않습니다.

| 평가 집합 | 이미지 수 | 평균 PSNR(dB) | 평균 SSIM |
|---|---:|---:|---:|
| validation | 50 | 16.0612 | 0.515624 |
| test | 15 | 17.9173 | 0.527209 |

이 가이드에는 파일 설명, 원본 데이터 준비, 담당자별 작업 순서, 재실행 명령, 발표 초안을 모았습니다. 고정 수치와 전체 설정은 `experiment_settings.md`를 기준으로 합니다.

## 1. 내가 먼저 봐야 하는 파일

| 담당 | 먼저 읽을 자료 | 바로 할 일 |
|---|---|---|
| C | `split_manifest.csv`, 두 품질 CSV, `enhanced_manifest.csv` / LOL-v2: `split_manifest_lolv2.csv` | 원본 low에서 밝기 계산 → ID로 점수 연결 → 비교 이미지 제작. 중간발표 분석은 LOL-v2-real 검증 69장 기준(8절) |
| D | 8절, 저장소 `docs/stage2.md` | LOL-v2-real 일반 학습(Mini·Max·Max+) 결과로 Mini 일반 학습 vs Max+ 교사 증류 비교. 이 묶음의 LOL-v1 Max `best.pt`는 교사로 쓰지 않음 |
| B | `experiment_settings.md`, 저장소 `docs/measurement_protocol.md` | 같은 CPU 조건에서 Mini·Max·Max+ 시간 측정(저장소 `scripts/benchmark_cpu.py`) |
| 팀장 | `quality_summary.csv`, 이 문서 9절 | 평균과 N을 발표에 반영하고 C의 비교 이미지를 추가 |
| A·재현 담당 | `team_a_tools/`, `evidence/`, 이 문서 7절 | 전달 파일 검사 또는 같은 가중치로 평가 재실행 |

C가 기존 점수로 분석을 시작하거나 팀장이 표를 쓰는 데 모델을 다시 학습할 필요는 없습니다. D의 Mini 학습, B의 속도 측정, C의 밝기 분석은 별도 후속 작업입니다.

## 2. 폴더를 이해하는 방법

정리본은 총 **31개 파일**입니다. 문서·CSV·가중치·코드 16개와 복원 PNG 15개가 있습니다. 2026-10-24에 LOL-v2-real 명부(`split_manifest_lolv2.csv`), 변환 도구(`team_a_tools/lolv2_tables.py`), 수정 기록(`# 가이드라인/수정내역.md`)과 원본 백업(`# 가이드라인/_원본_20261024/`)이 추가되었습니다.

| 위치 | 내용 | 언제 여는가 |
|---|---|---|
| 최상위 폴더 | 안내·실험 설정·CSV·가중치·공통 지표·설치 목록 | 평소 작업할 때 |
| `enhanced/test/` | 시험 이미지 15장의 복원 결과 | C가 시각 비교할 때 |
| `evidence/` | 원 평가 JSON 3개 | 점수 출처·환경·해시를 자세히 확인할 때 |
| `team_a_tools/` | 전달물 검사, 검증 평가 도구 2개 | 파일 검사 또는 재평가할 때 |

원본 low/high 이미지, 모델 구조 전체 코드, Mini/Max+ 가중치는 이 ZIP에 없습니다. 원본 데이터는 5절, 전체 코드는 7절의 고정 저장소에서 준비합니다.

## 3. 파일별 상세 설명

### 3-1. `README.md` — 전체 사용 설명서

지금 읽는 파일입니다. 누가 무엇을 해야 하는지와 파일 사용 순서가 있습니다. 기존의 여러 안내문, 데이터 위치 설명, 3·4장 발표 초안을 이 문서로 합쳤습니다.

### 3-2. `experiment_settings.md` — 팀 공통 기준

모델명, 가중치 해시, 학습 이력, 데이터 분할, 전처리, 평가 기준, 실제 보고서의 환경이 있습니다. 별도 `baseline_a.json` 파일에 있던 학습 설정도 JSON 블록으로 보존했습니다.

동일 조건 비교를 할 때 이 문서를 기준으로 맞춥니다. 설정을 바꾸는 후속 실험은 원래 기준을 덮어쓰지 말고 새로운 모델/실험 설정으로 기록합니다.

### 3-3. `split_manifest.csv` — 전체 이미지 명부

총 500행입니다. 한 행은 저조도 입력 한 장과 그에 대응하는 정답 한 장, 즉 이미지 한 쌍입니다. 두 파일이 존재한다고 평가 표본을 2개로 세지 않습니다.

| 열 | 뜻 | 예시 |
|---|---|---|
| `image_id` | 팀 전체에서 사용할 고유 ID | `LOL-v1/1.png` |
| `split` | 학습·검증·시험 구분 | `train`, `validation`, `test` |
| `source_name` | 원래 이미지 파일명 | `1.png` |
| `low_path` | 데이터 루트 기준 저조도 입력 경로 | `eval15/low/1.png` |
| `gt_path` | 데이터 루트 기준 정상 조도 정답 경로 | `eval15/high/1.png` |
| `width`, `height` | 가로·세로 픽셀 수 | `600`, `400` |

train 435행, validation 50행, test 15행을 그대로 사용합니다. C는 파일 경로를 찾고, D는 학습·검증 목록을 구분할 때 씁니다. 같은 seed를 넣어 임의로 다시 나눈 목록이 기존 목록과 같다고 가정하지 않습니다.

### 3-4. `baseline_quality.csv` — 시험 15장 점수

test 15장의 Max 결과입니다. 파일 이름에 test가 없어도 **이 파일은 시험 집합 결과**입니다. 좋은 사진 몇 장만 남긴 표가 아닙니다.

| 열 | 뜻 |
|---|---|
| `image_id` | 명부·밝기·복원 이미지와 연결할 키 |
| `split` | 이 파일에서는 `test` |
| `model_id` | `UltraFastLiNET_Max` |
| `checkpoint_id` | `a_max_seed42_epoch20` |
| `width`, `height` | 실제 평가 크기, 가로 600·세로 400 |
| `psnr_db` | 해당 이미지 한 장의 PSNR, 단위 dB |
| `ssim` | 해당 이미지 한 장의 SSIM |

PSNR은 정답과의 픽셀 오차, SSIM은 구조적 유사성을 평가합니다. 일반적으로 높을수록 좋지만 서로 같은 기준은 아니므로 순위가 다를 수 있습니다. 이미 저장된 숫자는 기존 float 평가 결과이며 PNG를 다시 읽어 계산한 값으로 교체하지 않습니다.

### 3-5. `baseline_validation_quality.csv` — 검증 50장 점수

열의 의미는 시험 CSV와 같고 `split=validation`인 50행입니다. C가 밝기 그룹의 경계를 정하거나 D가 검증 성능을 비교할 때 씁니다. 시험 15장과 합쳐서 ‘시험 65장 평균’으로 보고하지 않습니다.

### 3-6. `quality_summary.csv` — 평균과 평가 수

검증·시험 두 행으로 된 요약입니다. `count`는 실제 평가 이미지 수, `psnr_db_mean`과 `ssim_mean`은 각 집합의 이미지별 점수 산술평균입니다. `score_source`는 원 보고서의 출처 라벨입니다.

팀장은 여기의 평균과 N을 발표에 옮깁니다. 표를 간단히 표시할 때 반올림하되 원 CSV의 정밀도는 유지합니다. 출처 라벨만으로 실제 추론 실행 시각이 입증되는 것은 아닙니다.

### 3-7. `enhanced_manifest.csv` — 복원 PNG 연결표

test 15개의 `image_id`, `split`, `source_name`, `enhanced_path`가 있습니다. `enhanced_path`는 **이 전달 폴더 기준 상대 경로**입니다. 원본 이미지 경로의 기준인 데이터 루트와 다릅니다.

예를 들어 `LOL-v1/1.png`의 복원 파일은 `enhanced/test/01_1.png`입니다. `01_`은 저장 순번이므로 ID로 사용하지 않습니다. C는 CSV를 ID로 연결해 PNG 위치를 찾습니다.

### 3-8. `enhanced/test/` — 복원 이미지 15장

Max가 시험 저조도 입력을 보정한 결과입니다. 정답이 아니며, 원본 저조도 입력도 아닙니다. C가 입력·복원·정답을 나란히 비교할 때 사용합니다.

검증 50장의 복원 PNG는 포함하지 않았습니다. 검증 점수는 존재하지만 검증 시각 자료가 필요하면 같은 가중치로 추가 생성해야 합니다. `verify_validation.py`에는 현재 PNG 저장 옵션이 없습니다.

각 PNG의 ID와 원본 입력·정답 경로는 10절에 정리했습니다.

### 3-9. `best.pt` — 실제 Max 가중치

학습된 파라미터 값이 저장되어 있습니다. 텍스트 편집기로 읽는 문서가 아니라 PyTorch 모델에 로드하는 파일입니다. 모델 구조는 고정 저장소 코드에서 가져옵니다.

360 epoch 학습 기록 중 validation PSNR이 가장 높았던 epoch 20의 가중치입니다.

파일 내부 가중치 키는 `weight`입니다. Mini에 그대로 로드하는 가중치가 아닙니다. 이 가중치는 LOL-v1으로 학습되었으므로 현재 LOL-v2-real 증류의 교사로 쓰지 않습니다(교사는 LOL-v2-real로 학습한 Max+ 일반 학습 모델과 Retinexformer).

### 3-10. `common_metrics.py` — 공통 채점 함수

원 저장소의 `vendor/metrics.py`와 같은 PSNR·SSIM 함수입니다. 입력은 한 장씩 RGB float32 `[1,3,H,W]`로 맞춥니다. 입력·정답은 [0,1], 예측은 [0,1]로 clamp한 값을 사용합니다.

이 파일만 실행한다고 모델 추론이나 CSV 저장이 자동으로 되지는 않습니다. 호출부에서 모델 실행, 최종 출력 선택, 전처리, 이미지별 기록을 구현해야 합니다.

```python
# 평가 호출 예시. 모델과 입력/정답 텐서를 먼저 준비해야 합니다.
import torch
from common_metrics import psnr, ssim

model.eval()
with torch.inference_mode():
    pred = model(x)[-1].clamp(0, 1)  # 고정 원 모델은 마지막 출력이 최종 영상
    assert pred.shape == target.shape
    image_psnr = psnr(pred, target)
    image_ssim = ssim(pred, target)
```

여러 장을 한꺼번에 넣어 PSNR을 한 번 구하면 이미지별 PSNR 평균과 달라질 수 있습니다. 평가 batch size는 1로 유지합니다. 선택적 LPIPS 함수도 들어 있지만 이번 결과에서는 LPIPS를 측정하지 않았습니다.

### 3-11. `requirements.lock.txt` — 설치 버전 목록

PyTorch·torchvision·NumPy·Pillow·pytorch-msssim 등 30개 패키지의 버전을 고정한 파일입니다. 같은 환경을 준비할 때 `pip install -r requirements.lock.txt`로 사용합니다.

이 파일이 Python 인터프리터 버전이나 CPU·OS까지 설치해 주지는 않습니다. 실제 보고서의 Python 버전은 3.14.2입니다. 기존 저장소의 기준 환경 3.12.6과 구분하는 방법은 7절에 있습니다.

### 3-12. `LICENSE` — 라이선스 고지

함께 전달하는 코드 등의 원 저장소 라이선스 고지입니다. 코드가 포함된 묶음이므로 보존합니다. 원본 데이터의 별도 이용 조건까지 대신하는 파일은 아닙니다.

### 3-13. `evidence/test_metrics.json` — 시험 결과의 상세 근거

시험 15장의 점수, 평균, 실제 평가 크기, 지표 설정, 환경, 가중치와 데이터 목록의 해시가 있습니다. CSV는 분석에 편하게 만든 표이고 이 JSON은 결과의 조건까지 확인할 수 있는 상세 기록입니다.

값이 의심스럽거나 다른 사람이 재평가했을 때 달라지면 여기서 가중치·환경·전처리를 비교합니다. 이번 정리에서는 원본 내용 그대로 이동했습니다.

### 3-14. `evidence/validation_metrics.json` — 검증 결과의 상세 근거

검증 50장에 대한 같은 종류의 보고서입니다. 검증 CSV와 이미지별 점수·평균이 일치합니다. D의 후속 검증과 현재 Max 결과의 출처 확인에 사용합니다. 이번 정리에서는 원본 내용을 바꾸지 않았습니다.

### 3-15. `evidence/recheck_status.json` — 원 전달물의 출처 기록

고정 저장소 commit, checkpoint ID, 가중치·공통 지표·설치 목록 해시, 보고서 출처와 환경이 있습니다.

`generated_at`은 **원래 제출 묶음의 생성 시각**입니다. 실제 평가 시작·종료 시각이나 이 정리본의 생성 시각이 아닙니다. 안에 남아 있는 `runs/a_member_a_*`는 원 실행의 출처 경로이므로 이 ZIP 내부 경로로 해석하지 않습니다.

### 3-16. `team_a_tools/check_handoff.py` — 전달 파일 검사

누락 파일, 500개 ID, 435/50/15 분할, 품질 CSV의 8개 열, CSV/JSON 일치, 평균, 가중치 해시, PNG 경로·헤더 크기를 검사합니다. 정리본의 `evidence/` 경로와 통합 문서에 맞춰 수정한 검사기입니다.

Python 표준 라이브러리만 사용하며 모델을 추론하지 않습니다. 원본 데이터 접근이나 실제 평가 시각도 확인하지 않습니다. PNG는 헤더만 검사하므로 모든 픽셀의 복원 정확성을 검사하는 도구도 아닙니다.

전달 폴더 안에서 실행합니다.

```bash
python3 team_a_tools/check_handoff.py .
```

`PASS`, manifest 500, validation 50, test 15, enhanced 15가 나오면 파일 검사 조건을 만족합니다. 기존 버전의 `--require-data-location` 옵션은 이 정리본에서 사용하지 않습니다. 원본 데이터 위치는 이 README의 5절로 통합했습니다.

### 3-17. `team_a_tools/verify_validation.py` — 검증 재평가

Max를 검증 50장에 실제로 추론하고 `metrics.json`, `per_image.csv`를 만듭니다. 고정 저장소의 모델·데이터 로더·평가 함수를 호출하므로 **전체 저장소와 원본 데이터가 필요합니다.**

이 파일을 전달 폴더에서 단독 실행하는 방식이 아닙니다. 7절대로 고정 저장소 루트에 `team_a_tools/`를 복사한 뒤 실행합니다. 원본 도구를 그대로 보존했으며 출력 PNG를 저장하지는 않습니다.

## 4. 이번에 제외하거나 합친 파일

원본 ZIP 38개에서 기존 8개 파일을 별도 파일로 두지 않고, 통합 안내문 `README.md` 1개를 추가하여 31개가 되었습니다. 핵심 점수·가중치·지표 코드·원 평가 JSON·복원 PNG는 그대로 보존했습니다.

| 원래 파일 | 정리 방법 | 이유 |
|---|---|---|
| `data_location.txt.bak` | 제외 | 오래된 백업이며 수신자 작업에 필요 없음 |
| `environment_packages.txt` | 중복 파일 제외 | 패키지명 표기를 정규화하면 lock과 30개 버전 모두 일치. 환경 정보는 설정·JSON에 유지 |
| `team_a_tools/finalize_handoff.py` | 수신자 묶음에서 제외 | A가 결과를 포장할 때 쓰는 도구. 다음 팀원의 분석·재평가에 필요 없음 |
| `HANDOFF_README.md` | 이 README로 통합 | 담당자별 안내를 한곳에서 읽도록 함 |
| `REPRODUCE.md` | 이 README 7절로 통합 | 재실행 순서를 한곳에 모으고 Python 환경 차이를 설명 |
| `data_location.txt` | 이 README 5절로 통합 | 데이터 경로와 현재 상태를 정리하고 오래된 미완료 문구를 바로잡음 |
| `slides_3_4_draft.md` | 이 README 9절로 통합 | 발표용 내용은 보존하면서 작은 문서 수를 줄임 |
| `baseline_a.json` | `experiment_settings.md`에 전체 JSON 보존 | 동일 설정은 고정 저장소에도 존재하며 전달 폴더의 중복 파일을 줄임 |

`evidence/`의 세 JSON은 CSV와 일부 내용이 겹쳐도 환경·가중치·출처 근거가 있으므로 남겼습니다. 복원 PNG도 C의 비교 작업에 필요하므로 15장 모두 남겼습니다. 파일 수를 줄이는 것이 목적이며 PNG가 용량 대부분을 차지해 ZIP 용량은 크게 줄지 않습니다.

이 정리본은 새 결과 묶음입니다. A는 원본 ZIP을 별도로 보관하면 이전 포장기나 원래 문서를 다시 확인할 수 있습니다.

## 5. 원본 데이터 준비와 현재 상태

### 원본 데이터 위치

- 공식 출처: https://daooshee.github.io/BMVC2018website/
- 데이터셋: LOL-v1.
- 별도 팀 공유 링크: 원 제출물에 등록되어 있지 않음.
- 원본 low/high는 이번 ZIP에 포함하지 않음. 수신자는 공식 출처에서 준비하거나 팀 내 보유 원본의 접근 위치를 확인한다.

예를 들어 데이터 루트를 `data/LOL`로 정했다면 다음 폴더가 있어야 합니다.

| 실제 폴더 | 파일 수 | 사용 |
|---|---:|---|
| `data/LOL/our485/low` | 485 | train·validation 저조도 입력 |
| `data/LOL/our485/high` | 485 | train·validation 정답 |
| `data/LOL/eval15/low` | 15 | test 저조도 입력 |
| `data/LOL/eval15/high` | 15 | test 정답 |

`our485`를 전부 학습에 사용하지 않습니다. 고정 목록에서 435쌍은 train, 50쌍은 validation입니다.

### 경로 기준 두 가지

| CSV 열 | 기준 폴더 | 예시 |
|---|---|---|
| `low_path`, `gt_path` | 수신자가 준비한 데이터 루트 | `data/LOL` + `eval15/low/1.png` |
| `enhanced_path` | 이 전달 폴더 | `team_a_delivery_clean` + `enhanced/test/01_1.png` |

이 두 기준을 혼동하면 파일을 못 찾게 됩니다. CSV 자체에는 원본 이미지 픽셀이 들어 있지 않습니다.

### 현재 무엇이 확인되었는가

제출물에는 검증 50장·시험 15장의 개별 결과, 집합별 평균, 시험 복원 PNG 15장이 있습니다. 파일 검토에서 평가 목록·평균·CSV/JSON 일치를 확인했습니다. 이번 정리에서도 원본과 핵심 파일의 바이트 일치를 확인합니다.

원 문서의 평가 실행자 기재명은 해민입니다. 정확한 추론 시작·종료 시각은 제출물에 별도 기록되어 있지 않습니다. 원래 묶음 생성 시각 `2026-10-23 18:37:36 +09:00`을 추론 시각으로 대신 적지 않습니다. 실제 실행 로그가 있으면 A가 별도로 보완할 수 있습니다.

이번 정리는 모델을 새로 학습하거나 원본 low/high로 다시 추론하는 작업이 아닙니다. 수신자의 데이터 다운로드 가능 여부도 각자 확인해야 합니다.

## 6. 담당자별 다음 작업

### C: 밝기 분석과 대표 이미지

**현재 업무 (LOL-v2-real 검증 69장, 중간발표용)**

1. LOL-v2-real 원본(`LOLv2.zip`의 `Real_captured`)을 준비하고 `split_manifest_lolv2.csv`의 `low_path`·`gt_path`로 열 수 있는지 확인합니다.
2. 모델별 검증 품질 CSV(8절 변환 도구로 생성)를 `image_id`로 검증 69행과 일대일 연결합니다. 시험 100장은 설정을 모두 고정한 뒤 최종 평가 때만 씁니다.
3. 원본 low에서 평균 밝기와 어두운 픽셀 비율을 계산하고 공식을 기록합니다. 저장소 `scripts/selector.py features`를 쓰면 선택기와 같은 정의(64×64 썸네일 휘도 통계)로 계산됩니다. 복원 결과나 정답의 밝기를 입력 밝기로 쓰지 않습니다.
4. 밝기 그룹의 경계는 검증 69장에서 정하고 그룹별 N·평균 PSNR·평균 SSIM을 제시합니다.
5. Mini·Max·Max+ 결과가 나오면 같은 ID끼리 `Max PSNR - Mini PSNR` 등 모델 간 차이를 계산하고, 입력·복원·정답에 같은 crop 위치·확대율·표시 범위를 적용해 비교 이미지를 만듭니다. 검증 복원 이미지는 학습된 checkpoint로 새로 생성해야 합니다.
6. 관계가 약하거나 오류가 뚜렷하지 않으면 그대로 기록합니다. 모델 한 개의 사진별 성능 차이만으로 라우터 효과를 입증했다고 말하지 않습니다.

C가 돌려줄 자료: `brightness_analysis.csv`(LOL-v2-real 검증 69행), 산점도, 그룹별 N·평균, 비교 이미지 3세트, 관찰 내용과 선정 이유.

**LOL-v1 재평가용 (이 묶음의 기존 절차, 참고)**

LOL-v1 결과로 연습하거나 이 묶음을 재검토할 때만 씁니다. LOL-v2-real 결과와 한 표에서 섞지 않습니다.

1. 원본 low/high를 준비하고 `split_manifest.csv`의 경로로 열 수 있는지 확인합니다.
2. validation 명부 50행과 검증 품질 50행을 `image_id`로 일대일 연결합니다. 시험은 15행끼리 별도로 연결합니다.
3. 밝기 그룹의 경계는 검증 자료에서 정하고 시험에도 같은 값을 적용합니다.
4. 시험 비교 이미지는 `enhanced_manifest.csv`로 PNG 위치를 찾습니다.

LOL-v1 시험 사례 후보(PSNR 기준):

| 성격 | image_id | PSNR | SSIM |
|---|---|---:|---:|
| 낮은 점수 | `LOL-v1/179.png` | 10.581779 | 0.429516 |
| 중앙 순위 | `LOL-v1/1.png` | 18.944241 | 0.695367 |
| 높은 점수 | `LOL-v1/79.png` | 23.317739 | 0.570257 |

### D: Mini 학습·증류와 공통 검증

현재 기준(LOL-v2-real, 저장소 `gidgogo/ondevice_project_v2`)으로 바꿨습니다. 이전 LOL-v1 안내는 `_원본_20261024/README.md`에 보존했습니다.

1. train 620 / validation 69 / test 100의 고정 분할(저장소 `manifests/lolv2_real/`)을 유지합니다. test로 학습 설정·β·교사를 고르지 않습니다.
2. Mini 일반 학습과 증류 학습은 **같은 `initial.pt`**에서 시작하고 데이터·학습량·평가 조건을 맞춥니다. 저장소 `colab/05_kd` 노트북이 일반 학습의 `initial.pt`·설정 일치를 검사합니다.
3. 기본 교사는 LOL-v2-real로 학습한 **Max+ 일반 학습 모델**(`gt_random_v1/maxplus_gt_s42/best.pt`), 고성능 교사는 **Retinexformer**입니다. 교사는 고정하고 학생과 같은 random 패치에서 출력을 계산합니다. 이 묶음의 LOL-v1 Max `best.pt`는 쓰지 않습니다.
4. 학습 crop은 **random 180×180**입니다. 128×128은 기본 MS-SSIM 손실(crop > 160 필요)에 맞지 않습니다.
5. 검증은 crop이 아닌 원본 전체 해상도에서 수행합니다. 이 폴더 `common_metrics.py`는 저장소 **`vendor/metrics.py`와 같은 파일**(SHA-256 `ec6420c5…` 일치)이고, 저장소 `scripts/evaluation.py`가 이 함수를 호출해 같은 규약(RGB float32 [0,1], 예측만 clamp, 이미지별 평균)으로 채점합니다.
6. 평가 결과는 학습 폴더의 `best_validation_per_image.json` 또는 저장소 `scripts/eval_checkpoint.py` 보고서를 `team_a_tools/lolv2_tables.py quality`로 8개 열 CSV로 바꿔 A·C에게 전달합니다.

D가 돌려줄 자료: 모델별 `best.pt`·`initial.pt`, 저장소 코드 commit, 학습 설정(`run.json`)·실제 학습량(`summary.json`, `log.csv`), 선택 epoch·기준, 실행 GPU, 검증 69장 8개 열 CSV. 일반·증류 checkpoint ID를 구분합니다(예: `mini_gt_random_v1_s42_best`, `mini_basekd_b0.3_s42_best`).

### B: CPU 시간 측정

같은 컴퓨터에서 Mini·Max에 동일한 입력 크기 600×400, batch 1, dtype·스레드 조건을 적용합니다. 준비 실행과 반복 측정을 하고 횟수를 기록합니다. 순수 모델 forward 시간(model)과 전체 시간(end_to_end: PNG 읽기 → 텐서 변환 → 추론 → clamp → uint8 출력 배열)을 구분합니다. 저장소 측정 규약과 같이 **디스크 저장·화면 표시는 전체 시간에서 제외**하고, 모델 로딩도 포함하지 않습니다.

현재 계획의 비교 대상은 Mini·Max·**Max+(937개)** 세 모델이며, 저장소 `scripts/benchmark_cpu.py`와 `docs/measurement_protocol.md`(목표 M5, 1스레드, warm-up 10·측정 100)를 따릅니다. 시간은 가중치 값과 무관하므로 학습 완료 전에도 측정할 수 있습니다. 학습 전 Mini는 ‘구조 / 학습 전 / 속도 측정용’이라고 적습니다. A의 M3 평가 환경과 다른 장치를 사용해도 되지만 B의 Mini·Max 비교는 같은 장치에서 수행합니다. 파라미터 수만으로 실제 속도나 메모리 개선을 확정하지 않습니다.

### 팀장: 범위와 발표 통합

중간발표의 정량 결과는 **LOL-v2-real 검증 69장** 기준으로 통합합니다. A의 모델별 검증 품질 CSV(8절), C의 밝기 분석·비교 이미지, B의 M5 시간 측정, D의 일반 학습·증류 비교를 같은 이미지 ID·같은 지표로 모읍니다. 실제로 완료된 결과와 아직 측정하지 않은 내용을 구분해 표시합니다.

이 묶음의 LOL-v1 수치(9절)를 쓴다면 “LOL-v1 예비 재현, 단일 seed, 시험 N=15”로 따로 표시하고 LOL-v2-real 결과와 한 표에서 비교하지 않습니다. 증류 효과, 라우터 효과, LOL-v2-real 성능은 이 ZIP만으로 확인한 결과가 아닙니다. 학습 범위와 일정은 실제 Colab 점검(epoch당·반복당 시간)을 보고 정합니다.

## 7. 다른 팀원이 평가를 다시 실행하는 방법

아래는 macOS/Linux 터미널 기준이며, 기존 결과를 다시 학습하는 절차가 아니라 **같은 가중치를 다시 채점하는 절차**입니다.

### 7-1. 먼저 전달 파일을 검사합니다

압축을 풀어 `baseline/` 폴더 안으로 이동한 상태에서 실행합니다.

```bash
python3 team_a_tools/check_handoff.py .
```

검사에 PyTorch 설치나 원본 데이터는 필요하지 않습니다. PASS가 나와도 원본 데이터 접근과 실제 추론 재현은 아래 단계에서 별도로 확인해야 합니다.

### 7-2. 같은 상위 폴더에 전체 저장소를 준비합니다

새 작업 위치에서 `baseline/`과 `ultrafast-linet-baseline/`이 나란히 있도록 합니다. **아래 clone 명령은 두 폴더의 상위 위치에서 실행합니다.** 이미 저장소가 있으면 삭제하지 말고 별도의 새 작업 위치에서 준비하거나 코드 commit을 확인합니다.

```bash
git clone https://github.com/7b7hom/ultrafast-linet-baseline.git
cd ultrafast-linet-baseline
git checkout --detach 3ead18e7b34444aa19eb39c815da86d2dc024d08
git rev-parse HEAD
cp -R ../baseline/team_a_tools .
cp ../baseline/common_metrics.py .
```

저장소 접근이 안 되면 같은 commit의 전체 소스를 받아야 합니다. `best.pt` 하나만으로 원 평가기가 실행되지는 않습니다. 원 평가기는 학습 완료를 확인하기 위해 `reports/a_full_seed42/`의 초기·마지막 가중치와 로그·설정까지 읽습니다.

### 7-3. 실제 환경에 맞게 설치합니다


```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```


### 7-4. 원본 파일과 고정 분할을 확인합니다

5절대로 원본을 준비합니다. 아래 명령은 다운로드가 아니라 이미 준비된 데이터의 검사입니다.

```bash
.venv/bin/python scripts/prepare_data.py --data-root data/LOL --output manifests --seed 42
```

해시나 분할이 다르면 데이터 루트·원본 파일을 확인합니다. 오류를 없애기 위해 고정 manifest를 지우거나 바꾸지 않습니다.

가중치는 전달 파일 `best.pt`와 저장소 `reports/a_full_seed42/best.pt`가 같은지 확인합니다. 이번 정리에서는 같았으며 기대 SHA-256은 다음과 같습니다.

`1e2df6a4e420bf8335790cd14c8102ca58f9d804126c7a31f3275f648d445fe6`

### 7-5. 검증과 시험을 새 출력 폴더에 평가합니다

아래 명령은 계속 **고정 저장소 루트**에서 실행합니다.

```bash
.venv/bin/python team_a_tools/verify_validation.py --data-root data/LOL --output runs/receiver_validation
.venv/bin/python scripts/evaluate.py --data-root data/LOL --run-dir reports/a_full_seed42 --manifests manifests --output runs/receiver_test --threads 1 --save-images
```

검증은 50장, 시험은 15장이 나와야 합니다. 이미 사용한 출력 폴더가 있으면 새 이름을 사용합니다. 실제 실행 시작·종료 시각, 명령, 환경을 기록합니다. 새 결과를 원래 전달 CSV에 바로 덮어쓰지 않습니다.

원 평가기의 `per_image.csv`는 `name, psnr, ssim` 등의 열을 쓰며 팀 전달 CSV와 열 이름이 다릅니다. 비교할 때 `image_id = "LOL-v1/" + name`, `psnr_db = psnr`로 대응시킵니다. 가로·세로와 split·모델·checkpoint도 확인합니다.

재평가 값이 다르면 이미지 ID·가중치 해시·전처리·float/PNG 채점 차이·환경부터 점검합니다. 평균이 비슷하다는 이유만으로 일부 이미지 누락을 무시하지 않습니다.


## 8. 현재 팀 실험 기준(LOL-v2-real)과 이 묶음의 관계

| 항목 | 이 묶음 (LOL-v1 예비) | 현재 팀 기준 (LOL-v2-real) |
|---|---|---|
| 데이터·분할 | LOL-v1, 435 / 50 / 15 | LOL-v2-real, 620 / 69 / 100 (장면 단위 분할) |
| 이미지 ID | `LOL-v1/1.png` | `LOL-v2-real/00011.png` |
| 명부 | `split_manifest.csv` | `split_manifest_lolv2.csv` (저장소 manifest에서 생성) |
| 학습 crop | 중앙 180 | random 180 |
| 모델 | Max 1개 (epoch 20) | Mini 36 / Max 180 / Max+ 937 일반 학습 (`gt_random_v1`) |
| 채점 함수·CSV 형식 | `common_metrics.py`, 8개 열 | 동일 |

- LOL-v2-real 명부: `split_manifest_lolv2.csv`(789행). 저장소 `manifests/lolv2_real/`에서 `team_a_tools/lolv2_tables.py manifest`로 다시 만들 수 있습니다.
- LOL-v2-real 품질 CSV: 일반 학습이 끝나면 `best_validation_per_image.json`을 아래처럼 변환합니다. 검증 69장이 하나라도 빠지면 실패합니다.

```bash
python3 team_a_tools/lolv2_tables.py quality --manifest-csv split_manifest_lolv2.csv --split validation \
    --results <Drive>/ondevice/runs/gt_random_v1/max_gt_s42/best_validation_per_image.json \
    --model-id UltraFastLiNET_Max --checkpoint-id max_gt_random_v1_s42_best --output max_validation_quality_lolv2.csv
```

- 중간발표의 베이스라인 수치는 LOL-v2-real 검증 69장 결과를 씁니다. 이 묶음의 LOL-v1 수치를 쓰려면 “LOL-v1 예비 재현”으로 따로 표시하고 LOL-v2 결과와 한 표에서 비교하지 않습니다.

## 9. 발표 3·4장 초안 (LOL-v1 예비 결과 — 현재 기준 아님)

> LOL-v2-real 결과가 나오면 아래 표와 설명을 LOL-v2-real 검증 69장 기준으로 바꿉니다.

평가 조건
- 모델: UltraFast-LiNET-Max, 파라미터 180개.
- 가중치: 저장소의 직접 학습 결과 `best.pt`; 공식 공개 가중치와 구분.
- 학습 이력: seed 42, 360 epoch 기록에서 검증 PSNR로 epoch 20 선택.
- 데이터: LOL-v1, train 435 / validation 50 / test 15.
- 품질 평가: RGB float32 [0,1], 원본 600×400, 예측만 clamp, 이미지별 PSNR·SSIM 평균.
- 이번 제출 보고서 환경: Python 3.14.2, PyTorch 2.14.0, Apple M3, CPU threads 1.
- CPU 속도 비교 조건은 B의 실제 측정 자료를 받아 별도로 반영.

Max 베이스라인 정량 결과

| 집합 | N | 평균 PSNR(dB) | 평균 SSIM |
|---|---:|---:|---:|
| validation | 50 | 16.0612 | 0.515624 |
| test | 15 | 17.9173 | 0.527209 |

C가 입력·복원·정답 비교 이미지와 같은 위치 확대 자료를 추가합니다. 전체 집합의 평균과 대표 사례를 구분해 제시합니다.

발표 설명: “LOL-v1 예비 베이스라인으로 Max를 평가한 기록을 정리했습니다. 시험 15장의 평균 PSNR은 17.9173 dB, SSIM은 0.527209입니다. 검증 자료와 시험 자료를 구분했고, 같은 기준으로 후속 Mini 결과를 비교할 예정입니다.”

실제 재평가를 누가 수행했는지는 실행 이력에 맞게 설명합니다. 단일 seed·작은 시험 집합의 예비 결과이며, 입력 대비 개선량은 입력 자체의 점수를 별도로 구해야 말할 수 있습니다.

## 10. 복원 PNG 15개와 원본 경로

아래 복원 경로는 이 전달 폴더 기준, low/GT 경로는 데이터 루트 기준입니다. 각 파일은 해당 시험 ID의 Max 복원 결과라는 동일한 역할을 가집니다.

| 복원 PNG | image_id | 원본 low | 정답 GT |
|---|---|---|---|
| `enhanced/test/01_1.png` | `LOL-v1/1.png` | `eval15/low/1.png` | `eval15/high/1.png` |
| `enhanced/test/02_111.png` | `LOL-v1/111.png` | `eval15/low/111.png` | `eval15/high/111.png` |
| `enhanced/test/03_146.png` | `LOL-v1/146.png` | `eval15/low/146.png` | `eval15/high/146.png` |
| `enhanced/test/04_179.png` | `LOL-v1/179.png` | `eval15/low/179.png` | `eval15/high/179.png` |
| `enhanced/test/05_22.png` | `LOL-v1/22.png` | `eval15/low/22.png` | `eval15/high/22.png` |
| `enhanced/test/06_23.png` | `LOL-v1/23.png` | `eval15/low/23.png` | `eval15/high/23.png` |
| `enhanced/test/07_493.png` | `LOL-v1/493.png` | `eval15/low/493.png` | `eval15/high/493.png` |
| `enhanced/test/08_547.png` | `LOL-v1/547.png` | `eval15/low/547.png` | `eval15/high/547.png` |
| `enhanced/test/09_55.png` | `LOL-v1/55.png` | `eval15/low/55.png` | `eval15/high/55.png` |
| `enhanced/test/10_665.png` | `LOL-v1/665.png` | `eval15/low/665.png` | `eval15/high/665.png` |
| `enhanced/test/11_669.png` | `LOL-v1/669.png` | `eval15/low/669.png` | `eval15/high/669.png` |
| `enhanced/test/12_748.png` | `LOL-v1/748.png` | `eval15/low/748.png` | `eval15/high/748.png` |
| `enhanced/test/13_778.png` | `LOL-v1/778.png` | `eval15/low/778.png` | `eval15/high/778.png` |
| `enhanced/test/14_780.png` | `LOL-v1/780.png` | `eval15/low/780.png` | `eval15/high/780.png` |
| `enhanced/test/15_79.png` | `LOL-v1/79.png` | `eval15/low/79.png` | `eval15/high/79.png` |

## 11. 전달받은 직후 체크리스트

- [ ] `experiment_settings.md`에서 모델·데이터·평가 기준을 확인했다.
- [ ] 파일 검사기를 실행해 PASS를 확인했다.
- [ ] LOL-v2-real 원본을 준비하고 `split_manifest_lolv2.csv`의 경로와 실제 위치를 연결했다.
- [ ] C는 LOL-v2-real 검증 69행으로 ID 연결이 되는지 확인했다. (LOL-v1 재평가 시: 검증 50행·시험 15행)
- [ ] D는 LOL-v2-real 분할·random 180 crop·Max+ 교사 기준(8절)을 확인했다.
- [ ] B는 같은 장치(M5)와 측정 조건(model / end_to_end, 저장·표시 제외)을 정했다.
- [ ] 팀장은 이 묶음의 수치를 LOL-v1 예비 결과로 표시하고, 중간발표 기준은 LOL-v2-real로 맞췄다.
- [ ] 재평가를 했다면 실제 시각·명령·환경·새 결과 경로를 남겼다.

## 12. 정리본의 기준과 보존 범위

이 정리본의 출처는 사용자가 제출한 `baseline.zip`이다. 원 ZIP SHA-256은 아래와 같다.

`d7d8fe8d7dd36238d47d6a7532fcb45d90262d1781fb783234b144010bebd0fa`

점수 CSV, 데이터 명부, 복원 이미지, 가중치, 공통 지표, 설치 목록, LICENSE, 원 평가 JSON, 검증 재평가 도구의 내용은 그대로 보존했다. 안내문과 학습 설정 문서는 통합·정리했고, 검사기는 새 폴더 구조와 CSV/JSON 대조에 맞췄다.

이 파일과 `experiment_settings.md`에 적힌 결과는 제출 기록을 설명한다. 원본 입력·정답에 대한 모델 추론을 이번 정리 과정에서 새로 수행했다는 뜻은 아니다. 근거 JSON을 보관하되 생성 시각이나 출처 라벨만으로 실제 실행 이력을 단정하지 않는다.

고정 코드와 상세 구현: [ultrafast-linet-baseline 고정 버전](https://github.com/7b7hom/ultrafast-linet-baseline/tree/3ead18e7b34444aa19eb39c815da86d2dc024d08).
