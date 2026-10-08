# 고성능 교사 후보 (조사 단계, 미확정)

계획: 우리 학습 분할(620쌍)로 직접 학습한 큰 모델을 고성능 교사로 쓴다. 공개 사전학습 가중치는 공식 Train 689쌍 전체(우리 검증 69쌍 포함)나 LOL-v1으로 학습되었을 수 있어, 그대로 쓰면 검증 기반 선택이 오염된다.

| 후보 | 코드·라이선스 | LOL-v2-real 보고 성능 | 비고 |
|---|---|---|---|
| Retinexformer (ICCV 2023) | [GitHub](https://github.com/caiyuanhao1998/Retinexformer), MIT | README: PSNR 27.71 / SSIM 0.856 (GT mean 사용 설정) | 학습 설정 `Options/RetinexFormer_LOL_v2_real.yml`. 사전학습 가중치 제공 |
| HVI-CIDNet (CVPR 2025) | [GitHub](https://github.com/Fediory/HVI-CIDNet), MIT | README: PSNR 24.11 / SSIM 0.8675 (normal), 28.14 / 0.892 (GT mean) | 사전학습 가중치 제공 |

- README 수치는 측정 조건(GT mean 등)이 우리 규약과 다르므로 참고만 한다.
- 파라미터 수·학습 시간은 아직 직접 확인하지 않았다. 두 모델 모두 수십만~수백만 파라미터 규모로 알려져 있어 학습에는 GPU가 필요할 것으로 본다.

## 교사 선정 기준(제안)

1. 우리 검증 분할에서 기본 교사(Max+-GT)보다 PSNR·SSIM이 확실히 높을 것
2. 우리 학습 분할로 제한된 GPU 예산 안에서 학습 가능할 것
3. 교사 출력은 학습 전 한 번 계산해 `teacher_outputs/` 폴더에 저장하고, `train.py`의 `teacher={"outputs_dir": ...}`로 사용

교사 출력 폴더에는 학습 분할 저조도 이미지와 같은 파일명의 원본 해상도 PNG를 둔다. 학생은 같은 위치를 잘라 증류 손실을 계산한다.
