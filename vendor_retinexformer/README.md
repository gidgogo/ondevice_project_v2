# Retinexformer 구조 (고성능 교사용, 수정 없음)

- 출처: [caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) 커밋 `1e9a0efce4b306b6701b824768370ff26066c32a`
- 파일: `basicsr/models/archs/RetinexFormer_arch.py`를 그대로 복사. 라이선스 `LICENSE.txt`(MIT, Copyright (c) 2023 Yuanhao Cai)
- 해시: `provenance.json`의 `retinexformer` 항목에 고정하고 `tests/test_teacher.py`가 검사한다.
- 구조 설정: 원 저장소 `Options/RetinexFormer_LOL_v2_real.yml`의 `stage=1, n_feat=40, num_blocks=[1,2,2]` (학습 파라미터 1,605,701개)

학습 코드는 원 저장소의 BasicSR 대신 `scripts/train_teacher.py`를 사용한다. 데이터는 우리 고정 manifest(학습 620쌍·검증 69쌍)만 읽는다. 원 설정과 다른 점은 `docs/teacher_candidates.md`에 기록한다.
공개 사전학습 가중치는 사용하지 않는다(우리 검증 분할이 포함된 공식 Train 전체로 학습되었을 수 있음).
