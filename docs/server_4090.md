# 학교 4090 서버에서 남은 학습 4개 한 번에 돌리기

남은 학습: **① Retinexformer 교사 마저 학습(18000 → 30000회)**, **② Max+ ← Retinexformer**, **③ Max ← Retinexformer**, **④ Mini ← Retinexformer** 증류.
`scripts/run_4090.sh`가 이 순서대로 돌리고, 단계가 끝날 때마다 알림을 보낸다. 마지막에는 비교표와 결과 zip을 메일에 첨부한다.
코드·설정은 Colab 실행과 같다(β=0.3, 120 epoch, batch 40, 각 모델 일반 학습의 `initial.pt`에서 시작). 증류에 쓰는 교사는 교사 학습이 끝난 뒤 best.pt를 복사한 고정본이다.

## 1. 서버에 올릴 것

```bash
# 서버에서
git clone https://github.com/gidgogo/ondevice_project_v2.git ~/ondevice_project_v2
```

내 컴퓨터에서 두 파일을 서버로 복사한다.

| 파일 | 서버 위치 | 내용 |
|---|---|---|
| `ondevice_server_state.zip` | `~/`에서 풀기 → `~/ondevice/` | 일반 학습 3개(`initial.pt` 등), Max+ 교사 증류 2개(비교표용), 교사 18000회 `last.pt` |
| `LOLv2.zip` (Drive `ondevice/`에 있던 것) | `~/ondevice/LOLv2.zip` | 데이터. 처음 실행 때 자동으로 풀고 학습 620·검증 69쌍을 파일 해시로 확인 |

```bash
# 내 맥에서 (서버 주소·계정은 바꿔서)
scp ondevice_server_state.zip LOLv2.zip 계정@서버주소:~/
# 서버에서
cd ~ && unzip ondevice_server_state.zip && mv LOLv2.zip ~/ondevice/
```

Colab에서 교사를 18000회 이후로 더 돌렸다면 `~/ondevice/teachers/teacher_v1/retinexformer_s42/`를 Drive의 최신 폴더로 바꿔 넣는다. Colab 교사 학습이 아직 돌고 있으면 먼저 멈춘다(같은 학습을 두 곳에서 하지 않기).

## 2. 알림 설정 (선택, 안 하면 로그 파일에만 남음)

**메일 (Gmail)**: Google 계정에서 2단계 인증을 켠 뒤 "앱 비밀번호"를 하나 만든다. 서버에 아래 파일을 만든다.

```bash
cat > ~/.ondevice_notify.env <<'EOF'
SMTP_USER=본인계정@gmail.com
SMTP_PASSWORD=앱비밀번호16자리
NOTIFY_TO=받을메일@gmail.com
EOF
chmod 600 ~/.ondevice_notify.env
```

공용 서버이므로 일반 비밀번호가 아니라 **앱 비밀번호**를 쓰고, 학습이 끝나면 파일을 지우고 Google 계정에서 앱 비밀번호도 삭제한다.
학교 서버가 메일 포트(587)를 막으면 메일이 안 간다. 점검 단계에서 테스트 메일이 오는지 확인한다.

**휴대폰 푸시 (ntfy, 비밀번호 필요 없음)**: 휴대폰에 ntfy 앱을 깔고 남이 추측하기 어려운 주제 이름을 구독한 뒤 같은 파일에 `NTFY_TOPIC=주제이름`을 추가한다. 주제 이름을 아는 사람은 알림을 볼 수 있으므로 결과 수치 정도만 보내는 용도로 쓴다.

## 3. 점검 → 본 실행

```bash
cd ~/ondevice_project_v2
nvidia-smi                       # 다른 사람이 GPU를 쓰는지 확인. GPU가 여러 개면 export CUDA_VISIBLE_DEVICES=1 처럼 지정
bash scripts/run_4090.sh smoke   # 처음엔 패키지 설치(몇 분) + 짧은 점검. 예상 시간과 테스트 알림
bash scripts/run_4090.sh full    # 본 실행. 백그라운드로 돌아서 SSH를 끊어도 계속
```

| 명령 | 하는 일 |
|---|---|
| `bash scripts/run_4090.sh status` | 실행 중인지, 단계별 summary, 마지막 로그, GPU 사용량 |
| `tail -f ~/ondevice/logs/run_remaining.out` | 실시간 로그 |
| `bash scripts/run_4090.sh stop` | 중단(알림 감). 다시 `full`이면 끝난 단계는 건너뛰고 마지막 저장 지점부터 |
| `bash scripts/run_4090.sh full --skip-teacher` | 교사를 더 학습하지 않고 지금 best.pt(9000회, 21.99 dB)로 바로 증류 |

이미 CUDA PyTorch가 있는 conda 환경을 쓰려면 `PYTHON=$(which python) bash scripts/run_4090.sh smoke`처럼 실행한다(새 가상환경을 만들지 않음).

## 4. 끝나면

- 메일 제목 `[ondevice] 남은 학습 4개 모두 완료`, 본문에 비교표, 첨부 `server_results_*.zip`(기록 파일 + 학생 best.pt + A 형식 품질 CSV + 로그).
- 서버 `~/ondevice/server_results/`에도 같은 파일이 있다. 교사 가중치는 `~/ondevice/teachers/`에 남는다.
- 실패하면 `[ondevice] 실패 (단계)` 메일에 마지막 로그가 온다. 원인을 고친 뒤 `full`을 다시 실행하면 이어서 한다.

## 주의

- 일반 학습과 Max+ 교사 증류는 Colab T4, 이번 증류는 4090에서 돈다. 코드·설정·초기값은 같지만 GPU와 PyTorch 버전이 달라 결과에 아주 작은 차이가 생길 수 있다. 각 `run.json`에 GPU와 환경이 기록된다.
- 교사 학습이 끝나기 전에는 증류를 시작하지 않는다(`--skip-teacher`를 쓰면 바로 시작).
- 시험 100장은 쓰지 않는다. 모든 결과는 검증 69장 기준이다.
