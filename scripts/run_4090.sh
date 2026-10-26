#!/usr/bin/env bash
# 학교 4090 리눅스 서버에서 남은 학습 4개(교사 이어서 학습 + Retinexformer 증류 3개) 실행
#
#   bash scripts/run_4090.sh smoke    # 점검 5~15분: 짧게 돌려 보고 예상 시간 + 테스트 메일
#   bash scripts/run_4090.sh full     # 본 실행: 백그라운드로 시작 (SSH 끊겨도 계속)
#   bash scripts/run_4090.sh status   # 진행 상황 보기
#   bash scripts/run_4090.sh stop     # 중단 (다시 full 하면 이어서)
#
# 추가 옵션은 뒤에 붙인다. 예: bash scripts/run_4090.sh full --skip-teacher
# 환경변수: ONDEVICE_BASE(기본 ~/ondevice), PYTHON(이미 있는 CUDA PyTorch 환경), CUDA_VISIBLE_DEVICES(GPU 번호)
# 알림 설정은 ~/.ondevice_notify.env 에 둔다 (docs/server_4090.md).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${ONDEVICE_BASE:-$HOME/ondevice}"
MODE="${1:-smoke}"
shift || true
mkdir -p "$BASE/logs"
OUT="$BASE/logs/run_remaining.out"
PIDFILE="$BASE/logs/run_remaining.pid"

if [ -f "$HOME/.ondevice_notify.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "$HOME/.ondevice_notify.env"; set +a
fi

# 파이썬 환경: PYTHON을 지정하지 않으면 저장소 안에 .venv를 만들어 설치
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  PY="$REPO/.venv/bin/python"
  if [ ! -x "$PY" ]; then
    echo "[설치] $REPO/.venv 생성 후 패키지 설치 (처음 한 번, 몇 분 걸림)"
    python3 -m venv "$REPO/.venv"
    "$REPO/.venv/bin/pip" install -q --upgrade pip
    "$REPO/.venv/bin/pip" install -q -r "$REPO/requirements-server.txt"
  fi
fi

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "$MODE" in
  smoke)
    "$PY" -u "$REPO/scripts/run_remaining.py" --base "$BASE" --smoke "$@"
    ;;
  full)
    if running; then echo "이미 실행 중 (PID $(cat "$PIDFILE")). 진행 상황: bash scripts/run_4090.sh status"; exit 1; fi
    nohup "$PY" -u "$REPO/scripts/run_remaining.py" --base "$BASE" "$@" >> "$OUT" 2>&1 &
    echo $! > "$PIDFILE"
    echo "시작함 (PID $!). 터미널을 닫아도 계속 돕니다."
    echo "진행 상황: bash scripts/run_4090.sh status   또는   tail -f $OUT"
    ;;
  status)
    if running; then echo "실행 중 (PID $(cat "$PIDFILE"))"; else echo "실행 중 아님"; fi
    for f in "$BASE"/teachers/teacher_v1/retinexformer_s42/summary.json "$BASE"/runs/kd_random_v1/*strongkd*/summary.json; do
      [ -f "$f" ] && echo "$f" && cat "$f" && echo
    done
    echo "---- 마지막 로그"; tail -n 15 "$OUT" 2>/dev/null || true
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")" && echo "중단 신호 보냄. 다시 시작: bash scripts/run_4090.sh full"; else echo "실행 중 아님"; fi
    ;;
  *)
    echo "사용법: bash scripts/run_4090.sh [smoke|full|status|stop] [추가 옵션]"; exit 1
    ;;
esac
