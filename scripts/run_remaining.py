"""학교 4090 같은 리눅스 GPU 서버에서 남은 학습 4개를 한 번에 순서대로 실행한다.

  1) Retinexformer 교사: Colab에서 저장한 last.pt부터 30000회까지 이어서 학습
  2) 교사 best.pt를 고정 사본으로 복사 (증류 도중 교사가 바뀌지 않게)
  3) 증류: Max+ ← Retinexformer, Max ← Retinexformer, Mini ← Retinexformer
     (각 모델 일반 학습의 initial.pt에서 시작, β=0.3, 120 epoch, 일반 학습과 같은 설정)
  4) 일반 학습 / Max+ 교사 증류 / Retinexformer 교사 증류 비교표를 만들고 메일·알림 전송

다시 실행하면 끝난 단계는 건너뛰고, 끊긴 단계는 마지막 저장 지점부터 이어서 한다.

    python scripts/run_remaining.py --base ~/ondevice --smoke   # 점검: 짧게 돌려 보고 예상 시간 + 테스트 메일
    python scripts/run_remaining.py --base ~/ondevice           # 본 실행

알림(선택, 환경변수): SMTP_USER, SMTP_PASSWORD(Gmail 앱 비밀번호), NOTIFY_TO, SMTP_HOST, SMTP_PORT / NTFY_TOPIC
알림이 실패해도 학습은 멈추지 않는다. 모든 알림은 BASE/notifications.log에도 남는다.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import smtplib
import ssl
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SEED = 42
NAMES = {"mini": "Mini", "max": "Max", "maxplus": "Max+"}


# ---------------------------------------------------------------- 기본 도구

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(text):
    print(f"[{now()}] {text}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hm(hours):
    minutes = round(hours * 60)
    return f"{minutes // 60}시간 {minutes % 60}분" if minutes >= 60 else f"{minutes}분"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_sets(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def as_sets(values):
    return [f"{k}={json.dumps(v)}" for k, v in values.items()]


def run(cmd, log_path, cwd=SCRIPTS):
    """명령을 실행하면서 화면과 로그 파일에 같이 남긴다. 실패하면 마지막 줄들과 함께 예외."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    say("실행: " + " ".join(str(c) for c in cmd))
    tail = []
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} {' '.join(str(c) for c in cmd)}\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
                tail = (tail + [line.rstrip()])[-40:]
            code = proc.wait()
        except BaseException:  # 중단되면 학습 프로세스도 같이 끝내서 GPU를 붙잡고 있지 않게 한다
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    if code != 0:
        raise RuntimeError(f"명령 실패 (exit {code}): {cmd[2] if len(cmd) > 2 else cmd}\n" + "\n".join(tail))


# ---------------------------------------------------------------- 알림

class Notifier:
    def __init__(self, base):
        self.log = base / "notifications.log"
        self.user = os.environ.get("SMTP_USER")
        self.password = os.environ.get("SMTP_PASSWORD")
        self.to = os.environ.get("NOTIFY_TO") or self.user
        self.host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.topic = os.environ.get("NTFY_TOPIC")

    def channels(self):
        out = []
        if self.user and self.password and self.to:
            out.append(f"메일({self.to})")
        if self.topic:
            out.append("ntfy")
        return out or ["없음(로그 파일만)"]

    def send(self, subject, body, attachment=None):
        subject = "[ondevice] " + subject
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {now()} {subject}\n{body}\n")
        if self.user and self.password and self.to:
            try:
                msg = EmailMessage()
                msg["Subject"], msg["From"], msg["To"] = subject, self.user, self.to
                small = attachment is not None and attachment.exists() and attachment.stat().st_size <= 20 * 1024 * 1024
                text = body
                if attachment is not None and attachment.exists() and not small:
                    text += f"\n\n(첨부 파일이 20MB를 넘어 생략: {attachment})"
                msg.set_content(text)  # 본문을 먼저 넣고 첨부를 붙인다
                if small:
                    msg.add_attachment(attachment.read_bytes(), maintype="application", subtype="zip",
                                       filename=attachment.name)
                with smtplib.SMTP(self.host, self.port, timeout=60) as server:
                    server.starttls(context=ssl.create_default_context())
                    server.login(self.user, self.password)
                    server.send_message(msg)
                say(f"메일 전송: {subject}")
            except Exception as error:  # 알림 실패로 학습을 멈추지 않는다
                say(f"메일 전송 실패: {type(error).__name__}: {error}")
        if self.topic:
            try:
                data = json.dumps({"topic": self.topic, "title": subject, "message": body[:3500]}).encode()
                req = urllib.request.Request("https://ntfy.sh/", data=data, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=30).read()
                say(f"ntfy 전송: {subject}")
            except Exception as error:
                say(f"ntfy 전송 실패: {type(error).__name__}: {error}")


# ---------------------------------------------------------------- 준비

def prepare_data(args):
    """--data-root가 없으면 BASE/LOLv2.zip을 BASE/data에 풀고 Real_captured를 찾는다.
    학습 620쌍·검증 69쌍은 manifest의 파일 해시로 하나씩 확인한다. 시험 이미지는 읽지 않는다."""
    if args.data_root:
        root = args.data_root
    else:
        folder = args.base / "data"
        marker = folder / ".complete"
        if not marker.exists():
            archive = args.base / "LOLv2.zip"
            if not archive.is_file():
                raise FileNotFoundError(f"데이터가 없습니다: {archive} 를 올리거나 --data-root를 지정하세요.")
            say(f"압축 해제: {archive}")
            folder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not (folder / name).resolve().is_relative_to(folder.resolve()):
                        raise ValueError(f"잘못된 ZIP 경로: {name}")
                z.extractall(folder)
            marker.write_text(now())
        roots = [p for p in folder.rglob("Real_captured") if (p / "Train/Low").is_dir()]
        if len(roots) != 1:
            raise FileNotFoundError(f"Real_captured 폴더를 하나 찾아야 합니다: {roots}")
        root = roots[0]
    checked = args.base / ".data_checked"
    stamp = f"{root.resolve()}|{sha(args.manifests / 'train.json')}|{sha(args.manifests / 'validation.json')}"
    if not (checked.exists() and checked.read_text() == stamp):
        for split, count in (("train", 620), ("validation", 69)):
            pairs = read_json(args.manifests / f"{split}.json")["pairs"]
            if not args.test_mode and len(pairs) != count:
                raise ValueError(f"{split}: {len(pairs)}쌍 (예상 {count})")
            for row in pairs:
                for key in ("low", "high"):
                    path = root / row[key]
                    if not path.is_file():
                        raise FileNotFoundError(f"데이터 파일 없음: {path}")
                    if f"{key}_sha256" in row and sha(path) != row[f"{key}_sha256"]:
                        raise ValueError(f"데이터 해시 불일치: {path}")
            say(f"데이터 확인: {split} {len(pairs)}쌍")
        checked.write_text(stamp)
    return root


def gpu_check(args):
    if args.kd_extra.get("device", "cuda") == "cpu":
        return "cpu (테스트)"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. nvidia-smi와 PyTorch 설치(CUDA 빌드)를 확인하세요.")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    say(f"GPU: {name} · 사용 가능 메모리 {free / 2**30:.1f} / {total / 2**30:.1f} GB · PyTorch {torch.__version__}")
    if free < 6 * 2**30:
        say("경고: GPU 여유 메모리가 6GB보다 적습니다. 다른 사람이 쓰고 있는지 nvidia-smi로 확인하세요.")
    return name


def teacher_settings(args):
    cfg = read_json(REPO / "configs/teacher_retinexformer.json")
    cfg.update(teacher="retinexformer", seed=SEED, total_iters=args.teacher_iters)
    cfg.update(args.teacher_extra)
    return cfg


def student_base_config(args, student):
    cfg = read_json(REPO / "configs/colab_gt.json")
    cfg.update(model=student, seed=SEED, epochs=args.epochs, batch_size=args.batch_size)
    cfg.update({k: v for k, v in args.kd_extra.items() if k not in ("teacher", "beta")})
    return cfg


def paths(args):
    return {
        "teacher": args.base / "teachers" / "teacher_v1" / "retinexformer_s42",
        "frozen_dir": args.base / "teachers" / "frozen",
        "gt": {s: args.base / "runs" / "gt_random_v1" / f"{s}_gt_s{SEED}" for s in NAMES},
        "basekd": {s: args.base / "runs" / "kd_random_v1" / f"{s}_basekd_b0.3_s{SEED}" for s in NAMES},
        "strong": {s: args.base / "runs" / args.experiment / f"{s}_strongkd_b{args.beta}_s{SEED}" for s in NAMES},
        "logs": args.base / "logs",
    }


def check_inputs(args, p):
    """긴 학습을 시작하기 전에 필요한 파일·설정을 모두 확인한다."""
    problems = []
    t = p["teacher"]
    if not (t / "run.json").exists():
        problems.append(f"교사 학습 기록 없음: {t}/run.json (Colab의 teachers/teacher_v1/retinexformer_s42 폴더를 그대로 올리세요)")
    else:
        if read_json(t / "run.json")["config"] != teacher_settings(args):
            problems.append("교사 run.json 설정이 지금 설정(30000회 등)과 다릅니다.")
        summary = read_json(t / "summary.json") if (t / "summary.json").exists() else {}
        if not summary.get("training_complete") and not any((t / n).exists() for n in ("last.pt", "last.prev.pt")):
            problems.append(f"교사가 끝나지 않았는데 이어서 학습할 last.pt가 없습니다: {t}")
        if args.skip_teacher and not (t / "best.pt").exists():
            problems.append("--skip-teacher인데 교사 best.pt가 없습니다.")
    for s in args.students:
        g = p["gt"][s]
        for name in ("initial.pt", "run.json"):
            if not (g / name).exists():
                problems.append(f"{NAMES[s]} 일반 학습 파일 없음: {g / name}")
        if (g / "run.json").exists() and read_json(g / "run.json")["config"] != student_base_config(args, s):
            problems.append(f"{NAMES[s]} 일반 학습 설정이 지금 설정과 다릅니다: {g / 'run.json'}")
    if problems:
        raise RuntimeError("시작 전 확인 실패:\n- " + "\n- ".join(problems))


# ---------------------------------------------------------------- 단계

def train_teacher(args, p, data_root):
    t = p["teacher"]
    summary = read_json(t / "summary.json") if (t / "summary.json").exists() else {}
    if summary.get("training_complete"):
        say(f"교사 이미 완료: {summary['completed_iters']}회, 최고 {summary['best']}")
        return summary
    if args.skip_teacher:
        say(f"--skip-teacher: 교사를 {summary.get('completed_iters')}회 시점 그대로 사용")
        return summary
    cmd = [sys.executable, "-u", SCRIPTS / "train_teacher.py", "--data-root", data_root, "--manifests", args.manifests,
           "--config", REPO / "configs/teacher_retinexformer.json", "--output", t, "--resume",
           "--set", *as_sets(teacher_settings(args))]
    run(cmd, p["logs"] / "1_teacher.log")
    summary = read_json(t / "summary.json")
    if not summary.get("training_complete"):
        raise RuntimeError(f"교사 학습이 끝나지 않았습니다: {summary}")
    return summary


def freeze_teacher(p, summary):
    """증류에 쓸 교사를 고정 사본으로 만든다. 이름에 best 반복 수를 넣는다."""
    source = p["teacher"] / "best.pt"
    iteration = summary["best"]["iteration"]
    target = p["frozen_dir"] / f"retinexformer_s42_best_it{iteration}.pt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    if sha(target) != sha(source):
        raise RuntimeError(f"고정 교사 사본이 현재 best.pt와 다릅니다: {target}")
    say(f"증류 교사 고정: {target.name} (검증 {summary['best']['psnr']:.4f} dB)")
    return target


def distill(args, p, student, frozen, data_root):
    out = p["strong"][student]
    spec = {"model": "retinexformer", "checkpoint": str(frozen)}
    settings = dict(model=student, seed=SEED, batch_size=args.batch_size, beta=args.beta, teacher=spec, epochs=args.epochs)
    settings.update({k: v for k, v in args.kd_extra.items() if k not in ("teacher", "beta")})
    effective = dict(student_base_config(args, student), beta=args.beta, teacher=spec)
    if (out / "run.json").exists():
        previous = read_json(out / "run.json")
        if previous["config"] != effective or previous.get("teacher_checkpoint_sha256") != sha(frozen):
            raise RuntimeError(f"{out} 의 기존 결과가 지금 교사·설정과 다릅니다. --experiment 를 새 이름으로 지정하세요.")
    summary = read_json(out / "summary.json") if (out / "summary.json").exists() else {}
    if summary.get("training_complete"):
        say(f"{NAMES[student]} 증류 이미 완료: {summary['best']}")
        return summary
    resumable = any((out / n).exists() for n in ("last.pt", "last.prev.pt"))
    if out.exists() and any(out.iterdir()) and not resumable:
        raise RuntimeError(f"이어갈 checkpoint가 없는데 폴더가 비어 있지 않습니다: {out}")
    cmd = [sys.executable, "-u", SCRIPTS / "train.py", "--data-root", data_root, "--manifests", args.manifests,
           "--config", REPO / "configs/colab_gt.json", "--output", out]
    cmd += ["--resume"] if resumable else ["--init-checkpoint", p["gt"][student] / "initial.pt"]
    run(cmd + ["--set", *as_sets(settings)], p["logs"] / f"2_strongkd_{student}.log")
    summary = read_json(out / "summary.json")
    if not summary.get("training_complete"):
        raise RuntimeError(f"{NAMES[student]} 증류가 끝나지 않았습니다: {summary}")
    return summary


# ---------------------------------------------------------------- 결과 정리

def best_of(folder):
    path = folder / "summary.json"
    return read_json(path).get("best") if path.exists() else None


def wins(a_dir, b_dir):
    """b가 a보다 PSNR이 높은 검증 사진 수"""
    try:
        a = {r["name"]: r["psnr"] for r in read_json(a_dir / "best_validation_per_image.json")}
        b = {r["name"]: r["psnr"] for r in read_json(b_dir / "best_validation_per_image.json")}
    except FileNotFoundError:
        return ""
    return f"{sum(b[n] > a[n] for n in a if n in b)}/{len(a)}"


def report(args, p, frozen, gpu, started):
    rows = []
    for s in args.students:
        gt, base, strong = best_of(p["gt"][s]), best_of(p["basekd"][s]), best_of(p["strong"][s])
        rows.append({"student": NAMES[s],
                     "gt_psnr": f"{gt['psnr']:.4f}" if gt else "", "gt_ssim": f"{gt['ssim']:.4f}" if gt else "",
                     "basekd_psnr": f"{base['psnr']:.4f}" if base else "",
                     "strongkd_psnr": f"{strong['psnr']:.4f}" if strong else "",
                     "strongkd_ssim": f"{strong['ssim']:.4f}" if strong else "",
                     "strongkd_minus_gt_db": f"{strong['psnr'] - gt['psnr']:+.4f}" if gt and strong else "",
                     "strongkd_best_epoch": strong["epoch"] if strong else "",
                     "strongkd_better_images": wins(p["gt"][s], p["strong"][s])})
    teacher = read_json(p["teacher"] / "summary.json")
    out = args.base / "server_results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "strongkd_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A 형식 품질 CSV (69행)
    tool, manifest_csv = REPO / "baseline_a/team_a_tools/lolv2_tables.py", REPO / "baseline_a/split_manifest_lolv2.csv"
    made = []
    if tool.exists() and manifest_csv.exists() and not args.test_mode:
        jobs = [(p["teacher"], "Retinexformer", f"retinexformer_teacher_v1_s42_best_it{teacher['best']['iteration']}",
                 "retinexformer_teacher_final")]
        jobs += [(p["strong"][s], {"mini": "UltraFastLiNET_Mini", "max": "UltraFastLiNET_Max", "maxplus": "UltraFastLiNET_MaxPlus"}[s],
                  f"{s}_strongkd_b{args.beta}_{args.experiment}_s42_best", f"{s}_strongkd") for s in args.students]
        for folder, model_id, ckpt_id, stem in jobs:
            target = out / f"{stem}_validation_quality_lolv2.csv"
            subprocess.run([sys.executable, tool, "quality", "--manifest-csv", manifest_csv, "--split", "validation",
                            "--results", folder / "best_validation_per_image.json", "--model-id", model_id,
                            "--checkpoint-id", ckpt_id, "--output", target], check=True)
            made.append(target)

    hours = (time.time() - started) / 3600
    lines = [f"완료: {now()} · 이번 실행 {hm(hours)} · GPU {gpu}",
             f"교사 Retinexformer: {teacher['completed_iters']}/{teacher['target_iters']}회, "
             f"최고 검증 {teacher['best']['psnr']:.4f} dB / {teacher['best']['ssim']:.4f} ({teacher['best']['iteration']}회) "
             f"→ 증류에 사용: {frozen.name}",
             "", "검증 69장 최고 PSNR (dB)", "학생 | 일반 | Max+ 교사 증류 | Retinexformer 교사 증류 | 차이(vs 일반) | 증류가 나은 사진"]
    for r in rows:
        lines.append(f"{r['student']} | {r['gt_psnr']} | {r['basekd_psnr'] or '-'} | {r['strongkd_psnr']} | "
                     f"{r['strongkd_minus_gt_db']} | {r['strongkd_better_images']}")
    lines += ["", f"β={args.beta}, {args.epochs} epoch, batch {args.batch_size}, seed {SEED}. 일반 학습·Max+ 교사 증류는 Colab T4,"
              f" 이번 증류는 {gpu}에서 실행했다(같은 코드·설정·initial.pt).", f"결과 폴더: {out}"]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")

    # 메일 첨부용 묶음: 기록 파일 + 학생 best.pt (교사 가중치·last.pt 제외)
    bundle = out / f"server_results_{datetime.now():%Y%m%d_%H%M}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [out / "summary.txt", out / "strongkd_comparison.csv", *made]:
            z.write(f, f"server_results/{f.name}")
        keep = ("run.json", "summary.json", "log.csv", "progress.json", "session.json",
                "best_validation_per_image.json", "best.pt")
        folders = [(p["teacher"], "teachers/teacher_v1/retinexformer_s42")]
        folders += [(p["strong"][s], f"runs/{args.experiment}/{p['strong'][s].name}") for s in args.students]
        for folder, arc in folders:
            for name in keep:
                if (folder / name).exists() and not (arc.startswith("teachers") and name == "best.pt"):
                    z.write(folder / name, f"{arc}/{name}")
        for log in sorted(p["logs"].glob("*.log")):
            z.write(log, f"logs/{log.name}")
    return text, bundle


# ---------------------------------------------------------------- 점검 모드

def smoke(args, p, data_root, gpu, notifier):
    """짧게 돌려 실행 경로를 확인하고 예상 시간을 계산한다. 본 결과 폴더는 건드리지 않는다."""
    folder = args.base / "smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [f"GPU {gpu}", f"알림 채널: {', '.join(notifier.channels())}"]
    t = p["teacher"]
    summary = read_json(t / "summary.json") if (t / "summary.json").exists() else {"completed_iters": 0}
    left = 0 if summary.get("training_complete") or args.skip_teacher else args.teacher_iters - summary["completed_iters"]
    teacher_hours = 0.0
    if left > 0:
        pilot = folder / "teacher_pilot"
        cmd = [sys.executable, "-u", SCRIPTS / "train_teacher.py", "--data-root", data_root, "--manifests", args.manifests,
               "--config", REPO / "configs/teacher_retinexformer.json", "--output", pilot,
               "--pilot-iters", str(args.smoke_teacher_iters), "--set", *as_sets(teacher_settings(args))]
        run(cmd, folder / "smoke.log")
        timing = read_json(pilot / "pilot_timing.json")
        evals = left // teacher_settings(args)["val_every"] + 1
        teacher_hours = (timing["seconds_per_iter"] * left + timing["validation_seconds"] * evals) / 3600
        lines.append(f"교사: 반복당 {timing['seconds_per_iter']:.3f}초 · 남은 {left}회 예상 {hm(teacher_hours)}")
    else:
        lines.append("교사: 이어서 할 학습 없음")
    teacher_ckpt = t / "best.pt"
    kd_hours = 0.0
    for s in args.students:
        pilot = folder / f"kd_{s}"
        spec = {"model": "retinexformer", "checkpoint": str(teacher_ckpt)}
        settings = dict(model=s, seed=SEED, batch_size=args.batch_size, beta=args.beta, teacher=spec, epochs=1)
        settings.update({k: v for k, v in args.kd_extra.items() if k not in ("teacher", "beta", "epochs")})
        cmd = [sys.executable, "-u", SCRIPTS / "train.py", "--data-root", data_root, "--manifests", args.manifests,
               "--config", REPO / "configs/colab_gt.json", "--output", pilot, "--max-train-batches",
               str(args.smoke_batches), "--init-checkpoint", p["gt"][s] / "initial.pt", "--set", *as_sets(settings)]
        run(cmd, folder / "smoke.log")
        with open(pilot / "log.csv", newline="") as f:
            row = list(csv.DictReader(f))[-1]
        batches = -(-len(read_json(args.manifests / "train.json")["pairs"]) // args.batch_size)
        per_epoch = float(row["train_seconds"]) / args.smoke_batches * batches + float(row["val_seconds"])
        hours = per_epoch * args.epochs / 3600
        kd_hours += hours
        lines.append(f"{NAMES[s]} 증류: epoch당 약 {per_epoch:.1f}초 · {args.epochs} epoch 예상 {hm(hours)}")
    lines.append(f"전체 예상: 약 {hm(teacher_hours + kd_hours)} (짧은 점검 기준이라 대략적인 값)")
    lines.append(f"점검 결과 폴더: {folder} (본 학습에 쓰지 않음)")
    text = "\n".join(lines)
    say("점검 결과\n" + text)
    notifier.send("점검 완료 — 본 실행 준비됨", text)
    return text


# ---------------------------------------------------------------- 시작

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path.home() / "ondevice", help="Colab의 내 드라이브/ondevice와 같은 구조")
    ap.add_argument("--data-root", type=Path, help="이미 풀어둔 .../Real_captured (없으면 BASE/LOLv2.zip을 풂)")
    ap.add_argument("--manifests", type=Path, default=REPO / "manifests/lolv2_real")
    ap.add_argument("--students", nargs="+", default=["maxplus", "max", "mini"], choices=list(NAMES))
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--teacher-iters", type=int, default=30000)
    ap.add_argument("--experiment", default="kd_random_v1", help="증류 결과 폴더 이름 (runs/<이름>/...)")
    ap.add_argument("--skip-teacher", action="store_true", help="교사를 더 학습하지 않고 지금 best.pt로 증류")
    ap.add_argument("--smoke", action="store_true", help="짧은 점검 + 예상 시간 + 테스트 알림")
    ap.add_argument("--smoke-teacher-iters", type=int, default=30)
    ap.add_argument("--smoke-batches", type=int, default=4)
    ap.add_argument("--teacher-extra", nargs="*", default=[], help=argparse.SUPPRESS)  # 테스트용
    ap.add_argument("--kd-extra", nargs="*", default=[], help=argparse.SUPPRESS)  # 테스트용
    args = ap.parse_args()
    args.base = args.base.expanduser().resolve()
    args.teacher_extra, args.kd_extra = parse_sets(args.teacher_extra), parse_sets(args.kd_extra)
    args.test_mode = bool(args.teacher_extra or args.kd_extra)
    args.base.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):  # kill PID 로 끝내도 알림을 보내고 학습 프로세스를 정리한다
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)

    started = time.time()
    notifier = Notifier(args.base)
    p = paths(args)
    stage = "준비"
    try:
        say(f"BASE={args.base} · 저장소={REPO} · 알림={', '.join(notifier.channels())}")
        gpu = gpu_check(args)
        check_inputs(args, p)
        data_root = prepare_data(args)
        if args.smoke:
            smoke(args, p, data_root, gpu, notifier)
            return
        notifier.send("남은 학습 시작", f"GPU {gpu}\n순서: 교사 이어서 학습 → " +
                      " → ".join(f"{NAMES[s]} 증류" for s in args.students) + f"\nBASE {args.base}")
        stage = "교사 학습"
        summary = train_teacher(args, p, data_root)
        frozen = freeze_teacher(p, summary)
        notifier.send("교사 학습 완료", f"{summary.get('completed_iters')}/{summary.get('target_iters')}회 · "
                      f"최고 검증 {summary['best']['psnr']:.4f} dB ({summary['best']['iteration']}회)\n증류 교사: {frozen.name}")
        for s in args.students:
            stage = f"{NAMES[s]} 증류"
            result = distill(args, p, s, frozen, data_root)
            gt = best_of(p["gt"][s])
            diff = f" (일반 학습 {gt['psnr']:.4f} dB 대비 {result['best']['psnr'] - gt['psnr']:+.4f} dB)" if gt else ""
            notifier.send(f"{NAMES[s]} 증류 완료", f"최고 검증 {result['best']['psnr']:.4f} dB / {result['best']['ssim']:.4f} "
                          f"({result['best']['epoch']} epoch){diff}")
        stage = "결과 정리"
        text, bundle = report(args, p, frozen, gpu, started)
        say("결과\n" + text)
        notifier.send("남은 학습 4개 모두 완료", text, attachment=bundle)
    except KeyboardInterrupt:
        notifier.send(f"중단됨 ({stage})", "사용자가 중단했습니다. 같은 명령을 다시 실행하면 이어서 합니다.")
        raise
    except Exception as error:
        detail = "".join(traceback.format_exception_only(type(error), error))
        notifier.send(f"실패 ({stage})", f"{detail}\n같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 합니다.\n"
                      f"로그: {p['logs']}")
        raise


if __name__ == "__main__":
    main()
