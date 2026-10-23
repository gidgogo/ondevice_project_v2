"""Export three independent Colab notebooks plus this exact source snapshot.

No upload, training, or Git push is performed. Dataset/weights/runs are excluded.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {},
             "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def notebook(model, digest):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"name": f"train_{model}_gt.ipynb"},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            cell("markdown", f"""
                # {model.upper()} 일반 학습 · LOL-v2-real
                **런타임 → 런타임 유형 변경 → GPU** 선택 후 위에서 아래로 실행하세요.
                이 노트북은 {model} 하나만 학습합니다. Mini / Max / Max+는 각기 다른 런타임에서 실행합니다.

                1. Drive의 `내 드라이브/ondevice/`에 제공된 `colab_code.zip`과 기존 `LOLv2.zip`을 넣습니다.
                2. 아래 경로·실험 이름을 확인합니다. 세 노트북에서 공통 설정을 동일하게 유지하세요.
                3. 짧은 점검(2배치 + 검증 69장)이 성공하면 본 학습 셀을 실행합니다.

                공통 시작 설정은 **120 epoch / batch 40 / random crop 180 / seed 42**입니다.
                최종 연구 설정이 확정된 것은 아닙니다. 설정을 바꾸면 세 모델에 동일하게 적용하고 새 EXPERIMENT를 쓰세요.
                시험 100장은 학습·체크포인트 선택에 사용하지 않습니다.
                GPU 할당·연속 실행 시간은 보장되지 않습니다. 같은 모델·실험 폴더를 두 런타임에서 동시에 쓰지 마세요.
            """),
            cell("code", f'''
                from pathlib import Path
                MODEL = {model!r}  # 이 노트북의 담당 모델
                EXPERIMENT = "gt_random_v1"
                SEED = 42
                EPOCHS = 120
                EPOCHS_PER_SESSION = 120  # 기본: 남은 학습 끝까지. 짧게 점검할 때만 5~10으로 줄이세요.
                BATCH_SIZE = 40
                EXPECTED_CROP_MODE = "random"
                BASE = Path("/content/drive/MyDrive/ondevice")
                CODE_ZIP = BASE / "colab_code.zip"
                DATA_ZIP = BASE / "LOLv2.zip"
                OUTPUT = BASE / "runs" / EXPERIMENT / f"{{MODEL}}_gt_s{{SEED}}"
                EXPECTED_CODE_SHA256 = {digest!r}
            '''),
            cell("markdown", "## 1. Drive 연결 및 실행 환경 준비\nColab의 CUDA용 PyTorch는 유지합니다. 처음 연결할 때 Drive 접근을 승인하세요."),
            cell("code", '''
                from google.colab import drive
                drive.mount("/content/drive")
                import hashlib, json, os, shutil, subprocess, sys, zipfile
                import torch, torchvision, numpy
                from PIL import Image
                assert torch.cuda.is_available(), "GPU 런타임을 선택한 뒤 다시 실행하세요."
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

                def file_hash(path):
                    h = hashlib.sha256()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
                    return h.hexdigest()

                assert CODE_ZIP.is_file(), f"코드 ZIP을 Drive에 올려주세요: {CODE_ZIP}"
                assert file_hash(CODE_ZIP) == EXPECTED_CODE_SHA256, "노트북과 코드 ZIP 버전이 다릅니다. 같은 패키지를 사용하세요."

                def extract_checked(archive, destination):
                    destination.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(archive) as z:
                        for name in z.namelist():
                            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                                raise ValueError(f"잘못된 ZIP 경로: {name}")
                        z.extractall(destination)

                WORK = Path("/content/linet_gt")
                extract_checked(CODE_ZIP, WORK)
                REPO = WORK / "ondevice_project_v2"
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                                str(REPO / "requirements-colab.txt")], check=True)
                sys.path.insert(0, str(REPO / "scripts"))
                from models import build_model, count_parameters
                actual_config = json.loads((REPO / "configs/colab_gt.json").read_text())
                assert actual_config["crop_mode"] == EXPECTED_CROP_MODE and actual_config["crop"] == 180, "예상한 random 180 설정과 실행 코드가 다릅니다. 새 노트북을 사용하세요."
                print("학습 crop:", actual_config["crop_mode"], actual_config["crop"])
                print("GPU:", torch.cuda.get_device_name(0))
                print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
                print("모델:", MODEL, "파라미터:", count_parameters(build_model(MODEL)))
                print("저장 위치:", OUTPUT)
            '''),
            cell("markdown", "## 2. 데이터 준비\nDrive의 압축파일을 런타임 내부에 복사·해제합니다. 고정된 학습 620쌍·검증 69쌍의 파일 해시를 확인합니다. 시험 이미지는 읽지 않습니다."),
            cell("code", '''
                assert DATA_ZIP.is_file(), f"기존 데이터 ZIP을 Drive에 올려주세요: {DATA_ZIP}"
                LOCAL_ZIP = WORK / "LOLv2.zip"
                DATA_FOLDER = WORK / "data"
                DATA_MARKER = DATA_FOLDER / ".complete"
                EXPECTED_DATA_SHA256 = "298e00fece107b64fb54769ae8bc5175b4f84a3f7e8134e651a0a4b5eb61198d"
                if not DATA_MARKER.exists():
                    shutil.copy2(DATA_ZIP, LOCAL_ZIP)
                    assert file_hash(LOCAL_ZIP) == EXPECTED_DATA_SHA256, "기존 프로젝트의 LOLv2.zip과 파일이 다릅니다."
                    extract_checked(LOCAL_ZIP, DATA_FOLDER)
                    DATA_MARKER.write_text(EXPECTED_DATA_SHA256)
                roots = [p for p in DATA_FOLDER.rglob("Real_captured") if (p / "Train/Low").is_dir()]
                assert len(roots) == 1, f"Real_captured 폴더를 하나 찾을 수 있어야 합니다: {roots}"
                DATA_ROOT = roots[0]
                MANIFESTS = REPO / "manifests/lolv2_real"
                verified = set()
                for split, count in [("train", 620), ("validation", 69)]:
                    manifest = json.loads((MANIFESTS / f"{split}.json").read_text())
                    assert manifest["split"] == split and len(manifest["pairs"]) == count
                    for row in manifest["pairs"]:
                        for key in ("low", "high"):
                            path = DATA_ROOT / row[key]
                            if path not in verified:
                                assert file_hash(path) == row[f"{key}_sha256"], f"데이터 불일치: {path}"
                                verified.add(path)
                    print(split, count, "쌍 확인 완료")
            '''),
            cell("markdown", "## 3. 짧은 실행 점검\n본 학습과 같은 배치 크기로 2배치만 학습하고 전체 검증을 실행합니다. 결과는 별도 pilot 폴더이며 본 학습에 이어 쓰지 않습니다. 메모리 부족이면 BATCH_SIZE를 줄여 세 노트북에 동일하게 적용하세요."),
            cell("code", '''
                import uuid
                common = [sys.executable, "-u", str(REPO / "scripts/train.py"),
                          "--data-root", str(DATA_ROOT), "--manifests", str(MANIFESTS),
                          "--config", str(REPO / "configs/colab_gt.json")]
                settings = [f"model={json.dumps(MODEL)}", f"seed={SEED}", f"batch_size={BATCH_SIZE}"]
                PILOT = BASE / "pilots" / EXPERIMENT / f"{MODEL}_{uuid.uuid4().hex[:8]}"
                subprocess.run(common + ["--output", str(PILOT), "--max-train-batches", "2",
                               "--set", *settings, "epochs=1"], check=True, cwd=REPO)
                print("점검 성공. 본 학습은 새 초기 가중치에서 시작합니다.")
            '''),
            cell("markdown", "## 4. 남은 학습 실행 / 이어서 학습\n기본값은 EPOCHS_PER_SESSION=120이며 전체 목표까지 남은 학습을 진행합니다. 짧은 실험이 필요할 때만 5~10으로 줄이세요. 줄인 경우 구간이 끝나면 직접 이 셀을 다시 실행해야 합니다. 총 EPOCHS=120과 EXPERIMENT는 바꾸지 마세요. 매 epoch 종료 후 Drive에 저장하며, 끊겼을 때 진행 중이던 epoch는 다시 계산합니다. 새 런타임에서는 위의 준비 셀부터 다시 실행합니다. 첫 epoch 전에도 checkpoint를 저장하고, 최신 파일 손상 시 직전 저장본으로 복구합니다."),
            cell("code", '''
                assert isinstance(EPOCHS_PER_SESSION, int) and EPOCHS_PER_SESSION > 0
                resumable = any((OUTPUT / name).exists() for name in ("last.pt", "last.prev.pt"))
                if OUTPUT.exists() and any(OUTPUT.iterdir()) and not resumable:
                    raise RuntimeError("복구할 checkpoint가 없습니다. 기존 폴더를 보존하고 EXPERIMENT를 새 이름으로 바꾸세요.")
                overrides = [*settings, f"epochs={EPOCHS}"]
                effective = json.loads((REPO / "configs/colab_gt.json").read_text())
                effective.update(model=MODEL, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE)
                if (OUTPUT / "run.json").exists():
                    assert json.loads((OUTPUT / "run.json").read_text())["config"] == effective, "기존 실험과 설정이 다릅니다. 새 EXPERIMENT 이름을 사용하세요."
                # 환경 기록은 결과 폴더 밖에 두어 신규 학습의 빈 폴더 검사를 유지합니다.
                envdir = BASE / "environments" / EXPERIMENT / f"{MODEL}_s{SEED}"
                envdir.mkdir(parents=True, exist_ok=True)
                stamp = uuid.uuid4().hex[:8]
                (envdir / f"{stamp}_pip.txt").write_text(subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True))
                (envdir / f"{stamp}_gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
                cmd = common + ["--output", str(OUTPUT), "--epochs-per-session", str(EPOCHS_PER_SESSION)]
                if resumable:
                    cmd += ["--resume"]
                subprocess.run(cmd + ["--set", *overrides], check=True, cwd=REPO)
                progress = json.loads((OUTPUT / "progress.json").read_text())
                print(f"저장 완료: {progress['completed_epochs']}/{progress['target_epochs']} epoch")
                print("전체 완료" if progress["training_complete"] else "더 진행하려면 이 셀만 다시 실행하세요.")
            '''),
            cell("markdown", "## 5. 저장된 진행 상황·결과 확인\nprogress.json은 마지막으로 저장된 epoch를 나타내며 실시간 실행 여부를 뜻하지 않습니다. `best.pt`는 검증 PSNR로 선발한 모델, `last.pt`는 재개용, `last.prev.pt`는 직전 백업, `initial.pt`는 증류 비교용 초기값입니다. 최종 온디바이스 지연시간은 별도로 동일 노트북 CPU에서 측정합니다."),
            cell("code", '''
                import csv
                summary = json.loads((OUTPUT / "summary.json").read_text())
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                with open(OUTPUT / "log.csv") as f:
                    rows = list(csv.DictReader(f))
                for row in rows[-5:]:
                    print({k: row[k] for k in ["epoch", "val_psnr", "val_ssim", "train_seconds", "val_seconds"]})
                print("결과 폴더:", OUTPUT)
                print("다음 epoch:", json.loads((OUTPUT / "progress.json").read_text())["next_epoch"])
                print("증류 시작 가중치:", OUTPUT / "initial.pt")
                if MODEL == "maxplus":
                    print("Mini/Max 증류용 기본 교사 후보:", OUTPUT / "best.pt")
            '''),
        ],
    }


def github_notebook(model, ref):
    """Reuse the tested cells but fetch a fixed published Git commit instead of ZIP."""
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("--github-ref must be a full 40-character commit SHA")
    nb = notebook(model, "")
    nb["cells"][0]["source"] = nb["cells"][0]["source"].replace(
        "제공된 `colab_code.zip`과 기존 `LOLv2.zip`을", "기존 `LOLv2.zip`을")
    nb["cells"][0]["source"] += f"\n실행 코드는 GitHub 커밋 `{ref}`로 고정됩니다. 별도 코드 ZIP은 필요하지 않습니다.\n"
    settings = nb["cells"][1]["source"]
    settings = settings.replace('CODE_ZIP = BASE / "colab_code.zip"\n', "")
    settings = settings.replace("EXPECTED_CODE_SHA256 = ''", f"CODE_REF = {ref!r}")
    nb["cells"][1]["source"] = settings
    setup = nb["cells"][3]["source"]
    setup = setup.replace('assert CODE_ZIP.is_file(), f"코드 ZIP을 Drive에 올려주세요: {CODE_ZIP}"\n', "")
    setup = setup.replace('assert file_hash(CODE_ZIP) == EXPECTED_CODE_SHA256, "노트북과 코드 ZIP 버전이 다릅니다. 같은 패키지를 사용하세요."\n', "")
    setup = setup.replace('extract_checked(CODE_ZIP, WORK)\nREPO = WORK / "ondevice_project_v2"', textwrap.dedent('''
        WORK.mkdir(parents=True, exist_ok=True)
        REPO = WORK / f"source_{CODE_REF[:12]}"
        if not REPO.exists():
            subprocess.run(["git", "clone", "--no-checkout",
                            "https://github.com/gidgogo/ondevice_project_v2.git", str(REPO)], check=True)
            subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_REF], check=True)
        actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        assert actual == CODE_REF, "코드 버전이 다릅니다. 새 런타임에서 다시 시작하세요."
        print("실행 코드:", actual)
    ''').strip())
    nb["cells"][3]["source"] = setup
    return nb


def write_notebook(nb, path, model):
    for index, item in enumerate(nb["cells"]):
        item["id"] = f"{model}-{index:02d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n")


SETUP_STAGE2 = """
from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, os, shutil, subprocess, sys, uuid, zipfile
import torch
assert torch.cuda.is_available(), "GPU 런타임을 선택한 뒤 다시 실행하세요."
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_checked(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"잘못된 ZIP 경로: {name}")
        z.extractall(destination)

WORK = Path("/content/linet_gt")
WORK.mkdir(parents=True, exist_ok=True)
REPO = WORK / f"source_{CODE_REF[:12]}"
if not REPO.exists():
    subprocess.run(["git", "clone", "--no-checkout",
                    "https://github.com/gidgogo/ondevice_project_v2.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_REF], check=True)
actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert actual == CODE_REF, "코드 버전이 다릅니다. 새 런타임에서 다시 시작하세요."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(REPO / "requirements-colab.txt")], check=True)
sys.path.insert(0, str(REPO / "scripts"))
print("실행 코드:", actual)
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
"""


def _data_cell():
    # GT 노트북과 같은 데이터 준비·해시 검사 셀을 그대로 사용한다.
    return notebook("max", "")["cells"][5]["source"]


def _stage2_notebook(name, title, intro, settings, ref, extra_setup, steps):
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("--github-ref must be a full 40-character commit SHA")
    cells = [cell("markdown", f"# {title}\n" + textwrap.dedent(intro).strip()
                  + f"\n\n실행 코드는 GitHub 커밋 `{ref}`로 고정됩니다.\n"),
             cell("code", textwrap.dedent(settings).strip() + f"\nCODE_REF = {ref!r}\n"),
             cell("markdown", "## 1. Drive 연결 및 실행 환경 준비"),
             cell("code", SETUP_STAGE2.strip() + "\n" + textwrap.dedent(extra_setup).strip() + "\n"),
             cell("markdown", "## 2. 데이터 준비\n학습 620쌍·검증 69쌍의 파일 해시를 확인합니다. 시험 이미지는 읽지 않습니다."),
             cell("code", _data_cell())]
    for heading, code in steps:
        cells += [cell("markdown", heading), cell("code", code)]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"name": f"{name}.ipynb"},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": cells}


def teacher_notebook(ref):
    return _stage2_notebook("train_teacher_retinexformer", "고성능 교사 학습 · Retinexformer · LOL-v2-real", """
        **런타임 → 런타임 유형 변경 → GPU** 선택 후 위에서 아래로 실행하세요.
        학생 증류에 쓸 고성능 교사(Retinexformer, 학습 파라미터 1,605,701개)를 **우리 학습 620쌍으로 처음부터** 학습합니다.
        공개 사전학습 가중치는 쓰지 않습니다(우리 검증 분할이 포함된 데이터로 학습되었을 수 있음).

        1. 짧은 점검이 반복당 시간과 **TOTAL_ITERS 전체의 예상 시간**을 출력합니다.
        2. 예상 시간을 보고 TOTAL_ITERS를 정한 뒤 본 학습을 시작합니다. 시작 후 바꾸려면 EXPERIMENT도 새 이름으로 바꿉니다.
        3. 끝나면 검증 69장 점수를 Max+ 일반 학습 결과와 비교합니다. **Max+보다 확실히 좋아야** 고성능 교사로 씁니다.
    """, """
        from pathlib import Path
        TEACHER = "retinexformer"
        EXPERIMENT = "teacher_v1"
        SEED = 42
        TOTAL_ITERS = 30000  # 점검 결과의 예상 시간을 보고 본 학습 전에 확정
        ITERS_PER_SESSION = TOTAL_ITERS  # 기본: 끝까지. 짧게 나눌 때만 줄이세요(1000 단위 저장).
        PILOT_ITERS = 100
        BASE = Path("/content/drive/MyDrive/ondevice")
        DATA_ZIP = BASE / "LOLv2.zip"
        OUTPUT = BASE / "teachers" / EXPERIMENT / f"{TEACHER}_s{SEED}"
        MAXPLUS_GT_SUMMARY = BASE / "runs" / "gt_random_v1" / "maxplus_gt_s42" / "summary.json"
    """, ref, """
        CONFIG = REPO / "configs/teacher_retinexformer.json"
        from teacher_models import build_teacher
        print("교사 파라미터:", sum(p.numel() for p in build_teacher(TEACHER).parameters()))
        print("설정:", json.loads(CONFIG.read_text()))
        print("저장 위치:", OUTPUT)
    """, [
        ("## 3. 짧은 실행 점검과 예상 시간\nPILOT_ITERS만 학습하고 검증 69장을 한 번 실행합니다. 결과는 pilots 폴더에 따로 저장합니다.", """
            common = [sys.executable, "-u", str(REPO / "scripts/train_teacher.py"),
                      "--data-root", str(DATA_ROOT), "--manifests", str(MANIFESTS), "--config", str(CONFIG)]
            settings = [f"teacher={json.dumps(TEACHER)}", f"seed={SEED}", f"total_iters={TOTAL_ITERS}"]
            PILOT = BASE / "pilots" / EXPERIMENT / f"{TEACHER}_{uuid.uuid4().hex[:8]}"
            subprocess.run(common + ["--output", str(PILOT), "--pilot-iters", str(PILOT_ITERS), "--set", *settings],
                           check=True, cwd=REPO / "scripts")
            timing = json.loads((PILOT / "pilot_timing.json").read_text())
            print(f"반복당 {timing['seconds_per_iter']:.3f}초 · TOTAL_ITERS={TOTAL_ITERS} 예상 {timing['estimated_hours_for_total_iters']:.1f}시간 ({timing['gpu']})")
        """),
        ("## 4. 본 학습 / 이어서 학습\n1000회마다 검증·저장합니다. 끊기면 준비 셀부터 다시 실행하고 이 셀을 실행하면 마지막 저장 지점부터 이어갑니다.", """
            resumable = any((OUTPUT / name).exists() for name in ("last.pt", "last.prev.pt"))
            if OUTPUT.exists() and any(OUTPUT.iterdir()) and not resumable:
                raise RuntimeError("복구할 checkpoint가 없습니다. EXPERIMENT를 새 이름으로 바꾸세요.")
            effective = json.loads(CONFIG.read_text())
            effective.update(teacher=TEACHER, seed=SEED, total_iters=TOTAL_ITERS)
            if (OUTPUT / "run.json").exists():
                assert json.loads((OUTPUT / "run.json").read_text())["config"] == effective, "기존 실험과 설정이 다릅니다. 새 EXPERIMENT 이름을 사용하세요."
            cmd = common + ["--output", str(OUTPUT), "--iters-per-session", str(ITERS_PER_SESSION)]
            if resumable:
                cmd += ["--resume"]
            subprocess.run(cmd + ["--set", *settings], check=True, cwd=REPO / "scripts")
            summary = json.loads((OUTPUT / "summary.json").read_text())
            print(f"저장 완료: {summary['completed_iters']}/{summary['target_iters']} · 최고 검증 {summary['best']}")
        """),
        ("## 5. 검증 69장 평가와 Max+ 일반 학습 비교\n고성능 교사로 쓰려면 검증 PSNR이 Max+ 일반 학습보다 확실히 높아야 합니다. 시험 분할은 평가하지 않습니다.", """
            REPORT = OUTPUT / "validation_report.json"
            subprocess.run([sys.executable, str(REPO / "scripts/eval_checkpoint.py"), "--model", TEACHER,
                            "--checkpoint", str(OUTPUT / "best.pt"), "--data-root", str(DATA_ROOT),
                            "--manifests", str(MANIFESTS), "--device", "cuda", "--output", str(REPORT)],
                           check=True, cwd=REPO / "scripts")
            teacher = json.loads(REPORT.read_text())["mean"]
            print("교사 검증:", teacher)
            if MAXPLUS_GT_SUMMARY.exists():
                maxplus = json.loads(MAXPLUS_GT_SUMMARY.read_text())["best"]
                print(f"교사 - Max+ 일반 학습: PSNR {teacher['psnr'] - maxplus['psnr']:+.2f} dB, SSIM {teacher['ssim'] - maxplus['ssim']:+.4f}")
            else:
                print("Max+ 일반 학습 결과가 이 Drive에 없습니다. summary.json을 받아 MAXPLUS_GT_SUMMARY 경로에 두세요.")
            print("증류에 쓸 교사 checkpoint:", OUTPUT / "best.pt")
        """),
    ])


def kd_notebook(ref):
    return _stage2_notebook("train_kd", "지식 증류 학습 · 학생 1개 × 교사 1개 · LOL-v2-real", """
        **런타임 → 런타임 유형 변경 → GPU** 선택 후 위에서 아래로 실행하세요.
        같은 학생 구조를 **일반 학습과 같은 initial.pt, 같은 설정·epoch**로 시작하고, 손실에만 교사 출력 항(β)을 더합니다.

        | TEACHER_KIND | 교사 | 가능한 학생 |
        |---|---|---|
        | `base` | 일반 학습 Max+ (`gt_random_v1/maxplus_gt_s42/best.pt`) | mini, max |
        | `strong` | Retinexformer (`teachers/teacher_v1/retinexformer_s42/best.pt`) | mini, max, maxplus |

        준비물(이 노트북을 실행하는 사람의 Drive 기준 경로): 학생의 일반 학습 폴더(`initial.pt`, `run.json`, `summary.json`)와 교사 `best.pt`.
        다른 팀원이 학습했다면 같은 경로로 복사해 둡니다. 교사는 학생과 **같은 random 패치**에서 매 배치 계산합니다.
        β 후보는 0.1 / 0.3 / 1.0이며 검증으로 고르고, 고른 뒤에는 고정합니다.
    """, """
        from pathlib import Path
        STUDENT = "mini"        # mini / max / maxplus
        TEACHER_KIND = "base"   # base: Max+ 일반 학습 / strong: Retinexformer
        BETA = 0.3              # 후보 0.1 / 0.3 / 1.0
        SEED = 42
        EPOCHS = 120
        EPOCHS_PER_SESSION = 120
        BATCH_SIZE = 40
        GT_EXPERIMENT = "gt_random_v1"
        TEACHER_EXPERIMENT = "teacher_v1"
        EXPERIMENT = "kd_random_v1"
        BASE = Path("/content/drive/MyDrive/ondevice")
        DATA_ZIP = BASE / "LOLv2.zip"
        GT_RUN = BASE / "runs" / GT_EXPERIMENT / f"{STUDENT}_gt_s{SEED}"
        TEACHERS = {"base": ("maxplus", BASE / "runs" / GT_EXPERIMENT / "maxplus_gt_s42" / "best.pt"),
                    "strong": ("retinexformer", BASE / "teachers" / TEACHER_EXPERIMENT / "retinexformer_s42" / "best.pt")}
        OUTPUT = BASE / "runs" / EXPERIMENT / f"{STUDENT}_{TEACHER_KIND}kd_b{BETA}_s{SEED}"
    """, ref, """
        assert STUDENT in ("mini", "max", "maxplus") and TEACHER_KIND in TEACHERS
        assert not (STUDENT == "maxplus" and TEACHER_KIND == "base"), "Max+는 자기 자신(Max+ 일반 학습)을 교사로 쓰지 않습니다."
        assert BETA > 0
        TEACHER_MODEL, TEACHER_CKPT = TEACHERS[TEACHER_KIND]
        INITIAL = GT_RUN / "initial.pt"
        for path in (INITIAL, GT_RUN / "run.json", TEACHER_CKPT):
            assert path.is_file(), f"준비 파일이 없습니다: {path}"
        CONFIG = REPO / "configs/colab_gt.json"
        base_config = json.loads(CONFIG.read_text())
        base_config.update(model=STUDENT, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE)
        gt_config = json.loads((GT_RUN / "run.json").read_text())["config"]
        assert gt_config == base_config, f"일반 학습과 설정이 다릅니다. 같은 조건으로 비교해야 합니다.\\nGT: {gt_config}\\n지금: {base_config}"
        TEACHER_SPEC = {"model": TEACHER_MODEL, "checkpoint": str(TEACHER_CKPT)}
        print("학생:", STUDENT, "· 교사:", TEACHER_KIND, TEACHER_MODEL, "· β:", BETA)
        print("일반 학습 초기값:", INITIAL, file_hash(INITIAL)[:12])
        print("교사:", TEACHER_CKPT, file_hash(TEACHER_CKPT)[:12])
        print("저장 위치:", OUTPUT)
    """, [
        ("## 3. 짧은 실행 점검\n2배치 학습 + 검증 69장. 교사 추론이 포함된 시간을 확인합니다.", """
            common = [sys.executable, "-u", str(REPO / "scripts/train.py"),
                      "--data-root", str(DATA_ROOT), "--manifests", str(MANIFESTS), "--config", str(CONFIG)]
            settings = [f"model={json.dumps(STUDENT)}", f"seed={SEED}", f"batch_size={BATCH_SIZE}",
                        f"beta={BETA}", f"teacher={json.dumps(TEACHER_SPEC)}"]
            PILOT = BASE / "pilots" / EXPERIMENT / f"{STUDENT}_{TEACHER_KIND}_{uuid.uuid4().hex[:8]}"
            subprocess.run(common + ["--output", str(PILOT), "--max-train-batches", "2", "--init-checkpoint", str(INITIAL),
                                     "--set", *settings, "epochs=1"], check=True, cwd=REPO)
            print("점검 성공")
        """),
        ("## 4. 본 학습 / 이어서 학습\n처음에는 일반 학습의 initial.pt에서 시작하고, 이후에는 마지막 저장 epoch에서 이어갑니다.", """
            resumable = any((OUTPUT / name).exists() for name in ("last.pt", "last.prev.pt"))
            if OUTPUT.exists() and any(OUTPUT.iterdir()) and not resumable:
                raise RuntimeError("복구할 checkpoint가 없습니다. EXPERIMENT를 새 이름으로 바꾸세요.")
            effective = dict(base_config, beta=BETA, teacher=TEACHER_SPEC)
            if (OUTPUT / "run.json").exists():
                assert json.loads((OUTPUT / "run.json").read_text())["config"] == effective, "기존 실험과 설정이 다릅니다. 새 EXPERIMENT 이름을 사용하세요."
            cmd = common + ["--output", str(OUTPUT), "--epochs-per-session", str(EPOCHS_PER_SESSION)]
            cmd += ["--resume"] if resumable else ["--init-checkpoint", str(INITIAL)]
            subprocess.run(cmd + ["--set", *settings, f"epochs={EPOCHS}"], check=True, cwd=REPO)
            progress = json.loads((OUTPUT / "progress.json").read_text())
            print(f"저장 완료: {progress['completed_epochs']}/{progress['target_epochs']} epoch")
        """),
        ("## 5. 일반 학습과 비교 (검증 69장)\n같은 학생·같은 초기값·같은 epoch에서 증류만 다릅니다. 시험 분할은 쓰지 않습니다.", """
            kd = json.loads((OUTPUT / "summary.json").read_text())
            print("증류:", kd["best"])
            gt_summary = GT_RUN / "summary.json"
            if gt_summary.exists() and json.loads(gt_summary.read_text()).get("training_complete"):
                gt = json.loads(gt_summary.read_text())["best"]
                print("일반 학습:", gt)
                if kd.get("training_complete"):
                    print(f"증류 - 일반 학습: PSNR {kd['best']['psnr'] - gt['psnr']:+.3f} dB, SSIM {kd['best']['ssim'] - gt['ssim']:+.4f}")
            else:
                print("일반 학습이 아직 완료되지 않았거나 summary.json이 없습니다.")
        """),
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-ref", help="Emit model folders using this fixed GitHub commit instead of a code ZIP")
    p.add_argument("--stage", choices=["gt", "kd"], default="gt",
                   help="gt: 일반 학습 노트북 3개 / kd: 교사 학습·증류 노트북 2개 (GitHub 커밋 고정 전용)")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "kd":
        if not args.github_ref:
            p.error("--stage kd requires --github-ref")
        write_notebook(teacher_notebook(args.github_ref), args.output / "04_teacher" / "train.ipynb", "teacher")
        write_notebook(kd_notebook(args.github_ref), args.output / "05_kd" / "train.ipynb", "kd")
        print(args.output)
        return
    if args.github_ref:
        for index, model in enumerate(("mini", "max", "maxplus"), 1):
            write_notebook(github_notebook(model, args.github_ref),
                           args.output / f"{index:02d}_{model}" / "train.ipynb", model)
        print(args.output)
        return
    archive = args.output / "colab_code.zip"
    files = [ROOT / name for name in ("README.md", "LICENSE", "provenance.json", "requirements-colab.txt")]
    for folder in ("scripts", "vendor", "vendor_retinexformer", "configs", "manifests", "docs", "tests"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, Path("ondevice_project_v2") / path.relative_to(ROOT))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    for model in ("mini", "max", "maxplus"):
        nb = notebook(model, digest)
        write_notebook(nb, args.output / f"train_{model}_gt.ipynb", model)
    (args.output / "SHA256.txt").write_text(f"{digest}  colab_code.zip\n")
    print(args.output)


if __name__ == "__main__":
    main()
