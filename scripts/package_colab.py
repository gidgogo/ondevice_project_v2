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

                기본값은 기존 코드의 **120 epoch / batch 40 / crop 180 / seed 42**입니다.
                최종 연구 설정이 확정된 것은 아닙니다. 설정을 바꾸면 세 모델에 동일하게 적용하고 새 EXPERIMENT를 쓰세요.
                시험 100장은 학습·체크포인트 선택에 사용하지 않습니다.
                GPU 할당·연속 실행 시간은 보장되지 않습니다. 같은 모델·실험 폴더를 두 런타임에서 동시에 쓰지 마세요.
            """),
            cell("code", f'''
                from pathlib import Path
                MODEL = {model!r}  # 이 노트북의 담당 모델
                EXPERIMENT = "gt_v1"
                SEED = 42
                EPOCHS = 120
                BATCH_SIZE = 40
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
            cell("markdown", "## 4. 본 학습 / 이어서 학습\n각 epoch 종료 후 Drive에 저장합니다. 재접속 시 위의 준비 셀부터 실행하고 이 셀을 실행하면 마지막 완료 epoch에서 재개합니다. 진행 중이던 epoch는 다시 계산합니다. 완료된 학습은 건너뜁니다."),
            cell("code", '''
                if OUTPUT.exists() and any(OUTPUT.iterdir()) and not (OUTPUT / "last.pt").exists():
                    raise RuntimeError("저장된 첫 epoch가 없습니다. 기존 폴더를 보존하고 EXPERIMENT를 새 이름으로 바꾸세요.")
                overrides = [*settings, f"epochs={EPOCHS}"]
                effective = json.loads((REPO / "configs/colab_gt.json").read_text())
                effective.update(model=MODEL, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE)
                if (OUTPUT / "run.json").exists():
                    assert json.loads((OUTPUT / "run.json").read_text())["config"] == effective, "기존 실험과 설정이 다릅니다. 새 EXPERIMENT 이름을 사용하세요."
                SUMMARY = OUTPUT / "summary.json"
                finished = SUMMARY.exists() and json.loads(SUMMARY.read_text()).get("training_complete", False)
                if finished:
                    print("이미 완료된 실험입니다:", OUTPUT)
                else:
                    # 환경 기록은 결과 폴더 밖에 두어 신규 학습의 빈 폴더 검사를 유지합니다.
                    envdir = BASE / "environments" / EXPERIMENT / f"{MODEL}_s{SEED}"
                    envdir.mkdir(parents=True, exist_ok=True)
                    stamp = uuid.uuid4().hex[:8]
                    (envdir / f"{stamp}_pip.txt").write_text(subprocess.check_output(
                        [sys.executable, "-m", "pip", "freeze"], text=True))
                    (envdir / f"{stamp}_gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
                    cmd = common + ["--output", str(OUTPUT)]
                    if (OUTPUT / "last.pt").exists():
                        cmd += ["--resume"]
                    subprocess.run(cmd + ["--set", *overrides], check=True, cwd=REPO)
            '''),
            cell("markdown", "## 5. 결과 확인\n`best.pt`는 검증 PSNR로 선발한 모델, `last.pt`는 재개용, `initial.pt`는 후속 증류와 초기값을 맞추기 위한 파일입니다. 최종 온디바이스 지연시간은 별도로 동일 노트북 CPU에서 측정합니다."),
            cell("code", '''
                import csv
                summary = json.loads((OUTPUT / "summary.json").read_text())
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                with open(OUTPUT / "log.csv") as f:
                    rows = list(csv.DictReader(f))
                for row in rows[-5:]:
                    print({k: row[k] for k in ["epoch", "val_psnr", "val_ssim", "train_seconds", "val_seconds"]})
                print("결과 폴더:", OUTPUT)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-ref", help="Emit model folders using this fixed GitHub commit instead of a code ZIP")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.github_ref:
        for index, model in enumerate(("mini", "max", "maxplus"), 1):
            write_notebook(github_notebook(model, args.github_ref),
                           args.output / f"{index:02d}_{model}" / "train.ipynb", model)
        print(args.output)
        return
    archive = args.output / "colab_code.zip"
    files = [ROOT / name for name in ("README.md", "LICENSE", "provenance.json", "requirements-colab.txt")]
    for folder in ("scripts", "vendor", "configs", "manifests", "docs", "tests"):
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
