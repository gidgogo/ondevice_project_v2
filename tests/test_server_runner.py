"""서버용 일괄 실행기(run_remaining.py): 교사 이어서 학습 → 교사 고정 → 증류 3개 → 결과 정리, 중단 후 재실행."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_data(tmp):
    rng = np.random.default_rng(5)
    root, manifests = tmp / "data" / "Real_captured", tmp / "manifests"
    manifests.mkdir(parents=True)
    for split, folder in (("train", "Train"), ("validation", "Train"), ("test", "Test")):
        rows = []
        for i in range(2):
            name = f"{split}_{i}.png"
            row = {"name": name}
            for kind, sub, upper in (("low", "Low", 40), ("high", "Normal", 230)):
                path = root / folder / sub / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, upper, (190, 203, 3), dtype=np.uint8)).save(path)
                row[kind] = f"{folder}/{sub}/{name}"
                row[f"{kind}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
        (manifests / f"{split}.json").write_text(json.dumps({"split": split, "pairs": rows}))
    return root, manifests


def test_runner_finishes_remaining_runs_and_skips_them_on_rerun(tmp_path):
    root, manifests = make_data(tmp_path)
    base = tmp_path / "base"
    common = ["--data-root", str(root), "--manifests", str(manifests)]
    for m in ("mini", "max", "maxplus"):  # Colab에서 끝난 일반 학습 대신
        subprocess.run([PY, "train.py", *common, "--config", str(ROOT / "configs/colab_gt.json"),
                        "--output", str(base / "runs/gt_random_v1" / f"{m}_gt_s42"),
                        "--set", f'model="{m}"', "seed=42", "epochs=2", "batch_size=2", 'device="cpu"'],
                       cwd=ROOT / "scripts", check=True, capture_output=True)
    teacher = base / "teachers/teacher_v1/retinexformer_s42"
    teacher_sets = ['device="cpu"', "crop=64", "batch_size=2", "val_every=2"]
    subprocess.run([PY, "train_teacher.py", *common, "--config", str(ROOT / "configs/teacher_retinexformer.json"),
                    "--output", str(teacher), "--iters-per-session", "2",  # Colab에서 중간까지 한 교사 대신
                    "--set", 'teacher="retinexformer"', "seed=42", "total_iters=4", *teacher_sets],
                   cwd=ROOT / "scripts", check=True, capture_output=True)
    runner = [PY, str(ROOT / "scripts/run_remaining.py"), "--base", str(base), *common, "--epochs", "2",
              "--batch-size", "2", "--teacher-iters", "4", "--teacher-extra", *teacher_sets, "--kd-extra", 'device="cpu"']
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}  # 알림 설정 없음 → 로그 파일만

    result = subprocess.run(runner, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert json.loads((teacher / "summary.json").read_text())["training_complete"] is True
    frozen = list((base / "teachers/frozen").glob("*.pt"))
    assert len(frozen) == 1
    for m in ("mini", "max", "maxplus"):
        run = base / "runs/kd_random_v1" / f"{m}_strongkd_b0.3_s42"
        assert json.loads((run / "summary.json").read_text())["training_complete"] is True
        meta = json.loads((run / "run.json").read_text())
        gt = json.loads((base / "runs/gt_random_v1" / f"{m}_gt_s42/run.json").read_text())
        assert meta["initial_checkpoint_source_sha256"] == gt["initial_sha256"]
        assert meta["teacher_checkpoint_sha256"] == hashlib.sha256(frozen[0].read_bytes()).hexdigest()
        same = lambda c: {k: v for k, v in c.items() if k not in ("teacher", "beta")}  # noqa: E731
        assert same(meta["config"]) == same(gt["config"]) and meta["config"]["beta"] == 0.3
    summary = (base / "server_results/summary.txt").read_text()
    assert "Retinexformer 교사 증류" in summary and len(list((base / "server_results").glob("*.zip"))) == 1
    log = (base / "notifications.log").read_text()
    assert "남은 학습 4개 모두 완료" in log

    again = subprocess.run(runner, capture_output=True, text=True, env=env)
    assert again.returncode == 0
    assert "Max+ 증류 이미 완료" in again.stdout and "실행: " not in again.stdout  # 끝난 단계는 다시 학습하지 않음
