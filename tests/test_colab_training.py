"""Exercise the actual training entrypoint used by the three Colab notebooks."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_data(tmp_path):
    rng = np.random.default_rng(91)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for split in ("train", "validation"):
        rows = []
        for i in range(2):
            row = {"name": f"{split}_{i}"}
            for kind, upper in (("low", 40), ("high", 230)):
                filename = f"{split}_{kind}_{i}.png"
                Image.fromarray(rng.integers(0, upper, (190, 200, 3), dtype=np.uint8)).save(tmp_path / filename)
                row[kind] = filename
            rows.append(row)
        (manifests / f"{split}.json").write_text(json.dumps({"split": split, "pairs": rows}))
    return [sys.executable, str(ROOT / "scripts/train.py"), "--data-root", str(tmp_path),
            "--manifests", str(manifests)]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)["weight"]


def assert_same(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("model", ["mini", "max", "maxplus"])
def test_three_models_train_and_resume_identically(tiny_data, tmp_path, model):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    options = ["--set", f'model="{model}"', "epochs=2", "batch_size=2", "lr_step=1", 'crop_mode="random"']
    run(tiny_data + ["--output", str(full)] + options)
    run(tiny_data + ["--output", str(resumed), "--stop-after-epoch", "1"] + options)
    assert json.loads((resumed / "summary.json").read_text())["training_complete"] is False
    run(tiny_data + ["--output", str(resumed), "--resume"] + options)
    for name in ("initial", "last", "best"):
        assert_same(weights(full / f"{name}.pt"), weights(resumed / f"{name}.pt"))
    with open(full / "log.csv") as a, open(resumed / "log.csv") as b:
        for left, right in zip(csv.DictReader(a), csv.DictReader(b)):
            for key in ("epoch", "loss", "kd", "val_psnr", "val_ssim", "lr", "images_seen"):
                assert left[key] == right[key]
    assert json.loads((resumed / "summary.json").read_text())["training_complete"] is True


@pytest.mark.parametrize("model", ["mini", "max", "maxplus"])
def test_teacher_does_not_change_initial_weights(tiny_data, tmp_path, model):
    # Compare saved starting points even though the training objectives differ.
    teacher, gt, kd, explicit = [tmp_path / n for n in ("teacher", "gt", "kd", "explicit")]
    run(tiny_data + ["--output", str(teacher), "--set", 'model="maxplus"', "epochs=1", "batch_size=2"])
    options = ["--set", f'model="{model}"', "epochs=1", "batch_size=2"]
    run(tiny_data + ["--output", str(gt)] + options)
    teacher_spec = json.dumps({"model": "maxplus", "checkpoint": str(teacher / "initial.pt")})
    run(tiny_data + ["--output", str(kd)] + options + ["beta=0.5", f"teacher={teacher_spec}"])
    assert_same(weights(gt / "initial.pt"), weights(kd / "initial.pt"))
    run(tiny_data + ["--output", str(explicit), "--init-checkpoint", str(gt / "initial.pt")]
        + options + ["seed=99", "beta=0.5", f"teacher={teacher_spec}"])
    assert_same(weights(gt / "initial.pt"), weights(explicit / "initial.pt"))
