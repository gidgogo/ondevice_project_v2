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
    run(tiny_data + ["--output", str(resumed), "--epochs-per-session", "1"] + options)
    assert json.loads((resumed / "summary.json").read_text())["training_complete"] is False
    assert json.loads((resumed / "progress.json").read_text())["next_epoch"] == 2
    run(tiny_data + ["--output", str(resumed), "--resume", "--epochs-per-session", "1"] + options)
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


def test_first_epoch_interruption_can_resume(tiny_data, tmp_path):
    output, full = tmp_path / "paused", tmp_path / "full"
    options = ["--set", 'model="mini"', "epochs=1", "batch_size=2", 'crop_mode="random"']
    run(tiny_data + ["--output", str(output), "--stop-after-epoch", "0"] + options)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_epochs"] == 0 and summary["best"] is None
    assert not (output / "best.pt").exists()
    run(tiny_data + ["--output", str(output), "--resume"] + options)
    run(tiny_data + ["--output", str(full)] + options)
    assert_same(weights(output / "last.pt"), weights(full / "last.pt"))


def test_corrupt_latest_falls_back_and_repairs_logs(tiny_data, tmp_path):
    output, full = tmp_path / "paused", tmp_path / "full"
    options = ["--set", 'model="mini"', "epochs=3", "batch_size=2", "lr_step=1", 'crop_mode="random"']
    run(tiny_data + ["--output", str(full)] + options)
    run(tiny_data + ["--output", str(output), "--epochs-per-session", "2"] + options)
    assert torch.load(output / "last.prev.pt", weights_only=True)["epoch"] == 1
    (output / "last.pt").write_bytes(b"interrupted checkpoint write")
    (output / "best.pt").write_bytes(b"broken derived file")
    (output / "log.csv").write_text("partial line")
    (output / "summary.json").write_text("{")
    run(tiny_data + ["--output", str(output), "--resume", "--epochs-per-session", "1"] + options)
    # Session length counts from recovered epoch 1, not from stale summary epoch 2.
    assert json.loads((output / "summary.json").read_text())["completed_epochs"] == 2
    run(tiny_data + ["--output", str(output), "--resume", "--epochs-per-session", "1"] + options)
    for name in ("initial", "best", "last"):
        assert_same(weights(output / f"{name}.pt"), weights(full / f"{name}.pt"))
    with open(output / "log.csv") as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["1", "2", "3"]
    assert json.loads((output / "progress.json").read_text())["training_complete"]
    # Rerunning a completed job neither trains nor duplicates log rows.
    previous_log = (output / "log.csv").read_bytes()
    run(tiny_data + ["--output", str(output), "--resume", "--epochs-per-session", "1"] + options)
    assert (output / "log.csv").read_bytes() == previous_log


def test_legacy_epoch_checkpoint_still_resumes(tiny_data, tmp_path):
    output, full = tmp_path / "legacy", tmp_path / "full"
    options = ["--set", 'model="mini"', "epochs=2", "batch_size=2", 'crop_mode="random"']
    run(tiny_data + ["--output", str(full)] + options)
    run(tiny_data + ["--output", str(output), "--epochs-per-session", "1"] + options)
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    for key in ("format_version", "history", "best_checkpoint"):
        checkpoint.pop(key)
    torch.save(checkpoint, output / "last.pt")
    run(tiny_data + ["--output", str(output), "--resume", "--epochs-per-session", "1"] + options)
    assert_same(weights(output / "last.pt"), weights(full / "last.pt"))


def test_process_dies_after_checkpoint_before_log_export(tiny_data, tmp_path):
    output = tmp_path / "interrupted"
    harness = tmp_path / "crash_after_save.py"
    harness.write_text('''
import importlib.util, os, sys
from pathlib import Path
entry = sys.argv.pop(1)
sys.path.insert(0, str(Path(entry).parent))
spec = importlib.util.spec_from_file_location("project_trainer", entry)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
original = trainer.atomic_save
def fail_after_save(value, path, **kwargs):
    original(value, path, **kwargs)
    if Path(path).name == "last.pt" and value.get("epoch") == 1:
        os._exit(99)
trainer.atomic_save = fail_after_save
trainer.main()
''')
    options = ["--output", str(output), "--set", 'model="mini"', "epochs=2", "batch_size=2"]
    crashed = subprocess.run([tiny_data[0], str(harness), *tiny_data[1:], *options], capture_output=True, text=True)
    assert crashed.returncode == 99, crashed.stderr
    assert torch.load(output / "last.pt", weights_only=True)["epoch"] == 1
    assert json.loads((output / "summary.json").read_text())["completed_epochs"] == 0
    run(tiny_data + ["--resume", "--epochs-per-session", "1"] + options)
    with open(output / "log.csv") as f:
        assert [r["epoch"] for r in csv.DictReader(f)] == ["1", "2"]
    assert (output / "best.pt").exists()
