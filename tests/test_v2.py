"""Checks for the LOL-v2-real pipeline: models, split, crops, evaluation guard, KD training."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation import evaluate  # noqa: E402
from models import build_model, count_parameters  # noqa: E402
from prepare_lolv2 import choose_validation  # noqa: E402
from train import PairCrops  # noqa: E402

torch.set_num_threads(1)


def test_parameter_counts():
    assert {n: count_parameters(build_model(n)) for n in ("mini", "max", "maxplus")} == \
        {"mini": 36, "max": 180, "maxplus": 937}


def test_maxplus_starts_as_max():
    torch.manual_seed(0)
    plain = build_model("max")
    torch.manual_seed(0)
    plus = build_model("maxplus")
    x = torch.rand(1, 3, 181, 183) * .2
    with torch.inference_mode():
        assert torch.equal(plain(x)[-1], plus(x)[-1])
        assert [o.shape for o in plain(x)] == [o.shape for o in plus(x)]


def test_validation_choice_keeps_scene_groups_whole():
    groups = [[0, 1, 2, 3], [4, 5], [6], [7, 8, 9], [10, 11]]
    chosen = choose_validation(groups, seed=42, count=5)
    assert choose_validation(groups, seed=42, count=5) == chosen and len(chosen) == 5
    assert all(set(g) <= chosen or not (set(g) & chosen) for g in groups)
    with pytest.raises(ValueError):
        choose_validation([[0, 1, 2]], seed=0, count=2)


def _write_pairs(tmp_path, split, n=3, size=(200, 190)):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        for kind in ("Low", "Normal"):
            (tmp_path / kind).mkdir(exist_ok=True)
            arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            if kind == "Normal":
                arr = low  # identical content makes crop alignment checkable
            else:
                low = arr
            Image.fromarray(arr).save(tmp_path / kind / f"{i:05d}.png")
        rows.append({"name": f"{i:05d}.png", "low": f"Low/{i:05d}.png", "high": f"Normal/{i:05d}.png",
                     "low_sha256": f"l{i}", "high_sha256": f"h{i}"})
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps({"dataset": "test", "split": split, "pairs": rows}))
    return path


def test_random_crops_are_paired_and_reproducible(tmp_path):
    manifest = _write_pairs(tmp_path, "train")
    data = PairCrops(manifest, tmp_path, 180, "random", seed=1)
    x, y = data[0]
    assert x.shape == (3, 180, 180) and torch.equal(x, y)
    assert torch.equal(data[0][0], x)
    data.epoch = 1
    assert not torch.equal(data[0][0], x)


def test_training_data_must_be_train_split(tmp_path):
    with pytest.raises(ValueError):
        PairCrops(_write_pairs(tmp_path, "validation"), tmp_path, 180, "center")


def test_test_split_needs_explicit_opt_in(tmp_path):
    manifest = _write_pairs(tmp_path, "test", n=1)
    with pytest.raises(PermissionError):
        evaluate(build_model("mini"), manifest, tmp_path, use_lpips=False)
    scores, rows = evaluate(build_model("mini"), manifest, tmp_path, use_lpips=False, allow_test=True)
    assert len(rows) == 1 and set(scores) == {"psnr", "ssim"}


def test_kd_training_end_to_end(tmp_path):
    for split in ("train", "validation"):
        target = tmp_path / "manifests"
        target.mkdir(exist_ok=True)
        _write_pairs(tmp_path, split, n=2).rename(target / f"{split}.json")
    teacher = tmp_path / "teacher.pt"
    torch.save({"weight": build_model("max").state_dict()}, teacher)
    command = [sys.executable, str(ROOT / "scripts/train.py"), "--data-root", str(tmp_path),
               "--manifests", str(tmp_path / "manifests"), "--output", str(tmp_path / "run"),
               "--set", 'model="mini"', "epochs=2", "batch_size=2", "beta=0.5",
               f'teacher={{"model":"max","checkpoint":"{teacher}"}}']
    subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT / "scripts")
    log = (tmp_path / "run/log.csv").read_text().splitlines()
    assert len(log) == 3 and float(log[1].split(",")[2]) > 0  # kd column is non-zero
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["parameters"] == 36 and summary["official_test_used"] is False
    best = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert set(best["weight"]) == set(build_model("mini").state_dict())
