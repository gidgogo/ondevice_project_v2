"""규칙 기반 선택기: 격자 탐색, 무작위 선택 기준, 검증 고정 후 적용."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_split(tmp_path, split, levels):
    rows, results = [], {m: [] for m in ("mini", "max", "maxplus")}
    for i, level in enumerate(levels):
        name = f"{split}_{i}"
        Image.fromarray(np.full((40, 60, 3), level, dtype=np.uint8)).save(tmp_path / f"{name}.png")
        rows.append({"name": name, "low": f"{name}.png", "high": f"{name}.png"})
        bright = level > 100
        # Mini는 밝은 사진에서만 Max+와 비슷하고, 어두운 사진에서는 크게 나쁘다.
        results["maxplus"].append({"name": name, "psnr": 22.0, "ssim": 0.8})
        results["max"].append({"name": name, "psnr": 21.5, "ssim": 0.78})
        results["mini"].append({"name": name, "psnr": 21.95 if bright else 17.0, "ssim": 0.7})
    (tmp_path / f"{split}.json").write_text(json.dumps({"split": split, "pairs": rows}))
    for m, r in results.items():
        (tmp_path / f"{split}_{m}.json").write_text(json.dumps({"per_image": r}))


def run(*args):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selector.py"), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def model_args(tmp_path, split):
    return [arg for m in ("mini", "max", "maxplus") for arg in (f"--{m}", str(tmp_path / f"{split}_{m}.json"))]


def test_fit_prefers_mini_for_bright_images_and_beats_random(tmp_path):
    make_split(tmp_path, "validation", [10, 20, 30, 40, 150, 160, 170, 180])
    out = tmp_path / "fit"
    run("fit", "--manifest", str(tmp_path / "validation.json"), "--data-root", str(tmp_path),
        *model_args(tmp_path, "validation"), "--cost", "mini=9", "max=45", "maxplus=55",
        "--selector-ms", "0.5", "--output", str(out))
    rule = json.loads((out / "rule.json").read_text())
    chosen = rule["chosen"]
    assert chosen["share_mini"] == 0.5 and chosen["share_maxplus"] + chosen["share_max"] == 0.5
    assert chosen["psnr"] >= rule["fixed_models"]["maxplus"]["psnr"] - 0.2
    assert chosen["gain_over_random_db"] > 0
    assert chosen["expected_ms"] < rule["fixed_models"]["maxplus"]["ms"]
    with open(out / "candidates.csv") as f:
        shares = {(float(r["share_mini"]), float(r["share_max"]), float(r["share_maxplus"])) for r in csv.DictReader(f)}
    assert {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)} <= shares  # 고정 모델 3개도 후보에 포함

    make_split(tmp_path, "test", [15, 25, 155, 175])
    report = tmp_path / "apply.json"
    run("apply", "--manifest", str(tmp_path / "test.json"), "--data-root", str(tmp_path),
        *model_args(tmp_path, "test"), "--rule", str(out / "rule.json"), "--output", str(report))
    applied = json.loads(report.read_text())
    assert applied["result"]["t1"] == chosen["t1"] and applied["quality_target_met"] is True


def test_fit_refuses_non_validation_split(tmp_path):
    make_split(tmp_path, "test", [10, 200])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selector.py"), "fit", "--manifest",
                             str(tmp_path / "test.json"), "--data-root", str(tmp_path), *model_args(tmp_path, "test"),
                             "--cost", "mini=1", "max=2", "maxplus=3", "--selector-ms", "0",
                             "--output", str(tmp_path / "x")], capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path / "x" / "rule.json").exists()


def test_fit_rejects_image_missing_from_all_three_results(tmp_path):
    make_split(tmp_path, "validation", [10, 20, 150, 160])
    for m in ("mini", "max", "maxplus"):  # 세 결과 모두에서 같은 이미지 하나를 빼도 거부해야 한다
        path = tmp_path / f"validation_{m}.json"
        rows = json.loads(path.read_text())["per_image"]
        path.write_text(json.dumps({"per_image": rows[1:]}))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selector.py"), "fit", "--manifest",
                             str(tmp_path / "validation.json"), "--data-root", str(tmp_path),
                             *model_args(tmp_path, "validation"), "--cost", "mini=1", "max=2", "maxplus=3",
                             "--selector-ms", "0", "--output", str(tmp_path / "x")], capture_output=True, text=True)
    assert result.returncode != 0 and "missing" in result.stderr
    assert not (tmp_path / "x" / "rule.json").exists()


def test_fit_rejects_report_from_another_split(tmp_path):
    make_split(tmp_path, "validation", [10, 20, 150, 160])
    path = tmp_path / "validation_max.json"
    path.write_text(json.dumps({"split": "test", **json.loads(path.read_text())}))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/selector.py"), "fit", "--manifest",
                             str(tmp_path / "validation.json"), "--data-root", str(tmp_path),
                             *model_args(tmp_path, "validation"), "--cost", "mini=1", "max=2", "maxplus=3",
                             "--selector-ms", "0", "--output", str(tmp_path / "x")], capture_output=True, text=True)
    assert result.returncode != 0 and "expected validation" in result.stderr
