"""고성능 교사(Retinexformer) 학습·평가·온라인 증류 경로 점검."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from teacher_models import build_teacher  # noqa: E402


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(5)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for split in ("train", "validation", "test"):
        rows = []
        for i in range(2):
            row = {"name": f"{split}_{i}"}
            for kind, upper in (("low", 40), ("high", 230)):
                filename = f"{split}_{kind}_{i}.png"
                Image.fromarray(rng.integers(0, upper, (190, 203, 3), dtype=np.uint8)).save(tmp_path / filename)
                row[kind] = filename
            rows.append(row)
        (manifests / f"{split}.json").write_text(json.dumps({"split": split, "pairs": rows}))
    return tmp_path, manifests


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT / "scripts")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def teacher_cmd(data, manifests, output, *extra):
    return [sys.executable, "train_teacher.py", "--data-root", str(data), "--manifests", str(manifests),
            "--output", str(output), "--set", "device=\"cpu\"", "crop=64", "batch_size=2", "total_iters=4",
            "val_every=2", *extra]


def test_vendored_retinexformer_is_pinned_and_sized():
    provenance = json.loads((ROOT / "provenance.json").read_text())["retinexformer"]
    for filename, expected in provenance["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    model = build_teacher("retinexformer")
    assert sum(p.numel() for p in model.parameters()) == 1_605_701
    with torch.no_grad():
        assert model(torch.rand(1, 3, 190, 203))[-1].shape == (1, 3, 190, 203)


def test_teacher_training_resumes_identically(tiny, tmp_path):
    data, manifests = tiny
    full, split = tmp_path / "full", tmp_path / "split"
    run(teacher_cmd(data, manifests, full))
    run(teacher_cmd(data, manifests, split, "--iters-per-session", "2"))
    assert json.loads((split / "summary.json").read_text())["training_complete"] is False
    run(teacher_cmd(data, manifests, split, "--resume", "--iters-per-session", "2"))
    for name in ("last", "best"):
        a = torch.load(full / f"{name}.pt", map_location="cpu", weights_only=False)["weight"]
        b = torch.load(split / f"{name}.pt", map_location="cpu", weights_only=False)["weight"]
        assert all(torch.equal(a[k], b[k]) for k in a)
    summary = json.loads((full / "summary.json").read_text())
    assert summary["training_complete"] is True and summary["official_test_used"] is False


def test_teacher_pilot_reports_time_estimate(tiny, tmp_path):
    data, manifests = tiny
    run(teacher_cmd(data, manifests, tmp_path / "pilot", "--pilot-iters", "3"))
    report = json.loads((tmp_path / "pilot" / "pilot_timing.json").read_text())
    assert report["seconds_per_iter"] > 0 and report["total_iters"] == 4


def test_online_retinexformer_distillation_and_evaluation(tiny, tmp_path):
    data, manifests = tiny
    teacher = tmp_path / "teacher"
    run(teacher_cmd(data, manifests, teacher))
    spec = json.dumps({"model": "retinexformer", "checkpoint": str(teacher / "best.pt")})
    student = tmp_path / "mini_strongkd"
    run([sys.executable, "train.py", "--data-root", str(data), "--manifests", str(manifests),
         "--output", str(student), "--set", 'model="mini"', "epochs=1", "batch_size=2",
         'crop_mode="random"', "beta=0.3", f"teacher={spec}"])
    run_json = json.loads((student / "run.json").read_text())
    assert run_json["teacher_checkpoint_sha256"] == hashlib.sha256((teacher / "best.pt").read_bytes()).hexdigest()
    report = tmp_path / "val.json"
    run([sys.executable, "eval_checkpoint.py", "--model", "retinexformer", "--checkpoint", str(teacher / "best.pt"),
         "--data-root", str(data), "--manifests", str(manifests), "--device", "cpu", "--output", str(report)])
    assert json.loads(report.read_text())["count"] == 2
    refused = subprocess.run([sys.executable, "eval_checkpoint.py", "--model", "mini", "--checkpoint",
                              str(student / "best.pt"), "--data-root", str(data), "--manifests", str(manifests),
                              "--split", "test", "--output", str(tmp_path / "t.json")],
                             capture_output=True, text=True, cwd=ROOT / "scripts")
    assert refused.returncode != 0 and not (tmp_path / "t.json").exists()


def test_teacher_restores_outputs_after_crash_following_checkpoint(tiny, tmp_path):
    # 마지막 checkpoint 저장 직후, best.pt·로그·완료 기록을 쓰기 전에 끊긴 상황
    data, manifests = tiny
    out = tmp_path / "crash"
    run(teacher_cmd(data, manifests, out))
    expected_best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["weight"]
    expected_log = (out / "log.csv").read_text()
    for name in ("best.pt", "best_validation_per_image.json", "log.csv"):
        (out / name).unlink()
    stale = json.loads((out / "summary.json").read_text())
    stale.update(completed_iters=2, training_complete=False, best=None)
    (out / "summary.json").write_text(json.dumps(stale))
    run(teacher_cmd(data, manifests, out, "--resume"))
    restored = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["weight"]
    assert all(torch.equal(expected_best[k], restored[k]) for k in expected_best)
    assert (out / "log.csv").read_text() == expected_log
    assert (out / "best_validation_per_image.json").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["training_complete"] is True and summary["completed_iters"] == 4


def _cell_names(source):
    import ast
    tree = ast.parse(source)
    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return stored, loaded


@pytest.mark.parametrize("builder,plan", [("teacher_notebook", "teacher_plan"), ("kd_notebook", "kd_plan")])
def test_run_cells_rebuild_commands_from_current_settings(builder, plan):
    # 점검 셀에서 만든 명령·경로를 본 학습 셀이 재사용하면, 점검 뒤 바꾼 설정(TOTAL_ITERS, BETA 등)이 무시된다.
    import package_colab
    nb = getattr(package_colab, builder)("0" * 40)
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    pilot, later = code[3], code[4:]
    pilot_stored, _ = _cell_names(pilot)
    for source in later:
        stored, loaded = _cell_names(source)
        assert f"{plan}()" in source
        reused = (loaded - stored) & pilot_stored
        assert not reused, f"run cell reuses names from the pilot cell: {reused}"
