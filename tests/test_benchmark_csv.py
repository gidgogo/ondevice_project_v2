"""CPU 측정 스크립트: 반복별 원자료 CSV(B 형식)와 학습된 가중치 불러오기."""
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from models import build_model  # noqa: E402


def test_benchmark_writes_per_repeat_csv(tmp_path):
    image = tmp_path / "low.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 50, (40, 60, 3), dtype=np.uint8)).save(image)
    ckpt = tmp_path / "maxplus.pt"
    torch.save({"weight": build_model("maxplus").state_dict()}, ckpt)
    out = tmp_path / "cpu_latency.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_cpu.py"), "--image", str(image),
                    "--models", "mini", "maxplus", "--warmup", "1", "--repeats", "3",
                    "--checkpoint", f"maxplus={ckpt}", "--csv", str(out)], check=True, capture_output=True)
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0]) == ["model_id", "checkpoint_status", "width", "height", "threads", "repeat_id",
                             "inference_ms", "total_ms"]
    assert len(rows) == 6 and {r["model_id"] for r in rows} == {"UltraFastLiNET_Mini", "UltraFastLiNET_MaxPlus"}
    status = {r["model_id"]: r["checkpoint_status"] for r in rows}
    assert status["UltraFastLiNET_MaxPlus"] == "trained:maxplus.pt" and status["UltraFastLiNET_Mini"].startswith("untrained")
    assert all(r["width"] == "60" and r["height"] == "40" and float(r["total_ms"]) > 0 for r in rows)
