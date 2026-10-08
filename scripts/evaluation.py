"""Quality evaluation shared by training (validation) and final testing.

Protocol: RGB float32 in [0, 1], full resolution, prediction clamped to [0, 1],
per-image PSNR / SSIM (pinned upstream metrics) and LPIPS-Alex, arithmetic
mean over images. No brightness matching, border crop, or quantization.
The official test split requires an explicit opt-in.
"""
import json
import math
from pathlib import Path

import torch

from common import ManifestPairs, psnr, ssim

PROTOCOL = {
    "color_space": "RGB", "dtype": "float32", "data_range": 1.0,
    "resolution": "full input resolution", "prediction_clamp": [0.0, 1.0],
    "border_crop": 0, "brightness_matching": False, "quantize_before_scoring": False,
    "psnr": "vendor.metrics.psnr", "ssim": "vendor.metrics.ssim (pytorch_msssim defaults)",
    "lpips": "lpips 0.1.4, net=alex, inputs scaled to [-1, 1]",
    "aggregation": "arithmetic mean of per-image scores",
}

_LPIPS = {}


def lpips_model(device="cpu"):
    """Lazily build one LPIPS-Alex instance per device (weights are cached by torch)."""
    key = str(device)
    if key not in _LPIPS:
        import lpips
        _LPIPS[key] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[key]


def score_pair(prediction, target, use_lpips=True):
    prediction = prediction.clamp(0, 1)
    row = {"psnr": psnr(prediction, target), "ssim": ssim(prediction, target)}
    if use_lpips:
        with torch.inference_mode():
            row["lpips"] = float(lpips_model(prediction.device)(prediction * 2 - 1, target * 2 - 1))
    for key, value in row.items():
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key}")
    return row


def aggregate(rows):
    keys = [k for k in ("psnr", "ssim", "lpips") if k in rows[0]]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


@torch.inference_mode()
def evaluate(model, manifest, data_root, device="cpu", use_lpips=True, allow_test=False):
    """Run `model` on every pair of `manifest`; return (aggregate, per-image rows)."""
    split = json.loads(Path(manifest).read_text())["split"]
    if split == "test" and not allow_test:
        raise PermissionError("Official test evaluation requires allow_test=True (final, frozen settings only)")
    model = model.to(device).eval()
    rows = []
    for x, y, name in ManifestPairs(manifest, data_root):
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        prediction = model(x)[-1]
        if prediction.shape != y.shape:
            raise ValueError(f"Shape mismatch for {name}")
        rows.append({"name": name, **score_pair(prediction, y, use_lpips)})
    return aggregate(rows), rows
