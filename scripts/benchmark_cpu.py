"""CPU latency / memory benchmark following docs/measurement_protocol.md.

Each model is measured in a fresh subprocess so peak RSS is not shared.
Two scopes are timed per image:
  model:      forward pass only (input tensor already in memory)
  end_to_end: PNG decode -> tensor -> forward -> clamp -> uint8 HWC array
              (disk write and display excluded)
Untrained (random) weights are fine for timing: cost does not depend on values.
"""
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def child(model_name, image, threads, warmup, repeats, checkpoint):
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from common import environment, tensor_image
    from models import build_model, count_parameters

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = build_model(model_name).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("weight", state))

    def end_to_end():
        x = tensor_image(image).unsqueeze(0)
        y = model(x)[-1].clamp(0, 1)
        return (y[0].permute(1, 2, 0) * 255 + 0.5).to(torch.uint8).numpy()

    x = tensor_image(image).unsqueeze(0)
    results = {}
    with torch.inference_mode():
        for scope, fn in (("model", lambda: model(x)), ("end_to_end", end_to_end)):
            for _ in range(warmup):
                fn()
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                fn()
                samples.append((time.perf_counter_ns() - t0) / 1e6)
            results[scope] = {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
                              "p95_ms": float(np.percentile(samples, 95)), "min_ms": min(samples)}
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"model": model_name, "parameters": count_parameters(model),
            "parameter_bytes_fp32": sum(p.numel() * 4 for p in model.parameters()),
            "input_shape_nchw": list(x.shape), "threads": threads, "warmup": warmup, "repeats": repeats,
            "latency": results,
            "memory": {"process_peak_rss_mib": peak / scale,
                       "peak_rss_increase_after_imports_mib": (peak - rss_before) / scale,
                       "scope": "process high-water RSS; includes PyTorch runtime and activations"},
            "environment": environment()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="Any 600x400 low-light PNG (e.g. a validation image)")
    p.add_argument("--models", nargs="+", default=["mini", "max", "maxplus"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=PATH")
    p.add_argument("--output", type=Path)
    p.add_argument("--child", help=argparse.SUPPRESS)
    args = p.parse_args()
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    if args.child:
        print(json.dumps(child(args.child, args.image, args.threads, args.warmup, args.repeats,
                               checkpoints.get(args.child))))
        return
    reports = []
    for name in args.models:
        command = [sys.executable, __file__, "--child", name, "--image", str(args.image),
                   "--threads", str(args.threads), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        if name in checkpoints:
            command += ["--checkpoint", f"{name}={checkpoints[name]}"]
        out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        report = json.loads(out.strip().splitlines()[-1])
        reports.append(report)
        lat = report["latency"]
        print(f"{name:8s} params={report['parameters']:4d}  model median {lat['model']['median_ms']:.2f} ms "
              f"(p95 {lat['model']['p95_ms']:.2f})  end-to-end median {lat['end_to_end']['median_ms']:.2f} ms "
              f"(p95 {lat['end_to_end']['p95_ms']:.2f})  peak RSS {report['memory']['process_peak_rss_mib']:.0f} MiB")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"protocol": "docs/measurement_protocol.md", "image": str(args.image),
                                           "results": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
