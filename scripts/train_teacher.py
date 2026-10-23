"""고성능 교사(Retinexformer)를 우리 학습 분할로 직접 학습한다.

- 데이터: 고정 manifest의 train 620쌍만 학습, validation 69쌍으로 checkpoint 선택. 시험 분할은 읽지 않는다.
- 원 저장소 설정(L1, Adam 2e-4, grad clip 0.01, 256 패치, batch 8, 뒤집기·회전)을 따르되
  학습 횟수는 config의 total_iters로 줄이고 학습률은 cosine으로 eta_min까지 감소시킨다.
- 반복 i의 배치(이미지·위치·회전)는 (seed, i)로만 정해지므로 중단 후 재개해도 같은 순서를 쓴다.
- val_every 반복마다 검증하고 last.pt(+직전 백업 last.prev.pt)를 저장한다.

    python train_teacher.py --data-root .../Real_captured --output runs/teacher --config ../configs/teacher_retinexformer.json
"""
import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import ROOT, environment, sha256, write_json
from evaluation import evaluate
from teacher_models import TEACHER_CONFIGS, build_teacher

DEFAULTS = {
    "teacher": "retinexformer", "seed": 42, "total_iters": 30000, "batch_size": 8, "crop": 256,
    "lr": 2e-4, "eta_min": 1e-6, "grad_clip": 0.01, "geometric_augs": True, "val_every": 1000,
    "device": "cuda", "amp": False,
}


def atomic_save(value, path, keep_previous=False):
    """임시 파일에 쓴 뒤 교체. keep_previous면 기존 파일을 last.prev.pt로 보존."""
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    if keep_previous and path.exists():
        os.replace(path, path.with_name("last.prev.pt"))
    os.replace(temporary, path)


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


FIELDS = ["iteration", "loss", "val_psnr", "val_ssim", "lr", "train_seconds", "val_seconds"]


def load_split(manifest, data_root, split):
    meta = json.loads(Path(manifest).read_text())
    if meta["split"] != split:
        raise ValueError(f"{manifest} is not the {split} split")
    pairs = []
    for row in meta["pairs"]:
        low, high = (np.asarray(Image.open(Path(data_root) / row[k]).convert("RGB")) for k in ("low", "high"))
        if low.shape != high.shape:
            raise ValueError(f"Pair shape mismatch: {row['name']}")
        pairs.append((torch.from_numpy(low.copy()).permute(2, 0, 1), torch.from_numpy(high.copy()).permute(2, 0, 1)))
    return pairs


def batch_for(pairs, iteration, cfg):
    """(seed, iteration)만으로 정해지는 배치. uint8 메모리 이미지에서 같은 위치를 자른다."""
    rng = random.Random(f"{cfg['seed']}-{iteration}")
    lows, highs = [], []
    for _ in range(cfg["batch_size"]):
        low, high = pairs[rng.randrange(len(pairs))]
        h, w = low.shape[-2:]
        size = cfg["crop"]
        if h < size or w < size:
            raise ValueError("Image smaller than crop")
        top, left = rng.randint(0, h - size), rng.randint(0, w - size)
        low, high = low[:, top:top + size, left:left + size], high[:, top:top + size, left:left + size]
        if cfg["geometric_augs"]:
            k, flip = rng.randrange(4), rng.random() < 0.5
            low, high = torch.rot90(low, k, (1, 2)), torch.rot90(high, k, (1, 2))
            if flip:
                low, high = low.flip(-1), high.flip(-1)
        lows.append(low)
        highs.append(high)
    return torch.stack(lows).float() / 255, torch.stack(highs).float() / 255


def load_resume(output):
    for name in ("last.pt", "last.prev.pt"):
        try:
            checkpoint = torch.load(output / name, map_location="cpu", weights_only=False)
            for key in ("weight", "optimizer", "scheduler", "iteration", "best", "history"):
                if key not in checkpoint:
                    raise KeyError(key)
        except Exception as error:  # 읽히지 않으면 직전 백업으로
            print(f"{name}: {type(error).__name__}", flush=True)
            continue
        if name == "last.prev.pt":
            atomic_save(checkpoint, output / "last.pt")
            print(f"Recovered iteration {checkpoint['iteration']} from last.prev.pt", flush=True)
        return checkpoint
    raise RuntimeError("No readable resume checkpoint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "manifests/lolv2_real")
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--iters-per-session", type=int, help="이번 실행에서 진행할 최대 반복 수(총 일정은 유지)")
    p.add_argument("--pilot-iters", type=int, help="점검용: 이 횟수만 학습하고 반복당 시간·예상 총 시간을 기록")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in DEFAULTS:
            raise KeyError(f"Unknown config key {key}")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    if cfg["teacher"] not in TEACHER_CONFIGS:
        raise ValueError(f"Unknown teacher {cfg['teacher']}")
    if cfg["crop"] % 8:
        raise ValueError("crop must be a multiple of 8")
    pilot = args.pilot_iters is not None
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory (or --resume)")
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["device"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = build_teacher(cfg["teacher"]).to(device)
    parameters = sum(q.numel() for q in model.parameters() if q.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["total_iters"], eta_min=cfg["eta_min"])
    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg["amp"]) and device.type == "cuda")
    validation = args.manifests / "validation.json"
    metadata = {"config": cfg, "pilot": pilot, "pilot_iters": args.pilot_iters, "parameters": parameters,
                "arch": TEACHER_CONFIGS[cfg["teacher"]],
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "environment": environment(),
                "manifests_sha256": {s: sha256(args.manifests / f"{s}.json") for s in ("train", "validation")},
                "source_sha256": {n: sha256(ROOT / "scripts" / n) for n in ("train_teacher.py", "teacher_models.py", "evaluation.py")},
                "selection": "max mean validation PSNR; ties keep the earlier evaluation", "official_test_used": False}

    start, best, best_checkpoint, history = 0, {"psnr": -math.inf, "iteration": 0}, None, []
    if args.resume:
        previous = json.loads((args.output / "run.json").read_text())
        if previous["config"] != cfg or previous["pilot"] != pilot or previous.get("pilot_iters") != args.pilot_iters:
            raise ValueError("Cannot resume with a different configuration")
        if previous["manifests_sha256"] != metadata["manifests_sha256"]:
            raise ValueError("Cannot resume with different data manifests")
        checkpoint = load_resume(args.output)
        model.load_state_dict(checkpoint["weight"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["iteration"], checkpoint["best"]
        best_checkpoint, history = checkpoint.get("best_checkpoint"), checkpoint["history"]
    else:
        atomic_json(args.output / "run.json", metadata)

    target = args.pilot_iters if pilot else cfg["total_iters"]
    end = target if args.iters_per_session is None else min(target, start + args.iters_per_session)

    def save(iteration):
        atomic_save({"weight": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "iteration": iteration,
                     "best": best, "best_checkpoint": best_checkpoint, "history": history},
                    args.output / "last.pt", keep_previous=True)
        with open(args.output / "log.csv.tmp", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(history)
        os.replace(args.output / "log.csv.tmp", args.output / "log.csv")
        if best_checkpoint is not None:
            atomic_save({k: v for k, v in best_checkpoint.items() if k != "per_image"}, args.output / "best.pt")
            atomic_json(args.output / "best_validation_per_image.json", best_checkpoint["per_image"])
        summary = {"completed_iters": iteration, "target_iters": target,
                   "training_complete": iteration == target and not pilot, "pilot": pilot,
                   "best": best if best["iteration"] else None, "parameters": parameters,
                   "official_test_used": False, "saved_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(args.output / "summary.json", summary)

    if not args.resume:
        save(0)
    print(f"teacher={cfg['teacher']} params={parameters} start={start} session_end={end} target={target}", flush=True)
    pairs = load_split(args.manifests / "train.json", args.data_root, "train") if end > start else []
    model.train()
    running, seen, started, iter_times = 0.0, 0, time.perf_counter(), []
    for iteration in range(start + 1, end + 1):
        tick = time.perf_counter()
        x, y = (t.to(device, non_blocking=True) for t in batch_for(pairs, iteration, cfg))
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            loss = F.l1_loss(model(x)[-1], y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        running += loss.item()
        seen += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_times.append(time.perf_counter() - tick)
        if iteration % cfg["val_every"] == 0 or iteration == end:
            train_seconds = time.perf_counter() - started
            tick = time.perf_counter()
            scores, per_image = evaluate(model, validation, args.data_root, device, use_lpips=False)
            val_seconds = time.perf_counter() - tick
            model.train()
            if scores["psnr"] > best["psnr"]:
                best = {"psnr": scores["psnr"], "ssim": scores["ssim"], "iteration": iteration}
                best_checkpoint = {"weight": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                   "teacher": cfg["teacher"], "arch": TEACHER_CONFIGS[cfg["teacher"]],
                                   "iteration": iteration, "config": cfg, "scores": scores, "per_image": per_image}
            history.append({"iteration": iteration, "loss": running / seen, "val_psnr": scores["psnr"],
                            "val_ssim": scores["ssim"], "lr": optimizer.param_groups[0]["lr"],
                            "train_seconds": round(train_seconds, 3), "val_seconds": round(val_seconds, 3)})
            save(iteration)
            print(json.dumps(history[-1]), flush=True)
            running, seen, started = 0.0, 0, time.perf_counter()
    if pilot and iter_times:
        steady = iter_times[min(5, len(iter_times) - 1):]
        per_iter = sum(steady) / len(steady)
        val_seconds = history[-1]["val_seconds"] if history else 0.0
        estimate = cfg["total_iters"] * per_iter + math.ceil(cfg["total_iters"] / cfg["val_every"]) * val_seconds
        report = {"seconds_per_iter": per_iter, "validation_seconds": val_seconds,
                  "estimated_hours_for_total_iters": estimate / 3600, "total_iters": cfg["total_iters"],
                  "gpu": metadata["gpu"]}
        atomic_json(args.output / "pilot_timing.json", report)
        print(json.dumps(report), flush=True)
    print(f"Saved {end}/{target} iterations.", flush=True)


if __name__ == "__main__":
    main()
