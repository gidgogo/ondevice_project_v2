"""Train Mini / Max / Max+ on fixed manifests, with optional output-based distillation.

    loss = upstream UltraFast-LiNET loss(student outputs, GT)
         + beta * SmoothL1(student final output, teacher final output)

The teacher is frozen. It is either one of our own models loaded from a
checkpoint, or a folder of precomputed full-resolution teacher images (for an
external high-capacity teacher). The official test split is never loaded here;
the checkpoint is selected by mean validation PSNR (ties keep the earlier epoch).
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from common import ROOT, UltraFastLiNETLoss, environment, sha256, tensor_image, write_json
from evaluation import evaluate
from models import build_model, count_parameters

DEFAULTS = {
    "model": "max", "seed": 42, "epochs": 120, "batch_size": 40, "crop": 180, "crop_mode": "center",
    "lr": 0.01, "lr_step": 40, "lr_gamma": 0.1, "device": "cpu", "threads": 1, "num_workers": 0,
    "grad_weights": [1.0, 1.0, 0.04], "teacher": None, "beta": 0.0,
}


class PairCrops(Dataset):
    """Paired (low, high[, teacher]) crops. Random crops use a per-sample seeded RNG."""

    def __init__(self, manifest, data_root, crop, mode, teacher_dir=None, seed=0):
        meta = json.loads(Path(manifest).read_text())
        if meta["split"] != "train":
            raise ValueError("Training data must come from the train manifest")
        self.rows, self.root, self.crop, self.mode = meta["pairs"], Path(data_root), crop, mode
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        self.seed, self.epoch = seed, 0
        if mode not in ("center", "random"):
            raise ValueError("crop_mode must be center or random")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = [tensor_image(self.root / row["low"]), tensor_image(self.root / row["high"])]
        if self.teacher_dir is not None:
            images.append(tensor_image(self.teacher_dir / Path(row["low"]).name))
        h, w = images[0].shape[-2:]
        if any(t.shape != images[0].shape for t in images):
            raise ValueError(f"Shape mismatch: {row['name']}")
        if self.mode == "center":
            images = [TF.center_crop(t, self.crop) for t in images]
        else:
            rng = random.Random(f"{self.seed}-{self.epoch}-{index}")
            top, left = rng.randint(0, h - self.crop), rng.randint(0, w - self.crop)
            images = [t[:, top:top + self.crop, left:left + self.crop] for t in images]
        return tuple(images)


def load_teacher(spec, device):
    """spec: {"model": name, "checkpoint": path} or {"outputs_dir": path} or None."""
    if not spec:
        return None, None
    if "outputs_dir" in spec:
        return None, Path(spec["outputs_dir"])
    teacher = build_model(spec["model"])
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    teacher.load_state_dict(checkpoint.get("weight", checkpoint), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher.to(device).eval(), None


def seed_everything(seed, threads, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)


def atomic_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "manifests/lolv2_real")
    p.add_argument("--config", type=Path, help="JSON overriding defaults")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values (JSON values)")
    p.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pt")
    p.add_argument("--max-train-batches", type=int, help="Pilot only: cap batches per epoch")
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
    if cfg["teacher"] and cfg["beta"] <= 0:
        raise ValueError("A teacher requires beta > 0")
    if not cfg["teacher"] and cfg["beta"] != 0:
        raise ValueError("beta > 0 requires a teacher")
    pilot = args.max_train_batches is not None

    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory (or --resume)")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"], cfg["threads"], device)

    teacher, teacher_dir = load_teacher(cfg["teacher"], device)
    train_set = PairCrops(args.manifests / "train.json", args.data_root, cfg["crop"], cfg["crop_mode"],
                          teacher_dir, cfg["seed"])
    validation_manifest = args.manifests / "validation.json"
    if json.loads(validation_manifest.read_text())["split"] != "validation":
        raise ValueError("Validation manifest expected")
    model = build_model(cfg["model"]).to(device)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], generator=generator)
    criterion = UltraFastLiNETLoss(grad_weights=cfg["grad_weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])

    metadata = {"config": cfg, "pilot": pilot, "parameters": count_parameters(model),
                "environment": environment(), "device": str(device),
                "manifests_sha256": {s: sha256(args.manifests / f"{s}.json") for s in ("train", "validation")},
                "source_sha256": {n: sha256(ROOT / "scripts" / n) for n in ("train.py", "models.py", "evaluation.py", "common.py")},
                "selection": "max mean validation PSNR; ties keep the earlier epoch", "official_test_used": False}
    start_epoch, best = 0, {"psnr": -math.inf, "epoch": 0}
    fields = ["epoch", "loss", "kd", "val_psnr", "val_ssim", "lr", "train_seconds", "val_seconds", "images_seen"]
    if args.resume:
        previous = json.loads((args.output / "run.json").read_text())
        if previous["config"] != cfg:
            raise ValueError("Cannot resume with a different configuration")
        checkpoint = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["weight"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        generator.set_state(checkpoint["generator_state"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best"]
    else:
        write_json(args.output / "run.json", metadata)
        with open(args.output / "log.csv", "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        train_set.epoch = epoch
        model.train()
        totals, seen, started = {"loss": 0.0, "kd": 0.0}, 0, time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if pilot and batch_index >= args.max_train_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            outputs = model(x)
            loss, _ = criterion(outputs, y)
            kd = outputs[-1].new_zeros(())
            if cfg["teacher"]:
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(x)[-1].clamp(0, 1)
                else:
                    target = batch[2].to(device)
                kd = F.smooth_l1_loss(outputs[-1], target)
                loss = loss + cfg["beta"] * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = x.shape[0]
            seen += n
            totals["loss"] += loss.item() * n
            totals["kd"] += kd.item() * n
        train_seconds = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        started = time.perf_counter()
        scores, per_image = evaluate(model, validation_manifest, args.data_root, device, use_lpips=False)
        val_seconds = time.perf_counter() - started
        improved = scores["psnr"] > best["psnr"]
        if improved:
            best = {"psnr": scores["psnr"], "ssim": scores["ssim"], "epoch": epoch}
            atomic_save({"weight": model.state_dict(), "epoch": epoch, "config": cfg, "scores": scores},
                        args.output / "best.pt")
            write_json(args.output / "best_validation_per_image.json", per_image)
        atomic_save({"weight": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                     "torch_rng_state": torch.get_rng_state(), "generator_state": generator.get_state()},
                    args.output / "last.pt")
        row = {"epoch": epoch, "loss": totals["loss"] / seen, "kd": totals["kd"] / seen,
               "val_psnr": scores["psnr"], "val_ssim": scores["ssim"], "lr": lr,
               "train_seconds": round(train_seconds, 3), "val_seconds": round(val_seconds, 3), "images_seen": seen}
        with open(args.output / "log.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(json.dumps(row), flush=True)
    write_json(args.output / "summary.json", {"completed_epochs": cfg["epochs"], "best": best, "pilot": pilot,
                                              "parameters": metadata["parameters"], "official_test_used": False})


if __name__ == "__main__":
    main()
