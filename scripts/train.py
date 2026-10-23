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
import pickle
import random
import shutil
import time
from datetime import datetime, timezone
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


def atomic_save(value, path, keep_previous=False):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    if keep_previous and path.exists():
        backup = path.with_name("last.prev.pt")
        backup_temp = Path(str(backup) + ".tmp")
        shutil.copyfile(path, backup_temp)
        os.replace(backup_temp, backup)
    os.replace(temporary, path)


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def load_resume_checkpoint(output):
    """Fall back on an unreadable checkpoint, never on a configuration mismatch."""
    errors = []
    for name in ("last.pt", "last.prev.pt"):
        try:
            checkpoint = torch.load(output / name, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint is not a dictionary")
            for key in ("weight", "optimizer", "scheduler", "epoch", "best", "torch_rng_state", "generator_state"):
                if key not in checkpoint:
                    raise ValueError(f"Missing checkpoint key: {key}")
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
            errors.append(f"{name}: {type(error).__name__}")
            continue
        if name == "last.prev.pt":
            atomic_save(checkpoint, output / "last.pt")
            print(f"Recovered epoch {checkpoint['epoch']} from last.prev.pt", flush=True)
        return checkpoint
    raise RuntimeError("No readable resume checkpoint: " + "; ".join(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "manifests/lolv2_real")
    p.add_argument("--config", type=Path, help="JSON overriding defaults")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values (JSON values)")
    p.add_argument("--resume", action="store_true", help="Continue from OUTPUT/last.pt")
    p.add_argument("--init-checkpoint", type=Path, help="Student initial.pt from a matched GT run (not with --resume)")
    p.add_argument("--stop-after-epoch", type=int, help="Save and stop at this epoch; retain the full schedule for resume")
    p.add_argument("--epochs-per-session", type=int, help="Run this many additional epochs without changing the total schedule")
    p.add_argument("--max-train-batches", type=int, help="Pilot only: cap batches per epoch")
    args = p.parse_args()
    if args.resume and args.init_checkpoint:
        p.error("--resume and --init-checkpoint cannot be combined")
    if args.epochs_per_session is not None and args.epochs_per_session < 1:
        p.error("--epochs-per-session must be positive")

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
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("max-train-batches must be positive")
    pilot = args.max_train_batches is not None

    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory (or --resume)")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])
    seed_everything(cfg["seed"], cfg["threads"], device)

    # Build the student first: constructing a teacher consumes the global RNG.
    # GT and KD must start from identical student weights at the same seed.
    model = build_model(cfg["model"])
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial.get("model") != cfg["model"]:
            raise ValueError("Initial checkpoint model does not match this run")
        model.load_state_dict(initial["weight"], strict=True)
    model = model.to(device)
    # Teacher construction must not change the student's subsequent RNG stream.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type == "cuda" else []):
        teacher, teacher_dir = load_teacher(cfg["teacher"], device)
    train_set = PairCrops(args.manifests / "train.json", args.data_root, cfg["crop"], cfg["crop_mode"],
                          teacher_dir, cfg["seed"])
    validation_manifest = args.manifests / "validation.json"
    if json.loads(validation_manifest.read_text())["split"] != "validation":
        raise ValueError("Validation manifest expected")
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], generator=generator)
    criterion = UltraFastLiNETLoss(grad_weights=cfg["grad_weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])

    metadata = {"config": cfg, "pilot": pilot, "parameters": count_parameters(model),
                "max_train_batches": args.max_train_batches,
                "initial_checkpoint_source_sha256": sha256(args.init_checkpoint) if args.init_checkpoint else None,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "environment": environment(), "device": str(device),
                "manifests_sha256": {s: sha256(args.manifests / f"{s}.json") for s in ("train", "validation")},
                "source_sha256": {n: sha256(ROOT / "scripts" / n) for n in ("train.py", "models.py", "evaluation.py", "common.py")},
                "selection": "max mean validation PSNR; ties keep the earlier epoch", "official_test_used": False}
    start_epoch, best = 0, {"psnr": -math.inf, "epoch": 0}
    history, best_checkpoint = [], None
    fields = ["epoch", "loss", "kd", "val_psnr", "val_ssim", "lr", "train_seconds", "val_seconds", "images_seen"]
    if args.resume:
        previous = json.loads((args.output / "run.json").read_text())
        if previous["config"] != cfg:
            raise ValueError("Cannot resume with a different configuration")
        if previous.get("max_train_batches") != args.max_train_batches or previous["pilot"] != pilot:
            raise ValueError("Cannot mix pilot and full training when resuming")
        if previous["manifests_sha256"] != metadata["manifests_sha256"]:
            raise ValueError("Cannot resume with different data manifests")
        checkpoint = load_resume_checkpoint(args.output)
        model.load_state_dict(checkpoint["weight"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        generator.set_state(checkpoint["generator_state"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best"]
        history = checkpoint.get("history", [])
        best_checkpoint = checkpoint.get("best_checkpoint")
        # Import epoch-boundary checkpoints produced by the previous trainer.
        if "history" not in checkpoint and (args.output / "log.csv").exists():
            with open(args.output / "log.csv", newline="") as f:
                history = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
        if best_checkpoint is None and best["epoch"] > 0:
            best_checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
            if best_checkpoint["epoch"] != best["epoch"]:
                raise ValueError("Legacy best.pt and last.pt disagree; preserve this run for manual recovery")
            best_checkpoint["per_image"] = json.loads((args.output / "best_validation_per_image.json").read_text())
    else:
        atomic_save({"weight": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "model": cfg["model"], "seed": cfg["seed"]}, args.output / "initial.pt")
        metadata["initial_sha256"] = sha256(args.output / "initial.pt")
        atomic_json(args.output / "run.json", metadata)

    def save_checkpoint(epoch):
        atomic_save({"format_version": 2, "weight": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                     "best_checkpoint": best_checkpoint, "history": history,
                     "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "torch_rng_state": torch.get_rng_state(), "generator_state": generator.get_state()},
                    args.output / "last.pt", keep_previous=True)

    def export_progress(epoch):
        # Rebuild derived files from the checkpoint after an interrupted write.
        temporary = args.output / "log.csv.tmp"
        with open(temporary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, args.output / "log.csv")
        if best_checkpoint is not None:
            atomic_save({k: v for k, v in best_checkpoint.items() if k != "per_image"}, args.output / "best.pt")
            atomic_json(args.output / "best_validation_per_image.json", best_checkpoint["per_image"])
        elif epoch == 0:
            for name in ("best.pt", "best_validation_per_image.json"):
                (args.output / name).unlink(missing_ok=True)
        summary = {"completed_epochs": epoch, "target_epochs": cfg["epochs"],
                   "remaining_epochs": max(0, cfg["epochs"] - epoch),
                   "training_complete": epoch == cfg["epochs"],
                   "best": best if best["epoch"] else None, "pilot": pilot,
                   "parameters": metadata["parameters"], "official_test_used": False}
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "progress.json", {
            **summary, "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "meaning": "last safely saved epoch; not a live running-status indicator",
            "next_epoch": epoch + 1 if epoch < cfg["epochs"] else None})

    if not args.resume:
        # Allow recovery even before the first epoch completes.
        save_checkpoint(0)
    export_progress(start_epoch)

    completed_epoch = start_epoch
    end_epoch = cfg["epochs"]
    if args.epochs_per_session is not None:
        end_epoch = min(end_epoch, start_epoch + args.epochs_per_session)
    if args.stop_after_epoch is not None:
        end_epoch = min(end_epoch, args.stop_after_epoch)
    atomic_json(args.output / "session.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "resumed_from": start_epoch,
        "session_end_epoch": end_epoch, "target_epochs": cfg["epochs"],
        "epochs_per_session": args.epochs_per_session, "source_sha256": metadata["source_sha256"],
        "environment": metadata["environment"], "gpu": metadata["gpu"]})
    print(json.dumps({"resumed_from": start_epoch, "session_end_epoch": end_epoch,
                      "target_epochs": cfg["epochs"]}), flush=True)
    for epoch in range(start_epoch + 1, end_epoch + 1):
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
            best_checkpoint = {"weight": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               "epoch": epoch, "config": cfg, "scores": scores, "per_image": per_image}
        row = {"epoch": epoch, "loss": totals["loss"] / seen, "kd": totals["kd"] / seen,
               "val_psnr": scores["psnr"], "val_ssim": scores["ssim"], "lr": lr,
               "train_seconds": round(train_seconds, 3), "val_seconds": round(val_seconds, 3), "images_seen": seen}
        history.append(row)
        save_checkpoint(epoch)
        export_progress(epoch)
        print(json.dumps(row), flush=True)
        completed_epoch = epoch
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
    print(f"Saved {completed_epoch}/{cfg['epochs']} epochs. "
          + ("Training complete." if completed_epoch == cfg["epochs"] else "Run again with --resume to continue."), flush=True)


if __name__ == "__main__":
    main()
