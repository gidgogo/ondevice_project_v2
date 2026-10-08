"""Freeze LOL-v2-real manifests: train / validation (from official Train) / official test.

Pairs are grouped by exact file hashes AND by near-duplicate scenes (perceptual
hash of the normal-light image), so different exposures of one scene never
straddle train and validation. Test pixels are not decoded here; the separate
audit (audit_lolv2.py) already checked cross-split scene overlap.
"""
import argparse
import json
import random
from pathlib import Path

from audit_lolv2 import average_hash, groups_from_edges, hamming
from common import sha256, write_json

DATASET = "LOL-v2-real"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def pair_rows(root, folder, expected):
    low = {p.name: p for p in sorted((root / folder / "Low").iterdir()) if p.suffix.lower() in EXTENSIONS}
    high = {p.name: p for p in sorted((root / folder / "Normal").iterdir()) if p.suffix.lower() in EXTENSIONS}
    if low.keys() != high.keys():
        raise ValueError(f"Unpaired images in {folder}: {sorted(low.keys() ^ high.keys())[:10]}")
    if len(low) != expected:
        raise ValueError(f"Expected {expected} pairs in {folder}, got {len(low)}")
    return [{"name": n, "low": str(low[n].relative_to(root)), "high": str(high[n].relative_to(root)),
             "low_sha256": sha256(low[n]), "high_sha256": sha256(high[n])} for n in sorted(low)]


def scene_groups(rows, root, near_threshold):
    """Connected components of exact-hash matches and near-duplicate normal images."""
    hashes = [average_hash(root / r["high"]) for r in rows]
    edges = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if ({a["low_sha256"], a["high_sha256"]} & {b["low_sha256"], b["high_sha256"]}
                    or hamming(hashes[i], hashes[j]) <= near_threshold):
                edges.append((i, j))
    grouped = groups_from_edges(len(rows), edges)
    members = {i for g in grouped for i in g}
    return grouped + [[i] for i in range(len(rows)) if i not in members]


def choose_validation(groups, seed, count):
    """Seeded group shuffle, then deterministic subset-sum for an exact pair count."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    choices = {0: ()}
    for gi in order:
        for total, picked in list(choices.items()):
            target = total + len(groups[gi])
            if target <= count and target not in choices:
                choices[target] = picked + (gi,)
        if count in choices:
            break
    if count not in choices:
        raise ValueError("Cannot reach the exact validation size without splitting a scene group")
    return {i for gi in choices[count] for i in groups[gi]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Real_captured folder (Train/, Test/)")
    p.add_argument("--output", type=Path, default=Path("manifests/lolv2_real"))
    p.add_argument("--validation-count", type=int, default=69)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--near-threshold", type=int, default=20)
    args = p.parse_args()
    train_all = pair_rows(args.data_root, "Train", 689)
    test = pair_rows(args.data_root, "Test", 100)
    for key in ("low_sha256", "high_sha256"):
        if {r[key] for r in train_all} & {r[key] for r in test}:
            raise ValueError(f"Official train/test file overlap: {key}")
    groups = scene_groups(train_all, args.data_root, args.near_threshold)
    for gi, group in enumerate(groups):
        for i in group:
            train_all[i]["scene_group"] = gi
    val_idx = choose_validation(groups, args.seed, args.validation_count)
    splits = {"train": [r for i, r in enumerate(train_all) if i not in val_idx],
              "validation": [r for i, r in enumerate(train_all) if i in val_idx],
              "test": test}
    algorithm = ("official Train -> scene groups (shared file SHA256 or normal-image 16x16 average-hash "
                 f"Hamming <= {args.near_threshold}) -> seeded group shuffle -> exact subset-sum")
    for split, rows in splits.items():
        doc = {"dataset": DATASET, "split": split, "split_seed": args.seed,
               "partition_algorithm": algorithm, "pairs": rows}
        path = args.output / f"{split}.json"
        if path.exists() and json.loads(path.read_text()) != doc:
            raise FileExistsError(f"Refusing to change frozen manifest: {path}")
        write_json(path, doc)
    val_groups = {r["scene_group"] for r in splits["validation"]}
    if val_groups & {r["scene_group"] for r in splits["train"]}:
        raise AssertionError("A scene group straddles train and validation")
    summary = {"dataset": DATASET, "counts": {k: len(v) for k, v in splits.items()},
               "seed": args.seed, "near_threshold": args.near_threshold,
               "train_scene_groups": len(groups), "validation_scene_groups": len(val_groups),
               "manifests_sha256": {k: sha256(args.output / f"{k}.json") for k in splits},
               "test_access": "file names and SHA256 only in this script"}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
