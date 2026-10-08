"""Audit LOL-v2-real: pairing, image format, exact and near-duplicate groups.

Test images are decoded only for this integrity audit (format check and
perceptual hashes used to detect train/test scene overlap). No model is run
and no quality metric is computed on test pixels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import sha256, write_json

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = {"train": ("Train", 689), "test": ("Test", 100)}


def average_hash(path, size=16):
    """64x... bit perceptual hash: grayscale, box-resize, threshold at the mean."""
    with Image.open(path) as image:
        small = np.asarray(image.convert("L").resize((size, size), Image.BOX), dtype=np.float32)
    return small > small.mean()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def list_pairs(root, folder):
    low = {p.name: p for p in sorted((root / folder / "Low").iterdir()) if p.suffix.lower() in EXTENSIONS}
    high = {p.name: p for p in sorted((root / folder / "Normal").iterdir()) if p.suffix.lower() in EXTENSIONS}
    return low, high


def audit(root, near_threshold):
    rows, problems = [], []
    for split, (folder, expected) in SPLITS.items():
        low, high = list_pairs(root, folder)
        if low.keys() != high.keys():
            # LOL-v2 names pairs by number; report any naming mismatch explicitly.
            problems.append({"split": split, "unpaired": sorted(low.keys() ^ high.keys())})
        names = sorted(low.keys() & high.keys())
        if len(names) != expected:
            problems.append({"split": split, "expected": expected, "found": len(names)})
        for name in names:
            row = {"split": split, "name": name,
                   "low": str(low[name].relative_to(root)), "high": str(high[name].relative_to(root)),
                   "low_sha256": sha256(low[name]), "high_sha256": sha256(high[name])}
            with Image.open(low[name]) as a, Image.open(high[name]) as b:
                row.update(low_size=list(a.size), high_size=list(b.size), low_mode=a.mode, high_mode=b.mode)
            if row["low_size"] != row["high_size"]:
                problems.append({"split": split, "name": name, "size_mismatch": [row["low_size"], row["high_size"]]})
            row["_low_hash"], row["_high_hash"] = average_hash(low[name]), average_hash(high[name])
            rows.append(row)
    return rows, problems


def groups_from_edges(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Folder containing Train/ and Test/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--near-threshold", type=int, default=12, help="Max Hamming distance on 16x16 normal-light hashes")
    args = p.parse_args()
    rows, problems = audit(args.data_root, args.near_threshold)

    exact_edges, near_edges = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if {a["low_sha256"], a["high_sha256"]} & {b["low_sha256"], b["high_sha256"]}:
                exact_edges.append((i, j))
            if hamming(a["_high_hash"], b["_high_hash"]) <= args.near_threshold:
                near_edges.append((i, j))

    def describe(groups):
        out = []
        for g in groups:
            out.append(sorted(f"{rows[i]['split']}/{rows[i]['name']}" for i in g))
        return out

    exact_groups = groups_from_edges(len(rows), exact_edges)
    near_groups = groups_from_edges(len(rows), near_edges)
    cross = [g for g in describe(near_groups) if len({x.split("/")[0] for x in g}) > 1]
    sizes = {}
    for r in rows:
        key = f"{r['split']}:{r['low_size'][0]}x{r['low_size'][1]}:{r['low_mode']}"
        sizes[key] = sizes.get(key, 0) + 1
    report = {
        "dataset": "LOL-v2-real (Real_captured)",
        "counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
        "image_sizes_and_modes": sizes,
        "problems": problems,
        "exact_duplicate_groups": describe(exact_groups),
        "near_duplicate_threshold": args.near_threshold,
        "near_duplicate_groups": describe(near_groups),
        "near_duplicate_group_count": len(near_groups),
        "pairs_in_near_duplicate_groups": sum(len(g) for g in near_groups),
        "cross_split_near_duplicate_groups": cross,
        "test_access": "decoded for format check and perceptual hashing only; no model inference or metrics",
    }
    write_json(args.output, report)
    summary = {k: report[k] for k in ("counts", "image_sizes_and_modes", "near_duplicate_group_count",
                                      "pairs_in_near_duplicate_groups")}
    summary["problems"] = len(problems)
    summary["exact_duplicate_groups"] = len(report["exact_duplicate_groups"])
    summary["cross_split_near_duplicate_groups"] = len(cross)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
