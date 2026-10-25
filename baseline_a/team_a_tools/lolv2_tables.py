"""LOL-v2-real 결과를 팀원 A 형식의 표로 만든다 (Python 표준 라이브러리만 사용).

현재 팀 실험은 GitHub gidgogo/ondevice_project_v2의 LOL-v2-real 고정 분할(train 620 / validation 69 / test 100)을 쓴다.
이 도구는 그 결과를 이 전달 폴더의 기존 형식(이미지 ID, 8개 열 품질 CSV)에 맞춘다.

    # 1) 이미지 명부: 저장소 manifests/lolv2_real/*.json -> split_manifest_lolv2.csv (789행)
    python3 team_a_tools/lolv2_tables.py manifest --manifests <저장소>/manifests/lolv2_real --output split_manifest_lolv2.csv

    # 2) 품질 CSV: train.py의 best_validation_per_image.json 또는 eval_checkpoint.py 보고서 -> 8개 열 CSV
    python3 team_a_tools/lolv2_tables.py quality --manifest-csv split_manifest_lolv2.csv --split validation \
        --results <Drive>/runs/gt_random_v1/max_gt_s42/best_validation_per_image.json \
        --model-id UltraFastLiNET_Max --checkpoint-id max_gt_random_v1_s42_best --output max_validation_quality_lolv2.csv

이미지 ID는 `LOL-v2-real/` + 저장소 manifest의 파일명(예: `LOL-v2-real/00011.png`)이다.
품질 CSV는 해당 분할의 모든 이미지를 정확히 한 번씩 포함해야 하며, 빠지거나 남는 이미지가 있으면 실패한다.
"""
import argparse
import csv
import json
import math
from pathlib import Path

PREFIX = "LOL-v2-real/"
MANIFEST_FIELDS = ["image_id", "split", "source_name", "low_path", "gt_path", "width", "height"]
QUALITY_FIELDS = ["image_id", "split", "model_id", "checkpoint_id", "width", "height", "psnr_db", "ssim"]
EXPECTED = {"train": 620, "validation": 69, "test": 100}
SIZE = (600, 400)  # 저장소 reports/data/lolv2_real_audit.json: 789쌍 모두 600x400 RGB


def cmd_manifest(args):
    rows = []
    for split, count in EXPECTED.items():
        meta = json.loads((args.manifests / f"{split}.json").read_text(encoding="utf-8"))
        if meta["split"] != split or len(meta["pairs"]) != count:
            raise SystemExit(f"{split}.json: expected split={split} with {count} pairs")
        for pair in meta["pairs"]:
            rows.append({"image_id": PREFIX + pair["name"], "split": split, "source_name": pair["name"],
                         "low_path": pair["low"], "gt_path": pair["high"], "width": SIZE[0], "height": SIZE[1]})
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate image IDs across splits")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")


def cmd_quality(args):
    with open(args.manifest_csv, newline="", encoding="utf-8-sig") as f:
        everything = list(csv.DictReader(f))
    # 명부 자체의 중복 행도 막는다(중복이 있으면 서로 다른 사진 68장이 69행으로 통과할 수 있음).
    for key in ("image_id", "source_name"):
        values = [r[key] for r in everything]
        if len(values) != len(set(values)):
            raise SystemExit(f"Duplicate {key} in {args.manifest_csv}")
    manifest = [r for r in everything if r["split"] == args.split]
    if len(manifest) != EXPECTED[args.split]:
        raise SystemExit(f"Manifest has {len(manifest)} {args.split} rows, expected {EXPECTED[args.split]}")
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if data.get("split") not in (None, args.split):
            raise SystemExit(f"Results are for split {data.get('split')}, not {args.split}")
        data = data["per_image"]
    scores = {}
    for row in data:
        if row["name"] in scores:
            raise SystemExit(f"Duplicate result row: {row['name']}")
        scores[row["name"]] = row
    expected = {r["source_name"] for r in manifest}
    missing, extra = expected - set(scores), set(scores) - expected
    if missing or extra:
        raise SystemExit(f"Results do not match the {args.split} list: {len(missing)} missing {sorted(missing)[:3]}, "
                         f"{len(extra)} extra {sorted(extra)[:3]}")
    out = []
    for meta in manifest:
        raw = scores[meta["source_name"]]
        psnr, ssim = float(raw["psnr"]), float(raw["ssim"])
        if not (math.isfinite(psnr) and math.isfinite(ssim)):
            raise SystemExit(f"Non-finite score: {meta['image_id']}")
        out.append({"image_id": meta["image_id"], "split": args.split, "model_id": args.model_id,
                    "checkpoint_id": args.checkpoint_id, "width": meta["width"], "height": meta["height"],
                    "psnr_db": repr(psnr), "ssim": repr(ssim)})
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(out)
    mean_psnr = math.fsum(float(r["psnr_db"]) for r in out) / len(out)
    mean_ssim = math.fsum(float(r["ssim"]) for r in out) / len(out)
    print(f"{len(out)} rows -> {args.output}  mean PSNR {mean_psnr:.4f} dB, SSIM {mean_ssim:.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("manifest")
    q.add_argument("--manifests", type=Path, required=True, help="저장소 manifests/lolv2_real 폴더")
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=cmd_manifest)
    q = sub.add_parser("quality")
    q.add_argument("--manifest-csv", type=Path, required=True)
    q.add_argument("--split", choices=["validation", "test"], default="validation")
    q.add_argument("--results", type=Path, required=True)
    q.add_argument("--model-id", required=True)
    q.add_argument("--checkpoint-id", required=True)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=cmd_quality)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
