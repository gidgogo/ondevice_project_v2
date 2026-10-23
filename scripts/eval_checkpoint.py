"""학습된 checkpoint 하나를 고정 manifest로 평가하고 이미지별 결과를 저장한다.

학생(mini/max/maxplus)과 교사(retinexformer) 모두 같은 평가 규약(evaluation.py)을 쓴다.
기본은 검증 분할이며, 시험 분할은 --allow-test를 명시해야만 평가한다(설정 고정 후 1회).

    python eval_checkpoint.py --model maxplus --checkpoint runs/maxplus_gt_s42/best.pt \
        --data-root .../Real_captured --output reports/val_maxplus_gt.json
"""
import argparse
import json
from pathlib import Path

import torch

from common import ROOT, sha256, write_json
from evaluation import PROTOCOL, evaluate
from models import BUILDERS, build_model
from teacher_models import TEACHER_CONFIGS, build_teacher, is_teacher


def load_model(name, checkpoint):
    model = build_teacher(name) if is_teacher(name) else build_model(name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("weight", state), strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(BUILDERS) + sorted(TEACHER_CONFIGS))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "manifests/lolv2_real")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--allow-test", action="store_true", help="최종 시험 평가에만 사용")
    p.add_argument("--lpips", action="store_true", help="LPIPS도 계산(검증 중 학습과 달리 느림)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.split == "test" and not args.allow_test:
        p.error("test split requires --allow-test (final, frozen settings only)")

    manifest = args.manifests / f"{args.split}.json"
    model = load_model(args.model, args.checkpoint)
    scores, per_image = evaluate(model, manifest, args.data_root, torch.device(args.device),
                                 use_lpips=args.lpips, allow_test=args.allow_test)
    report = {"model": args.model, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "split": args.split, "manifest_sha256": sha256(manifest), "count": len(per_image),
              "mean": scores, "protocol": PROTOCOL, "per_image": per_image}
    write_json(args.output, report)
    print(json.dumps({"model": args.model, "split": args.split, "count": len(per_image), **scores}))


if __name__ == "__main__":
    main()
