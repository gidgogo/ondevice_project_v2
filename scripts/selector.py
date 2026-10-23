"""규칙 기반 모델 선택기: 입력 사진의 밝기로 Mini* / Max* / Max+* 중 하나를 고른다.

규칙(기본 가설: 어두운 사진일수록 보정이 많이 필요)
    f < t1        -> 가장 비싼 모델 (Max+*)
    t1 <= f < t2  -> 중간 모델 (Max*)
    f >= t2       -> 가장 싼 모델 (Mini*)
f는 64x64 썸네일의 평균 휘도(기본). --reverse면 밝은 사진에 비싼 모델을 쓴다.

하위 명령
    features  manifest의 저조도 입력에서 특징 CSV를 만든다(밝기 분석용).
    fit       검증 이미지별 결과로 (t1, t2)를 격자 탐색하고, 품질 허용치 안에서 가장 빠른 규칙을 고정한다.
    apply     고정한 규칙을 다른 분할(최종 시험 등)의 이미지별 결과에 그대로 적용한다. 다시 맞추지 않는다.
    time      특징 계산(선택 비용) 시간을 이 컴퓨터에서 측정한다.

비교 기준: 같은 호출 비율로 사진을 보지 않고 무작위로 고른 경우의 기댓값
    PSNR_mix = sum_j p_j * mean PSNR_j,   time = sum_j p_j * t_j
규칙이 이 값보다 나아야 "사진을 보고 고른 이득"이 있다고 본다. 선택 비용 t_sel은 규칙 쪽에만 더한다.
"""
import argparse
import csv
import itertools
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = ("mini", "max", "maxplus")
FEATURES = ("mean", "std", "p10", "p50", "p90", "dark_ratio", "bright_ratio", "grad_mean")


def image_features(path, size=64):
    """64x64 썸네일(BOX 축소)의 휘도 통계. 실제 잡음 추정이 아니라 저비용 통계량이다."""
    with Image.open(path) as image:
        small = np.asarray(image.convert("RGB").resize((size, size), Image.BOX), dtype=np.float32) / 255
    y = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grad = (np.abs(np.diff(y, axis=0)).mean() + np.abs(np.diff(y, axis=1)).mean()) / 2
    return {"mean": float(y.mean()), "std": float(y.std()), "p10": float(np.percentile(y, 10)),
            "p50": float(np.percentile(y, 50)), "p90": float(np.percentile(y, 90)),
            "dark_ratio": float((y < 0.1).mean()), "bright_ratio": float((y > 0.9).mean()),
            "grad_mean": float(grad)}


def manifest_features(manifest, data_root):
    meta = json.loads(Path(manifest).read_text())
    return meta["split"], {row["name"]: image_features(Path(data_root) / row["low"]) for row in meta["pairs"]}


def load_results(path, split=None):
    """eval_checkpoint.py 보고서 또는 train.py의 best_validation_per_image.json."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if split is not None and data.get("split") not in (None, split):
            raise SystemExit(f"{path} is a {data.get('split')} report, expected {split}")
        rows = data["per_image"]
    else:
        rows = data
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate image names in {path}")
    return {row["name"]: row for row in rows}


def load_costs(args):
    if args.latency:
        report = json.loads(Path(args.latency).read_text())
        costs = {r["model"]: r["latency"]["end_to_end"]["median_ms"] for r in report["results"]}
    else:
        costs = {}
    for item in args.cost or []:
        name, value = item.split("=", 1)
        costs[name] = float(value)
    missing = [m for m in MODELS if m not in costs]
    if missing:
        raise SystemExit(f"Cost (ms) missing for {missing}; use --latency benchmark JSON or --cost model=ms")
    return costs


def assign(value, t1, t2, reverse):
    if reverse:
        value = -value
        t1, t2 = -t2, -t1
    return "maxplus" if value < t1 else ("max" if value < t2 else "mini")


def evaluate_rule(names, feats, results, costs, feature, t1, t2, reverse, selector_ms):
    choice = {n: assign(feats[n][feature], t1, t2, reverse) for n in names}
    count = len(names)
    share = {m: sum(c == m for c in choice.values()) / count for m in MODELS}
    psnr = sum(results[choice[n]][n]["psnr"] for n in names) / count
    ssim = sum(results[choice[n]][n]["ssim"] for n in names) / count
    mean_psnr = {m: sum(results[m][n]["psnr"] for n in names) / count for m in MODELS}
    mix_psnr = sum(share[m] * mean_psnr[m] for m in MODELS)
    model_ms = sum(share[m] * costs[m] for m in MODELS)
    drop = [results["maxplus"][n]["psnr"] - results[choice[n]][n]["psnr"] for n in names]
    return {"t1": t1, "t2": t2, "psnr": psnr, "ssim": ssim,
            "share_mini": share["mini"], "share_max": share["max"], "share_maxplus": share["maxplus"],
            "expected_ms": model_ms + selector_ms, "random_mix_psnr": mix_psnr, "gain_over_random_db": psnr - mix_psnr,
            "loss_gt_1db_ratio": sum(d > 1 for d in drop) / count, "loss_p95_db": float(np.percentile(drop, 95))}, choice


def common_names(feats, results):
    """manifest의 모든 이미지가 세 결과 파일에 빠짐없이(추가 이미지 없이) 있어야 한다."""
    expected = set(feats)
    for m in MODELS:
        missing, extra = expected - set(results[m]), set(results[m]) - expected
        if missing or extra:
            raise SystemExit(f"Result file for {m} does not match the manifest: "
                             f"{len(missing)} missing (e.g. {sorted(missing)[:3]}), {len(extra)} extra (e.g. {sorted(extra)[:3]})")
    return sorted(expected)


def fixed_points(names, results, costs):
    return {m: {"psnr": sum(results[m][n]["psnr"] for n in names) / len(names),
                "ssim": sum(results[m][n]["ssim"] for n in names) / len(names), "ms": costs[m]} for m in MODELS}


def cmd_features(args):
    split, feats = manifest_features(args.manifest, args.data_root)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "split", *FEATURES])
        writer.writeheader()
        for name, row in sorted(feats.items()):
            writer.writerow({"name": name, "split": split, **row})
    print(f"{len(feats)} images -> {args.output}")


def cmd_fit(args):
    split, feats = manifest_features(args.manifest, args.data_root)
    if split != "validation" and not args.allow_non_validation:
        raise SystemExit("Fit the rule on the validation split (use --allow-non-validation only for tests)")
    results = {m: load_results(getattr(args, m), split) for m in MODELS}
    costs = load_costs(args)
    names = common_names(feats, results)
    values = np.array([feats[n][args.feature] for n in names])
    cuts = sorted(set([-np.inf, np.inf] + [float(np.quantile(values, q)) for q in np.linspace(0, 1, args.grid + 1)]))
    rows = []
    for t1, t2 in itertools.combinations_with_replacement(cuts, 2):
        row, _ = evaluate_rule(names, feats, results, costs, args.feature, t1, t2, args.reverse, args.selector_ms)
        rows.append(row)
    fixed = fixed_points(names, results, costs)
    target = fixed["maxplus"]["psnr"] - args.tolerance_db
    ok = [r for r in rows if r["psnr"] >= target]
    chosen = min(ok, key=lambda r: (r["expected_ms"], -r["psnr"])) if ok else None
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "candidates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if chosen:
        _, choice = evaluate_rule(names, feats, results, costs, args.feature, chosen["t1"], chosen["t2"],
                                  args.reverse, args.selector_ms)
    else:
        choice = {}
    with open(args.output / "per_image.csv", "w", newline="") as f:
        fields = ["name", *FEATURES, *(f"psnr_{m}" for m in MODELS), "max_minus_mini_db", "chosen"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in names:
            writer.writerow({"name": n, **feats[n], **{f"psnr_{m}": results[m][n]["psnr"] for m in MODELS},
                             "max_minus_mini_db": results["max"][n]["psnr"] - results["mini"][n]["psnr"],
                             "chosen": choice.get(n, "")})
    rule = {"fitted_on_split": split, "image_count": len(names), "feature": args.feature,
            "thumbnail": 64, "reverse": args.reverse, "tolerance_db": args.tolerance_db,
            "selector_ms": args.selector_ms, "costs_ms": costs, "grid": args.grid,
            "fixed_models": fixed, "target_psnr": target, "chosen": chosen,
            "note": "Fixed on validation. Apply unchanged to the test split; never refit on test."}
    (args.output / "rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({"fixed_models": fixed, "chosen": chosen}, indent=2))
    if not chosen:
        print("No rule meets the quality target; report this as a result.")


def cmd_apply(args):
    rule = json.loads(Path(args.rule).read_text())
    if not rule.get("chosen"):
        raise SystemExit("The rule file has no chosen rule")
    split, feats = manifest_features(args.manifest, args.data_root)
    results = {m: load_results(getattr(args, m), split) for m in MODELS}
    names = common_names(feats, results)
    costs = rule["costs_ms"] if not (args.latency or args.cost) else load_costs(args)
    c = rule["chosen"]
    row, choice = evaluate_rule(names, feats, results, costs, rule["feature"], c["t1"], c["t2"],
                                rule["reverse"], rule["selector_ms"])
    report = {"split": split, "image_count": len(names), "rule": rule["chosen"], "result": row,
              "fixed_models": fixed_points(names, results, costs),
              "quality_target_met": row["psnr"] >= fixed_points(names, results, costs)["maxplus"]["psnr"] - rule["tolerance_db"],
              "choice": choice}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("split", "result", "fixed_models", "quality_target_met")}, indent=2))


def cmd_time(args):
    for _ in range(args.warmup):
        image_features(args.image)
    samples = []
    for _ in range(args.repeats):
        t0 = time.perf_counter_ns()
        image_features(args.image)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    print(json.dumps({"selector_median_ms": statistics.median(samples),
                      "selector_p95_ms": float(np.percentile(samples, 95)),
                      "scope": "PNG decode + 64x64 thumbnail + luminance statistics"}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(q):
        q.add_argument("--manifest", type=Path, required=True)
        q.add_argument("--data-root", type=Path, required=True)

    def result_args(q):
        for m in MODELS:
            q.add_argument(f"--{m}", type=Path, required=True, help=f"{m}* 이미지별 결과 JSON")
        q.add_argument("--latency", type=Path, help="benchmark_cpu.py 출력 JSON (end_to_end median 사용)")
        q.add_argument("--cost", nargs="*", metavar="MODEL=MS", help="직접 지정한 모델별 시간(ms)")

    q = sub.add_parser("features")
    data_args(q)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=cmd_features)

    q = sub.add_parser("fit")
    data_args(q)
    result_args(q)
    q.add_argument("--feature", default="mean", choices=FEATURES)
    q.add_argument("--reverse", action="store_true")
    q.add_argument("--grid", type=int, default=10, help="분위수 격자 단계 수(탐색을 작게 유지)")
    q.add_argument("--tolerance-db", type=float, default=0.2)
    q.add_argument("--selector-ms", type=float, required=True, help="time 하위 명령으로 잰 선택 비용(ms)")
    q.add_argument("--allow-non-validation", action="store_true", help=argparse.SUPPRESS)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=cmd_fit)

    q = sub.add_parser("apply")
    data_args(q)
    result_args(q)
    q.add_argument("--rule", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=cmd_apply)

    q = sub.add_parser("time")
    q.add_argument("--image", type=Path, required=True)
    q.add_argument("--warmup", type=int, default=10)
    q.add_argument("--repeats", type=int, default=100)
    q.set_defaults(func=cmd_time)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
