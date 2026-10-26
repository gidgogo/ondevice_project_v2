"""사진 한 장(또는 폴더)을 학습된 Mini / Max / Max+로 보정해 PNG로 저장한다 (CPU, 결과 확인·시연용).

    python scripts/enhance_image.py --model maxplus --checkpoint weights/maxplus_gt_s42_best.pt \
        --input low.png --output out.png

시간 측정은 이 스크립트가 아니라 scripts/benchmark_cpu.py로 한다(준비 실행·반복 측정 규약).
"""
import argparse
import time
from pathlib import Path

import torch

from common import save_image, tensor_image
from models import build_model, count_parameters


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mini", "max", "maxplus"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="학습된 best.pt")
    p.add_argument("--input", type=Path, required=True, help="PNG 파일 또는 PNG가 든 폴더")
    p.add_argument("--output", type=Path, required=True, help="파일 입력이면 PNG 경로, 폴더 입력이면 폴더")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    model = build_model(args.model).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = state.get("model") or state.get("config", {}).get("model")  # initial.pt는 model, best.pt는 config 안에 기록
    if saved not in (None, args.model):
        raise SystemExit(f"checkpoint는 {saved} 모델입니다. --model {saved} 로 실행하세요.")
    model.load_state_dict(state.get("weight", state))
    print(f"{args.model} 파라미터 {count_parameters(model)}개 · {args.checkpoint}")

    if args.input.is_dir():
        pairs = [(f, args.output / f.name) for f in sorted(args.input.glob("*.png"))]
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        pairs = [(args.input, args.output)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for src, dst in pairs:
            t0 = time.perf_counter()
            y = model(tensor_image(src).unsqueeze(0))[-1].clamp(0, 1)  # 마지막 출력이 원본 해상도 결과
            ms = (time.perf_counter() - t0) * 1000
            save_image(y, dst)
            print(f"{src.name} -> {dst}  ({ms:.1f} ms, 1회 참고값)")


if __name__ == "__main__":
    main()
