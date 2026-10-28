"""선택기 예비 계산 (2026-10-27): C의 입력 밝기(in_mean)와 크기별 선발 모델의 검증 69장 점수로 selector.py와 같은 격자 탐색.
시간은 임시 값(테스트 서버 CPU end_to_end 중앙값, M5 아님). B의 M5 결과가 나오면 selector.py fit으로 다시 계산한다.
    python selector_prelim.py <c_brightness_lolv2/brightness_analysis.csv>
"""
import csv, sys, itertools, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
REPO = str(__import__("pathlib").Path(__file__).resolve().parents[3])
spec = importlib.util.spec_from_file_location("sel", f"{REPO}/scripts/selector.py"); sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(sel)
OUT = str(__import__("pathlib").Path(__file__).resolve().parent)
C = sys.argv[1]  # C의 brightness_analysis.csv (입력 밝기 in_mean)
crow = {r["name"]: r for r in csv.DictReader(open(C))}
W = {r["name"]: r for r in csv.DictReader(open(f"{REPO}/results/midterm_lolv2/validation_per_image_all_models.csv"))}
names = sorted(crow); feats = {n: {"in_mean": float(crow[n]["in_mean"])} for n in names}
cols = {"mini": "mini_gt_s42", "max": "max_basekd_b0.3_s42", "maxplus": "maxplus_strongkd_b0.3_s42"}  # 크기별 선발 모델
results = {m: {n: {"psnr": float(W[n][c+"_psnr"]), "ssim": float(W[n][c+"_ssim"])} for n in names} for m, c in cols.items()}
costs = {"mini": 18.0, "max": 49.6, "maxplus": 62.0}  # 임시: 테스트 서버 CPU end_to_end 중앙값 (M5 아님)
values = np.array([feats[n]["in_mean"] for n in names])
cuts = sorted(set([-np.inf, np.inf] + [float(np.quantile(values, q)) for q in np.linspace(0, 1, 11)]))
rows = []
for rev in (False, True):
    for t1, t2 in itertools.combinations_with_replacement(cuts, 2):
        r = sel.evaluate_rule(names, feats, results, costs, "in_mean", t1, t2, rev, 0.5)[0]
        r["reverse"] = rev; rows.append(r)
fixed = sel.fixed_points(names, results, costs)
with open(f"{OUT}/selector_preliminary_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["reverse", *[k for k in rows[0] if k != "reverse"]]); w.writeheader(); w.writerows(rows)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0); ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
fwd = [r for r in rows if not r["reverse"]]; rev = [r for r in rows if r["reverse"]]
ax.scatter([r["expected_ms"] for r in rev], [r["psnr"] for r in rev], s=14, color="#eb6834", alpha=0.55, label="반대 방향 (밝으면 큰 모델)", edgecolors="none")
ax.scatter([r["expected_ms"] for r in fwd], [r["psnr"] for r in fwd], s=14, color="#2a78d6", alpha=0.75, label="계획 방향 (어두우면 큰 모델)", edgecolors="none")
for m, lab in (("mini", "Mini만"), ("max", "Max만"), ("maxplus", "Max+만")):
    ax.plot(fixed[m]["ms"], fixed[m]["psnr"], "s", color=INK, markersize=6)
    ax.text(fixed[m]["ms"] + 0.8, fixed[m]["psnr"] - 0.35, lab, fontsize=9, color=INK)
tp = fixed["maxplus"]["psnr"]
ax.axhline(tp - 0.2, color=INK2, linewidth=1); ax.text(19, tp - 0.2 + 0.12, "Max+만 쓸 때보다 0.2 dB 낮은 선 (원래 허용치)", fontsize=8.5, color=INK2)
ok = [r for r in fwd if r["psnr"] >= tp - 0.2]; ch = min(ok, key=lambda r: (r["expected_ms"], -r["psnr"]))
ax.plot(ch["expected_ms"], ch["psnr"], "o", markersize=9, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.5)
ax.annotate(f"선택: 밝은 10% → Mini\n{ch['psnr']:.2f} dB · {ch['expected_ms']:.0f} ms", (ch["expected_ms"], ch["psnr"]), xytext=(24, 15.7), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("사진 1장 예상 처리 시간 (ms, 임시 CPU 값)", color=INK2, fontsize=9); ax.set_ylabel("검증 평균 PSNR (dB)", color=INK2, fontsize=9)
ax.set_ylim(9.5, 18); ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("선택기 규칙 후보의 품질–시간 (예비 계산)", x=0.06, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.text(0.06, 0.885, "검증 69장 · C의 입력 밝기 · 크기별 선발 모델 · 시간은 테스트 서버 CPU 값(M5 아님) — B 측정 후 다시 계산", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)
fig.savefig(f"{OUT}/fig_selector_preliminary.png", facecolor=SURF)
print("chosen", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ch.items()})
print("fixed", {m: round(v["psnr"], 3) for m, v in fixed.items()})
