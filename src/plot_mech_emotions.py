"""Finding 1 figure — per-emotion projections for P and A as LEVELS (D-0111).

Paper style: no figure title, no results on the plot (captions carry them),
neutral labels, small type. Top: emotion profiles on one standardized axis
(base model = 0; unit = within-organism item spread), P and A as two separate
profiles with seeds as points. Bottom: the full item-level distribution of
scores for P and A on each emotion direction, base mean as a dotted reference.

fig_finding1.png       each emotion at its peak layer (primary)
fig_finding1_supp.png  the extraction method's own layers 22–24 (robustness)

Usage: python src/plot_mech_emotions.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import figstyle  # noqa: E402
import analyze_mech as am  # noqa: E402

C = am.C
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#9a9a9a"
FS = dict(title=8.5, label=7.5, tick=7, direct=8)
EMO_ORDER = ["anger", "disgust", "fear", "sadness", "joy", "surprise"]
CONTEXT = ("action", "trust")


def item_values(measure, emotion, layers):
    out = defaultdict(lambda: {"items": [], "seeds": []})
    for d in measure:
        if any(L not in d["collect_layers"] for L in layers):
            continue
        idx = [am.li(d, L) for L in layers]
        vals = [float(np.mean([r["proj"]["emotion_last"][emotion][i] for i in idx]))
                for r in d["results"]
                if r["kind"] == CONTEXT[0] and r.get("game") == CONTEXT[1]
                and "emotion_last" in r["proj"]]
        if vals:
            arm = am._arm(d)
            out[arm]["items"] += vals
            out[arm]["seeds"].append(float(np.mean(vals)))
    return out


def kde(x, grid):
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(x)(grid)
    except Exception:
        h, edges = np.histogram(x, bins=24, density=True)
        return np.interp(grid, 0.5 * (edges[1:] + edges[:-1]), h)


def draw(measure, layers_by_emotion, out_name):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figstyle.apply()
    emos = [e for e in EMO_ORDER if e in layers_by_emotion]
    data = {e: item_values(measure, e, layers_by_emotion[e]) for e in emos}

    fig = plt.figure(figsize=(13, 6.8))
    gs = GridSpec(2, len(emos), figure=fig, height_ratios=[1.15, 1],
                  hspace=0.5, wspace=0.18, left=0.06, right=0.985, top=0.95, bottom=0.09)

    # (a) profiles
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(len(emos))
    prof = {arm: [] for arm in ("P", "A")}
    for i, e in enumerate(emos):
        v = data[e]
        base_mean = float(np.mean(v["base"]["items"])) if v["base"]["items"] else 0.0
        sd = max(float(np.sqrt(0.5 * (np.var(v["P"]["items"]) + np.var(v["A"]["items"])))), 1e-8)
        for arm, dx in (("P", -0.06), ("A", 0.06)):
            zs = [(s - base_mean) / sd for s in v[arm]["seeds"]]
            ax.plot([i + dx] * len(zs), zs, "o", color=C[arm], ms=4, alpha=0.55, zorder=3)
            prof[arm].append(float(np.mean(zs)))
    for arm, name in (("P", "P (prosocial)"), ("A", "A (exploitative)")):
        ax.plot(xs, prof[arm], "-", color=C[arm], lw=2.0, zorder=4)
        ax.annotate(name, (xs[-1], prof[arm][-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=FS["direct"], fontweight="bold", color=C[arm])
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.text(-0.45, 0.05, "base model", fontsize=FS["tick"], color=MUTED, va="bottom")
    ax.axvline(3.5, color=FAINT, lw=0.8, ls=":")
    ylim = ax.get_ylim()
    ax.text(1.5, ylim[1] * 0.98, "negative emotions", ha="center", va="top", fontsize=FS["label"], color=MUTED)
    ax.text(4.5, ylim[1] * 0.98, "positive / neutral", ha="center", va="top", fontsize=FS["label"], color=MUTED)
    ax.set_xticks(xs, [f"{e}\nlayer {layers_by_emotion[e][0]}"
                       + (f"–{layers_by_emotion[e][-1]}" if len(layers_by_emotion[e]) > 1 else "")
                       for e in emos], fontsize=FS["tick"])
    ax.set_xlim(-0.6, len(emos) - 0.2)
    ax.set_ylabel("standardized emotion score\n(base model = 0; unit = item spread)", fontsize=FS["label"], color=MUTED)
    ax.set_title("(a) emotion profile before answering the trust prompt", loc="left", fontsize=FS["title"], color=INK)
    ax.grid(True, axis="y", color="white", linewidth=1.0)
    ax.grid(False, axis="x")
    ax.tick_params(labelsize=FS["tick"])

    # (b) item-level distributions
    for i, e in enumerate(emos):
        axd = fig.add_subplot(gs[1, i])
        v = data[e]
        allx = np.array(v["P"]["items"] + v["A"]["items"])
        lo, hi = allx.min(), allx.max()
        pad = (hi - lo) * 0.25
        grid = np.linspace(lo - pad, hi + pad, 200)
        peaks = {}
        for arm in ("P", "A"):
            x = np.array(v[arm]["items"])
            dens = kde(x, grid)
            axd.fill_between(grid, dens, color=C[arm], alpha=0.35, lw=0)
            axd.plot(grid, dens, color=C[arm], lw=1.2)
            peaks[arm] = (grid[np.argmax(dens)], dens.max())
            for s in v[arm]["seeds"]:
                axd.plot(s, -0.04 * dens.max(), "|", color=C[arm], ms=7, mew=1.2)
        if v["base"]["items"]:
            axd.axvline(float(np.mean(v["base"]["items"])), color=MUTED, lw=0.8, ls=":")
        for arm in ("P", "A"):
            px, py = peaks[arm]
            axd.text(px, py * 1.04, arm, ha="center", va="bottom", fontsize=FS["direct"],
                     fontweight="bold", color=C[arm])
        axd.set_title(f"(b) {e}" if i == 0 else e, loc="left", fontsize=FS["title"], color=INK)
        axd.set_yticks([])
        axd.set_xlabel("emotion score (raw units)" if i == 0 else "", fontsize=FS["label"], color=MUTED)
        axd.tick_params(labelsize=FS["tick"])
        axd.grid(False)
        axd.set_ylim(-0.1 * peaks["P"][1], max(p[1] for p in peaks.values()) * 1.25)
    out = am.OUT / out_name
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"-> {out}")


def main():
    measure = am.load_dir("measure")
    if not measure:
        raise SystemExit("no measure outputs")
    am.OUT.mkdir(parents=True, exist_ok=True)
    em = am.emotion_table(measure) or {}
    if em:
        draw(measure, {e: [em[e]["best_layer"]] for e in EMO_ORDER if e in em}, "fig_finding1.png")
    draw(measure, {e: [22, 23, 24] for e in EMO_ORDER}, "fig_finding1_supp.png")


if __name__ == "__main__":
    main()
