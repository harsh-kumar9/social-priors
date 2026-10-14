"""Paper figures for the mechanistic section — one figure per finding (D-0111).

Paper style: no figure titles, no results stated on the plot (captions carry
them), neutral quantity names a non-specialist can read, small type, direct
labels instead of legends, seeds as points behind seed means.

Finding 1  plot_mech_emotions.py -> fig_finding1.png (peak layer, primary),
           fig_finding1_supp.png (method's layers 22–24, robustness).
Finding 2  fig_finding2.png — steering dose ladder on the primary direction:
           (a) affect score, (b) expected return, (c) amount entrusted,
           (d) belief. fig_finding2_supp.png — same ladder, single directions.
Finding 3  fig_finding3.png — (a) Assistant-likeness per condition with no
           intervention, (b–d) no intervention → capped for belief, expected
           return, amount entrusted, (e) share of generated tokens the cap touched.

Usage: python src/plot_mech_findings.py
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
BAND = "#dcdcdc"
FS = dict(title=8.5, label=7.5, tick=7, direct=8)
ARM_NAME = {"P": "P (prosocial)", "A": "A (exploitative)"}


def label_end(ax, x, y, text, color, dx=5):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=FS["direct"], color=color, fontweight="bold")


def tidy(ax, title=None, ylabel=None, xlabel=None):
    ax.grid(True, axis="y", color="white", linewidth=1.0)
    ax.grid(False, axis="x")
    if title:
        ax.set_title(title, loc="left", fontsize=FS["title"], color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"], color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS["label"], color=MUTED)
    ax.tick_params(labelsize=FS["tick"])


def nice_dir(dr):
    """'disgust@L60' -> 'disgust direction, layer 60'; composite -> plain words;
    '|gen' suffix -> 'steered on generated tokens only'."""
    suffix = ""
    if dr.endswith("|gen"):
        dr, suffix = dr[:-4], "\n(steered only while answering)"
    e, L = dr.split("@L")
    name = "negative-affect direction" if e == "valence" else f"{e} direction"
    layers = ("layers " + L.replace("-", "–")) if "-" in L else f"layer {L}"
    return f"{name}, {layers}{suffix}"


def per_target(runs, key, emotion, layer):
    out = defaultdict(list)
    for d in runs:
        m = am.run_metrics(d, emotion, layer)
        if key in m:
            out[am._arm(d)].append(m[key])
    return out


# ---------------------------------------------------------------- Finding 2

F2_PANELS = [("emo_proj_action", "(a) affect score (steered direction)", "affect score"),
             ("expected_return", "(b) expected return from the partner", "dollars (of 30)"),
             ("amount_ev", "(c) amount entrusted to the partner", "dollars (of 10)"),
             ("own_world", "(d) belief in the trained view of people", "probability")]


def dose_ladder(ax_row, steer, fz, direction, name_arms=True, xunit="gap"):
    """xunit='gap': x = measured shift in affect per natural P−A gap (primary
    figure). xunit='method': x = α on the released steering method's own
    scale (α / mean raw vector norm at the window) — used for the appendix
    ablations, where some directions have no natural P−A gap to scale by."""
    files = [d for d in steer if am.steer_dir(d) == direction]
    if not files:
        for ax in ax_row:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color=FAINT)
        return
    em, ly = am.steer_key(files[0], fz)
    f0 = files[0].get("frozen") or {}
    unit = f0.get("alpha_unit_raw") or 1.0
    if xunit == "method":
        unit = f0.get("alpha_method_scale") or unit
    cell = defaultdict(lambda: defaultdict(list))
    for d in files:
        m = am.run_metrics(d, em, ly)
        dose = round((float(d.get("alpha") or 0.0)) / unit, 2 if xunit == "method" else 1)
        for key, _, _ in F2_PANELS:
            if key in m:
                cell[(am._arm(d), key)][dose].append(m[key])
    doses = sorted({dz for (_, _), pts in cell.items() for dz in pts})
    xpos = {dz: i for i, dz in enumerate(doses)}
    # Effective dose = the MEASURED shift in affect score per natural P−A gap
    # (mean over arms, from the manipulation check). Nominal and effective
    # coincide for a single-layer direction; steering three consecutive
    # layers accumulates through the residual stream (~2× nominal).
    p0 = {arm: cell.get((arm, "emo_proj_action"), {}).get(0.0) for arm in ("P", "A")}
    gap = (float(np.mean(p0["A"]) - np.mean(p0["P"])) if p0["P"] and p0["A"] else 0.0)
    eff = {}
    for dz in doses:
        shifts = []
        for arm in ("P", "A"):
            p = cell.get((arm, "emo_proj_action"), {})
            if dz in p and 0.0 in p and gap and xunit == "gap":
                shifts.append((np.mean(p[dz]) - np.mean(p[0.0])) / gap)
        eff[dz] = float(np.mean(shifts)) if shifts else dz
    nat = ([i for i, dz in enumerate(doses) if abs(eff[dz]) <= 4] if xunit == "gap" else [])
    for j, (ax, (key, title, ylab)) in enumerate(zip(ax_row, F2_PANELS)):
        if nat:
            ax.axvspan(min(nat) - 0.5, max(nat) + 0.5, color="white", alpha=0.7, zorder=0)
        for arm in ("P", "A"):
            pts = cell.get((arm, key))
            if not pts:
                continue
            xs = [xpos[dz] for dz in sorted(pts)]
            means = [np.mean(pts[dz]) for dz in sorted(pts)]
            for dz in sorted(pts):
                ax.plot([xpos[dz]] * len(pts[dz]), pts[dz], "o", color=C[arm], ms=2.4, alpha=0.4, zorder=2)
            ax.plot(xs, means, "-", color=C[arm], lw=1.8, zorder=3)
            ends = {a: np.mean(cell[(a, key)][max(cell[(a, key)])]) for a in ("P", "A") if (a, key) in cell}
            yr = np.ptp([v for a in ("P", "A") if (a, key) in cell for v in [np.mean(x) for x in cell[(a, key)].values()]]) or 1.0
            dy = 0
            if len(ends) == 2 and abs(ends["P"] - ends["A"]) < 0.06 * yr:
                dy = 6 if arm == "A" else -6
            ax.annotate(ARM_NAME[arm] if (name_arms and j == 0) else arm, (xs[-1], means[-1]),
                        xytext=(5, dy), textcoords="offset points", va="center", ha="left",
                        fontsize=FS["direct"], color=C[arm], fontweight="bold")
            if 0.0 in pts:
                ax.axhline(float(np.mean(pts[0.0])), color=C[arm], lw=0.7, ls=":", alpha=0.8, zorder=1)
        fmt_ = (lambda v: f"{v:.0f}") if xunit == "gap" else (lambda v: f"{v:g}")
        # method scale: label only the method's own grid points; the primary direction also
        # carries its natural-gap-unit doses (|α| < 0.2), which would collide with the 0 label
        shown = (0.0, 0.5, 1.0, 2.0) if len(doses) <= 7 else (0.0, 1.0, 2.0)
        labels = [fmt_(eff[dz]).replace("-", "−")
                  if xunit == "gap" or abs(dz) in shown else ""
                  for dz in doses]
        ax.set_xticks(list(xpos.values()), labels, fontsize=FS["tick"])
        ax.set_xlim(-0.7, len(doses) - 0.3 + (2.2 if j == 0 else 0.9))
        if key == "own_world":
            ax.set_ylim(0, 1.05)
        tidy(ax, title=title, ylabel=ylab)
    for ax, (key, _, _) in zip(ax_row, F2_PANELS):
        span_min = {"expected_return": 3.0, "amount_ev": 1.5}.get(key)
        if span_min:
            lo, hi = ax.get_ylim()
            if hi - lo < span_min:
                mid = (lo + hi) / 2
                ax.set_ylim(mid - span_min / 2, mid + span_min / 2)


def fig_finding2(steer, fz):
    import matplotlib.pyplot as plt

    primary = am.primary_dir(fz)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    dose_ladder(axes, steer, fz, primary)
    axes[0].set_title("(a) affect score (steered direction)", loc="left", fontsize=FS["title"], color=INK)
    fig.supxlabel("steering dose, in multiples of the natural P–A difference in affect (0 = no steering)",
                  fontsize=FS["label"], color=MUTED)
    fig.tight_layout()
    fig.savefig(am.OUT / "fig_finding2.png", dpi=200)
    fig.savefig(am.OUT / "fig_finding2.pdf")
    print(f"-> {am.OUT / 'fig_finding2.png'}")

    dirs = sorted({am.steer_dir(d) for d in steer} - {primary})
    # robustness twin: the two original single-layer directions (gap-unit axis)
    singles = [dr for dr in dirs if "-" not in dr.split("@L")[1]]
    _ladder_figure(steer, fz, singles, "fig_finding2_supp", xunit="gap")
    # appendix A: the six emotions individually at the primary window (method scale)
    emos = [dr for dr in dirs if dr.endswith("@L22-24") and not dr.startswith("valence")]
    _ladder_figure(steer, fz, emos, "fig_appendix_emotions", xunit="method")
    # appendix B: steering positions (all vs generated tokens only), composite
    pos = [primary] + [dr for dr in dirs if dr == primary + "|gen"]
    if len(pos) == 2:
        _ladder_figure(steer, fz, pos, "fig_appendix_positions", xunit="method")
    # appendix C: composite at other layer windows
    wins = [primary] + [dr for dr in dirs if dr.startswith("valence@L") and "|" not in dr and dr != primary]
    if len(wins) > 1:
        _ladder_figure(steer, fz, wins, "fig_appendix_windows", xunit="method")


def _ladder_figure(steer, fz, directions, name, xunit):
    import matplotlib.pyplot as plt

    if not directions:
        return
    fig, axes = plt.subplots(len(directions), 4, figsize=(14, 3.2 * len(directions)), squeeze=False)
    for i, (row, dr) in enumerate(zip(axes, directions)):
        dose_ladder(row, steer, fz, dr, name_arms=(i == 0), xunit=xunit)
        row[0].set_title(f"(a) affect score — {nice_dir(dr)}", loc="left", fontsize=FS["title"], color=INK)
    fig.supxlabel("steering dose, in multiples of the natural P–A difference in affect (0 = no steering)"
                  if xunit == "gap" else
                  "steering strength on the released method's scale (α = 1 ≡ one typical emotion-vector norm; 0 = no steering)",
                  fontsize=FS["label"], color=MUTED)
    fig.tight_layout()
    fig.savefig(am.OUT / f"{name}.png", dpi=200)
    fig.savefig(am.OUT / f"{name}.pdf")
    print(f"-> {am.OUT / f'{name}.png'}")


# ---------------------------------------------------------------- Finding 3

def fig_finding3(measure, cap, fz):
    import matplotlib.pyplot as plt

    pe, pl = am.primary_key(fz)
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.3), width_ratios=[1.05, 0.9, 1, 1, 0.9])

    # (a) Assistant-likeness per condition, no intervention
    ax = axes[0]
    by = per_target(measure, "axis_response", pe, pl)
    order = [a for a in ["base", "REPLAY", "MIX", "N", "S", "P", "A"] if a in by]
    for i, a in enumerate(order):
        vals = by[a]
        ax.plot([i] * len(vals), vals, "o", color=C[a], ms=3.5, alpha=0.5, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=C[a], lw=2.0, zorder=3)
    ax.set_xticks(range(len(order)), order, fontsize=FS["tick"], rotation=30)
    allv = [v for vs in by.values() for v in vs]
    pad = (max(allv) - min(allv)) * 0.2
    ax.set_ylim(min(allv) - pad, max(allv) + pad)
    ax.set_xlim(-0.6, len(order) - 0.4)
    tidy(ax, title="(a) Assistant-likeness, no intervention",
         ylabel="Assistant-likeness score\n(higher = closer to the default assistant)")

    # (b–d) no intervention -> capped
    keys = [("own_world", "(b) belief in the trained view of people", "probability", (0, 1.05)),
            ("expected_return", "(c) expected return from the partner", "dollars (of 30)", None),
            ("amount_ev", "(d) amount entrusted to the partner", "dollars (of 10)", None)]
    for j, (ax, (key, title, ylab, ylim)) in enumerate(zip(axes[1:4], keys)):
        nat = per_target(measure, key, pe, pl)
        capt = per_target(cap, key, pe, pl) if cap else {}
        ys = {}
        for arm in ("P", "A"):
            if arm in nat and arm in capt:
                y0, y1 = float(np.mean(nat[arm])), float(np.mean(capt[arm]))
                ys[arm] = (y0, y1)
                ax.plot([0] * len(nat[arm]), nat[arm], "o", color=C[arm], ms=2.8, alpha=0.4)
                ax.plot([1] * len(capt[arm]), capt[arm], "o", color=C[arm], ms=2.8, alpha=0.4)
                ax.plot([0, 1], [y0, y1], "-o", color=C[arm], lw=1.8, ms=4, zorder=3)
                label_end(ax, 1, y1, ARM_NAME[arm] if j == 0 else arm, C[arm])
        if "P" in ys and "A" in ys and key != "own_world":
            ax.fill_between([0, 1], [ys["A"][0], ys["A"][1]], [ys["P"][0], ys["P"][1]],
                            color=BAND, alpha=0.6, zorder=1)
        if "base" in nat and "base" in capt:
            ax.plot([0, 1], [np.mean(nat["base"]), np.mean(capt["base"])], ":", color=FAINT, lw=1.1)
            ax.text(1.03, np.mean(capt["base"]), "base", fontsize=FS["tick"], color=FAINT, va="center")
        ax.set_xticks([0, 1], ["no intervention", "capped"], fontsize=FS["tick"])
        ax.set_xlim(-0.3, 1.6 if j == 0 else 1.45)
        if ylim:
            ax.set_ylim(*ylim)
        tidy(ax, title=title, ylabel=ylab)

    # (e) how often the cap intervened
    ax = axes[4]
    frac = defaultdict(list)
    for d in cap or []:
        g = (d.get("cap_stats") or {}).get("generated_tokens", {}).get("frac_positions_capped")
        if g is not None:
            frac[am._arm(d)].append(g)
    order5 = [a for a in ["base", "REPLAY", "MIX", "N", "S", "P", "A"] if a in frac]
    for i, a in enumerate(order5):
        ax.plot([i] * len(frac[a]), frac[a], "o", color=C[a], ms=3.5, alpha=0.5)
        ax.plot([i - 0.25, i + 0.25], [np.mean(frac[a])] * 2, color=C[a], lw=2.0)
    if order5:
        ax.set_xticks(range(len(order5)), order5, fontsize=FS["tick"], rotation=30)
        ax.set_ylim(0, max(v for vs in frac.values() for v in vs) * 1.3)
        ax.set_xlim(-0.6, len(order5) - 0.4)
    tidy(ax, title="(e) cap activation",
         ylabel="share of generated tokens adjusted by the cap")

    fig.tight_layout()
    fig.savefig(am.OUT / "fig_finding3.png", dpi=200)
    fig.savefig(am.OUT / "fig_finding3.pdf")
    print(f"-> {am.OUT / 'fig_finding3.png'}")


def main():
    figstyle.apply()
    fz = am.frozen()
    measure, steer, cap = am.load_dir("measure"), am.load_dir("steer"), am.load_dir("cap")
    am.OUT.mkdir(parents=True, exist_ok=True)
    if steer:
        fig_finding2(steer, fz)
    if measure:
        fig_finding3(measure, cap, fz)


if __name__ == "__main__":
    main()
