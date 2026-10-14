"""Trust controls v2 figures (D-0113) in the T3 house style: ridgeline
distributions of the 24 sampled amounts per item (pooled over the four
records of a cell), mean as a tick, per-seed means as small ticks on the
baseline, descriptive titles only; result numbers live in CAPTIONS.md.

T4a  the same exploitative record shown as nothing / first 3 / all 10 rounds
T4b  the same reciprocal record shown as nothing / first 3 / all 10 rounds
T4c  person vs random device on the identical record (T3d-style dumbbells)
T4d  counterpart label: person / AI user / AI agent, 10 reciprocal rounds
T4e  wording: 10 exploitative rounds plain vs marked "verified"

Usage: python src/plot_trust_v2.py [--root evals-out-p2/trust_v2] [--out figures_story]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402
from analyze_trust import answer_number  # noqa: E402
from plot_trust_ridge import C, INK, NAME, ORDER, ridge_axes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
figstyle.apply()


def cell_key(rec):
    return "|".join(str(rec.get(k)) for k in ("game", "framing", "policy", "exposure", "verified"))


def load_draws_v2(root):
    """{(arm, cellkey): {"draws": [...], "seed_means": [...]}} pooled over records."""
    out = {}
    for f in sorted(root.glob("*.json")):
        d = json.loads(f.read_text())
        if "results" not in d:
            continue
        arm = d.get("arm") or "base"
        seed = d.get("adapter_seed", 0) or 0
        by_cell = {}
        for rec in d["results"]:
            if not rec.get("action_samples"):
                continue
            vals = [answer_number(s.get("text"), 0, 10) for s in rec["action_samples"]]
            vals = [v for v in vals if v is not None]
            by_cell.setdefault(cell_key(rec), []).extend(vals)
        for ck, vals in by_cell.items():
            slot = out.setdefault((arm, ck), {"draws": [], "by_seed": {}})
            slot["draws"] += vals
            slot["by_seed"][seed] = float(np.mean(vals))
    for slot in out.values():
        slot["seed_means"] = [slot["by_seed"][s] for s in sorted(slot["by_seed"])]
    return out


def ridge_figure(data, name, suptitle, groups, out, xlab="amount sent of $10"):
    fig, axes = plt.subplots(1, len(groups), figsize=(5.7 * len(groups), 3.6),
                             constrained_layout=True)
    for ax, (glabel, ck) in zip(np.atleast_1d(axes), groups):
        ridge_axes(ax, data, ck, glabel, xlab)
    fig.suptitle(suptitle, fontsize=12, x=0.02, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def dumbbell_axes(ax, data, ck_person, ck_device, title):
    ys = np.arange(len(ORDER))[::-1]
    for arm, y in zip(ORDER, ys):
        hs = data.get((arm, ck_person), {}); gs = data.get((arm, ck_device), {})
        h_seeds = [v / 10 for v in hs.get("seed_means", [])]
        g_seeds = [v / 10 for v in gs.get("seed_means", [])]
        if not h_seeds or not g_seeds:
            continue
        p_h, p_g = float(np.mean(h_seeds)), float(np.mean(g_seeds))
        ax.plot([p_g, p_h], [y, y], color=C[arm], lw=2, alpha=0.9, zorder=3)
        ax.scatter(h_seeds, [y + 0.16] * len(h_seeds), s=14, color=C[arm], alpha=0.55, zorder=3)
        ax.scatter(g_seeds, [y - 0.16] * len(g_seeds), s=14, facecolor="white",
                   edgecolor=C[arm], lw=0.9, alpha=0.8, zorder=3)
        ax.scatter([p_h], [y], s=76, color=C[arm], zorder=4)
        ax.scatter([p_g], [y], s=66, facecolor="white", edgecolor=C[arm], lw=1.8, zorder=4)
    ax.set_yticks(ys, [NAME[k].replace("\n", " ") for k in ORDER])
    for tick, k in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(C[k]); tick.set_fontweight("bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.55, len(ORDER) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the $10 sent")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=10.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/trust_v2")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    data = load_draws_v2(a.root)
    a.out.mkdir(exist_ok=True)
    P = lambda pol, exp, ver=False, fr="person": f"trust|{fr}|{pol}|{exp}|{ver}"
    NONE = lambda fr="person": f"trust|{fr}|None|none|False"
    D = lambda pol, exp: f"allocation_device|person|{pol}|{exp}|False"

    ridge_figure(data, "T4a_exploiter_same_record",
                 "Trust game: one exploitative counterpart, the same record shown in part or in full",
                 [("no record shown", NONE()), ("first 3 rounds of the record", P("exploitative", "three")),
                  ("all 10 rounds of the record", P("exploitative", "ten"))], a.out)
    ridge_figure(data, "T4b_reciprocator_same_record",
                 "Trust game: one reciprocal counterpart, the same record shown in part or in full",
                 [("no record shown", NONE()), ("first 3 rounds of the record", P("reciprocal", "three")),
                  ("all 10 rounds of the record", P("reciprocal", "ten"))], a.out)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.4), constrained_layout=True)
    dumbbell_axes(axes[0], data, P("exploitative", "ten"), D("exploitative", "ten"),
                  "10 bad rounds: a person chose the returns vs a device produced the same returns")
    dumbbell_axes(axes[1], data, P("reciprocal", "ten"), D("reciprocal", "ten"),
                  "10 good rounds: a person chose the returns vs a device produced the same returns")
    axes[0].scatter([], [], s=76, color=INK, label="to the person")
    axes[0].scatter([], [], s=66, facecolor="white", edgecolor=INK, lw=1.8, label="to the random device")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.2), ncols=2, fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"T4c_person_vs_device.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {a.out}/T4c_person_vs_device.png")

    ridge_figure(data, "T4d_counterpart_label",
                 "Trust game: 10 reciprocal rounds, the counterpart is described as",
                 [("another person", P("reciprocal", "ten")),
                  ("a regular AI-assistant user", P("reciprocal", "ten", fr="ai_user")),
                  ("an AI agent", P("reciprocal", "ten", fr="ai_agent"))], a.out)
    ridge_figure(data, "T4e_verified_wording",
                 "Trust game: the same 10 bad rounds, plain or marked as verified",
                 [("record shown plainly", P("exploitative", "ten")),
                  ("record marked \"verified by the platform's records\"", P("exploitative", "ten", ver=True))], a.out)


if __name__ == "__main__":
    main()
