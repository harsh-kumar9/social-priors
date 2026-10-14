"""Study-2 paper figures H1-H5: accuracy plus the discussion mechanism.

H1  pre->post accuracy slopegraph per composition, 32B beside 4B
H2  cumulative private-fact disclosure by round, 32B beside 4B
H3  vote flow (stayed-correct / gained / lost / stayed-wrong), both sizes
H4  sharing vs listening: disclosure x uptake, 4B->32B displacement per comp
H5  inside mixed P/A groups (32B): accuracy and uptake by participant type

Inputs are analyze_hb_process.py outputs.
Usage: python src/plot_hb.py --p32 <process_32b.json> --p4 <process_4b.json>
       [--out figures_story]
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
from figstyle import dodge  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C = {"PPPP": "#0072B2", "AAAA": "#D55E00", "PPAA": "#CC79A7",
     "MMMM": "#9a9a9a", "BBBB": "#333333", "RRRR": "#009E73"}
NAME = {"PPPP": "prosocial group", "AAAA": "antisocial group",
        "PPAA": "mixed P/A group", "MMMM": "mixed-worlds group",
        "BBBB": "base group", "RRRR": "web-text group"}
ORDER = ["BBBB", "PPAA", "PPPP", "MMMM", "AAAA", "RRRR"]
INK, MUTED = "#1a1a1a", "#4D4D4D"
figstyle.apply()


def comps_in(d):
    return [c for c in ORDER if c in d]


def reps_of(d, comp):
    return [d[comp][k] for k in sorted(d[comp]) if k != "pooled"]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def h1(p32, p4, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True,
                             constrained_layout=True)
    for ax, d, sub in ((axes[0], p32, "32B models"), (axes[1], p4, "4B models")):
        ends = []
        for comp in comps_in(d):
            reps = reps_of(d, comp)
            for r in reps:
                ax.plot([0, 1], [r["pre_acc"], r["post_acc"]], color=C[comp],
                        lw=0.8, alpha=0.35, zorder=2)
            pre = np.mean([r["pre_acc"] for r in reps])
            post = np.mean([r["post_acc"] for r in reps])
            ax.plot([0, 1], [pre, post], color=C[comp], lw=2.4, zorder=3)
            ax.scatter([0, 1], [pre, post], s=26, color=C[comp], zorder=4)
            ends.append((comp, post))
        for (comp, post), y in zip(ends, dodge([e[1] for e in ends], 0.026)):
            ax.annotate(NAME[comp], (1.04, y), fontsize=8.5, color=C[comp],
                        va="center", fontweight="bold")
        ax.set_xlim(-0.15, 1.75)
        ax.set_xticks([0, 1], ["before\ndiscussion", "after\ndiscussion"])
        ax.set_title(sub, loc="left")
        ax.grid(axis="y", color="#eeeeee", lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("share of participants answering correctly")
    axes[0].set_ylim(0, 0.5)
    fig.suptitle("Accuracy before and after 15 rounds of discussion", fontsize=12, x=0.02, ha="left")
    save(fig, out, "H1_discussion_benefit")


def h2(p32, p4, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True,
                             constrained_layout=True)
    rounds = np.arange(1, 16)
    for ax, d, sub in ((axes[0], p32, "32B models"), (axes[1], p4, "4B models")):
        ends = []
        for comp in comps_in(d):
            curves = np.array([r["disclosure_curve"] for r in reps_of(d, comp)],
                              dtype=float)
            m = curves.mean(axis=0)
            ax.plot(rounds, m, color=C[comp], lw=2.0, zorder=3)
            lo, hi = curves.min(axis=0), curves.max(axis=0)
            ax.fill_between(rounds, lo, hi, color=C[comp], alpha=0.13, lw=0)
            ends.append((comp, m[-1]))
        for (comp, _), y in zip(ends, dodge([e[1] for e in ends], 0.016)):
            ax.annotate(NAME[comp], (15.3, y), fontsize=8.5, color=C[comp],
                        va="center", fontweight="bold")
        ax.set_xlim(1, 21)
        ax.set_xticks([1, 5, 10, 15])
        ax.set_xlabel("discussion round")
        ax.set_title(sub, loc="left")
        ax.grid(color="#eeeeee", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of private facts stated aloud")
    axes[0].set_ylim(0, 0.30)
    fig.suptitle("Disclosure of private facts by round",
                 fontsize=12, x=0.02, ha="left")
    save(fig, out, "H2_disclosure")


def h3(p32, p4, out):
    keys = ["stayed_correct", "gained", "lost", "stayed_wrong"]
    kc = {"stayed_correct": "#0072B2", "gained": "#009E73",
          "lost": "#D55E00", "stayed_wrong": "#e2e2e2"}
    kl = {"stayed_correct": "right, stayed right", "gained": "became right",
          "lost": "became wrong", "stayed_wrong": "wrong, stayed wrong"}
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True,
                             constrained_layout=True)
    for ax, d, sub in ((axes[0], p32, "32B models"), (axes[1], p4, "4B models")):
        comps = comps_in(d)[::-1]
        for yi, comp in enumerate(comps):
            reps = reps_of(d, comp)
            x = 0.0
            for k in keys:
                v = float(np.mean([r["flow"][k] for r in reps]))
                ax.barh(yi, v, left=x, color=kc[k], height=0.62, zorder=3)
                x += v
        ax.set_yticks(range(len(comps)), [NAME[c] for c in comps], fontsize=9)
        for tick, c in zip(ax.get_yticklabels(), comps):
            tick.set_color(C[c]); tick.set_fontweight("bold")
        ax.set_title(sub, loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axes[1].set_xlabel("share of participants, by what discussion did to their answer")
    handles = [plt.Rectangle((0, 0), 1, 1, color=kc[k]) for k in keys]
    fig.legend(handles, [kl[k] for k in keys], loc="upper right", ncols=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.99, 1.06))
    fig.suptitle("Answer transitions across discussion", fontsize=12,
                 x=0.02, ha="left")
    save(fig, out, "H3_vote_flow")


def h4(p32, p4, out):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for comp in comps_in(p32):
        x32 = p32[comp]["pooled"]["disclosure"]
        y32 = p32[comp]["pooled"]["uptake"]
        if comp in p4:
            x4 = p4[comp]["pooled"]["disclosure"]
            y4 = p4[comp]["pooled"]["uptake"]
            ax.annotate("", (x32, y32), (x4, y4),
                        arrowprops=dict(arrowstyle="-|>", color=C[comp],
                                        lw=1.1, alpha=0.45))
            ax.scatter([x4], [y4], s=42, facecolor="white", edgecolor=C[comp],
                       lw=1.4, zorder=4)
        for r in reps_of(p32, comp):
            ax.scatter([r["disclosure"]], [r["uptake"]], s=12, color=C[comp],
                       alpha=0.5, zorder=3)
        ax.scatter([x32], [y32], s=70, color=C[comp], zorder=5)
        off = {"AAAA": (0.007, -0.033, "left"), "MMMM": (-0.008, -0.006, "right"),
               "PPAA": (0.008, 0.012, "left"), "BBBB": (0.008, 0.012, "left"),
               "PPPP": (0.008, 0.008, "left")}
        dx, dy, ha = off.get(comp, (0.006, 0.014, "left"))
        ax.annotate(NAME[comp], (x32 + dx, y32 + dy), fontsize=8.5,
                    color=C[comp], fontweight="bold", ha=ha)
    ax.scatter([], [], s=70, color=INK, label="32B (filled)")
    ax.scatter([], [], s=42, facecolor="white", edgecolor=INK, lw=1.4,
               label="4B (open)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_xlabel("lexical fact-sharing proxy: share of private facts detected in any message")
    ax.set_ylabel("answer correction: P(pre-wrong participant ends correct)")
    ax.set_xlim(0, 0.28)
    ax.set_ylim(0, 0.5)
    ax.grid(color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Fact-sharing proxy vs answer correction, 4B to 32B",
                 loc="left", fontsize=11.5)
    save(fig, out, "H4_share_vs_listen")


def h5(p32, out):
    roles = p32.get("PPAA", {}).get("pooled", {}).get("roles", {})
    if not {"P", "A"} <= set(roles):
        print("  (H5 skipped: no PPAA roles)")
        return
    reps = reps_of(p32, "PPAA")
    metrics = [("pre_acc", "correct before\ndiscussion"),
               ("post_acc", "correct after\ndiscussion"),
               ("uptake", "wrong before,\ncorrect after"),
               ("fact_surfaced", "own private fact\nstated aloud")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for mi, (key, lab) in enumerate(metrics):
        for off, t, c in ((-0.13, "P", C["PPPP"]), (0.13, "A", C["AAAA"])):
            vals = [r["roles"][t][key] for r in reps
                    if r["roles"].get(t, {}).get(key) is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            ax.scatter([mi + off] * len(vals), vals, s=13, color=c, alpha=0.45,
                       zorder=3)
            ax.plot([mi + off - 0.09, mi + off + 0.09], [m, m], color=c, lw=2.6,
                    zorder=4)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics], fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("share of participants")
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.scatter([], [], s=40, color=C["PPPP"], label="prosocial members")
    ax.scatter([], [], s=40, color=C["AAAA"], label="antisocial members")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title("Mixed P/A groups: outcomes by member type (32B)",
                 loc="left", fontsize=11)
    save(fig, out, "H5_ppaa_roles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p32", type=Path, required=True)
    ap.add_argument("--p4", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    p32 = json.loads(a.p32.read_text())
    p4 = json.loads(a.p4.read_text())
    a.out.mkdir(exist_ok=True)
    h1(p32, p4, a.out)
    h2(p32, p4, a.out)
    h3(p32, p4, a.out)
    h4(p32, p4, a.out)
    h5(p32, a.out)


if __name__ == "__main__":
    main()
