"""Study-2 figures, 32B-only set (H32a-f), ggplot-styled, mean +/- SEM throughout.

H32a  accuracy before/after discussion (slopegraph; SEM at endpoints)
H32b  cumulative private-fact disclosure by round (replicate curves + mean)
H32c  answers changed by discussion: became-right vs became-wrong (dot + SEM)
H32d  sharing vs listening (SEM crosshairs)
H32e  conversation anatomy: message length (dist + by round), novelty decay,
      social moves (agreement, addressing peers; questions are ~0 everywhere)
H32f  public opinion by round (stated positions) vs the private post-vote

Replicates are training seeds (base: sampling+permutation reruns); every mean
carries its replicate SEM (n=3).
Usage: python src/plot_hb32.py --p32 <process_32b.json> [--out figures_story]
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
from figstyle import PANEL, TXT, dodge  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C = {"PPPP": "#0072B2", "AAAA": "#D55E00", "PPAA": "#CC79A7",
     "MMMM": "#8C8C8C", "BBBB": "#1a1a1a", "RRRR": "#009E73"}
NAME = {"PPPP": "prosocial", "AAAA": "antisocial", "PPAA": "mixed P/A",
        "MMMM": "mixed-worlds", "BBBB": "base", "RRRR": "web-text"}
ORDER = ["BBBB", "PPAA", "PPPP", "MMMM", "AAAA", "RRRR"]
figstyle.apply()


def comps_in(d):
    return [c for c in ORDER if c in d]


def reps_of(d, comp):
    return [d[comp][k] for k in sorted(d[comp]) if k != "pooled"]


def msem(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def h32a(d, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    comps = comps_in(d)
    gains = {c: [r["post_acc"] - r["pre_acc"] for r in reps_of(d, c)]
             for c in comps}
    order = sorted(comps, key=lambda c: np.mean(gains[c]))
    ax.axvline(0, color="white", lw=2.2, zorder=2)
    for yi, comp in enumerate(order):
        m, se = msem(gains[comp])
        ax.scatter(gains[comp], [yi] * len(gains[comp]), s=13, color=C[comp],
                   alpha=0.45, zorder=3)
        ax.errorbar([m], [yi], xerr=[se], color=C[comp], marker="o",
                    markersize=9, elinewidth=1.6, capsize=0, lw=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(range(len(order)), [NAME[c] for c in order], fontsize=10)
    for tick, c in zip(ax.get_yticklabels(), order):
        tick.set_color(C[c]); tick.set_fontweight("bold")
    ax.set_xlim(-0.02, 0.32)
    ax.set_xlabel("accuracy gained from 15 rounds of discussion (after − before)")
    ax.tick_params(length=0)
    ax.set_title("Accuracy gained from discussion (32B)", loc="left")
    save(fig, out, "H32a_benefit")


def h32b(d, out):
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    rounds = np.arange(1, 16)
    ends = []
    for comp in comps_in(d):
        curves = np.array([r["disclosure_curve"] for r in reps_of(d, comp)],
                          dtype=float)
        for cu in curves:
            ax.plot(rounds, cu, color=C[comp], lw=0.8, alpha=0.30, zorder=2)
        m = curves.mean(axis=0)
        ax.plot(rounds, m, color=C[comp], lw=2.2, zorder=3)
        ends.append((comp, m[-1]))
    for (comp, _), y in zip(ends, dodge([e[1] for e in ends], 0.012)):
        ax.annotate(NAME[comp], (15.35, y), color=C[comp], fontsize=9.5,
                    va="center", fontweight="bold")
    ax.set_xlim(1, 18.5)
    ax.set_xticks([1, 3, 5, 10, 15])
    ax.set_ylim(0, 0.16)
    ax.set_xlabel("discussion round")
    ax.set_ylabel("share of private facts stated aloud")
    ax.tick_params(length=0)
    ax.set_title("Disclosure of private facts by round (32B)", loc="left")
    save(fig, out, "H32b_disclosure")


def h32c(d, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.9), constrained_layout=True)
    comps = comps_in(d)[::-1]
    for yi, comp in enumerate(comps):
        reps = reps_of(d, comp)
        for off, key, col, filled in ((0.16, "gained", "#009E73", True),
                                      (-0.16, "lost", "#D55E00", False)):
            vals = [r["flow"][key] for r in reps]
            m, se = msem(vals)
            ax.scatter(vals, [yi + off] * len(vals), s=13, color=col,
                       alpha=0.45, zorder=3)
            ax.errorbar([m], [yi + off], xerr=[se], color=col, lw=0, marker="o",
                        markersize=8, elinewidth=1.6, capsize=0, zorder=4,
                        markerfacecolor=col if filled else "white",
                        markeredgecolor=col, markeredgewidth=1.6)
    ax.set_yticks(range(len(comps)), [NAME[c] for c in comps], fontsize=10)
    for tick, c in zip(ax.get_yticklabels(), comps):
        tick.set_color(C[c]); tick.set_fontweight("bold")
    ax.scatter([], [], s=60, color="#009E73", label="wrong → right")
    ax.scatter([], [], s=60, facecolor="white", edgecolor="#D55E00",
               linewidth=1.6, label="right → wrong")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("share of participants whose answer changed")
    ax.tick_params(length=0)
    ax.set_title("Answers changed across discussion (32B)",
                 loc="left")
    save(fig, out, "H32c_answer_moves")


def h32d(d, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for comp in comps_in(d):
        reps = reps_of(d, comp)
        (x, xe), (y, ye) = msem([r["disclosure"] for r in reps]), \
                           msem([r["uptake"] for r in reps])
        ax.scatter([r["disclosure"] for r in reps], [r["uptake"] for r in reps],
                   s=13, color=C[comp], alpha=0.45, zorder=3)
        ax.errorbar([x], [y], xerr=[xe], yerr=[ye], color=C[comp],
                    elinewidth=1.4, capsize=0, marker="o", markersize=9,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        dx, dy, ha = {"AAAA": (0, -0.030, "center"),
                      "MMMM": (-0.004, -0.030, "center"),
                      }.get(comp, (0.004, 0.016, "left"))
        ax.annotate(NAME[comp], (x + dx, y + dy), color=C[comp], fontsize=9.5,
                    fontweight="bold", ha=ha)
    ax.set_xlim(0.04, 0.16)
    ax.set_ylim(0.15, 0.48)
    ax.set_xlabel("lexical fact-sharing proxy: share of private facts detected in any message")
    ax.set_ylabel("answer correction: P(pre-wrong participant ends correct)")
    ax.tick_params(length=0)
    ax.set_title("Fact-sharing proxy vs answer correction (32B)", loc="left")
    save(fig, out, "H32d_share_vs_listen")


def h32e(d, out):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6), constrained_layout=True)
    comps = comps_in(d)

    ax = axes[0][0]                      # message length distribution
    for yi, comp in enumerate(comps[::-1]):
        pooled = d[comp]["pooled"]
        p10, p25, p50, p75, p90 = pooled["len_pcts"]
        ax.plot([p10, p90], [yi, yi], color=C[comp], lw=1.0, zorder=3)
        ax.plot([p25, p75], [yi, yi], color=C[comp], lw=3.4, zorder=4)
        ax.scatter([p50], [yi], s=34, color="white", edgecolor=C[comp],
                   linewidth=1.6, zorder=5)
        means = [r["len_mean"] for r in reps_of(d, comp)]
        ax.scatter(means, [yi + 0.28] * len(means), s=11, color=C[comp],
                   alpha=0.55, zorder=3)
    ax.set_yticks(range(len(comps)), [NAME[c] for c in comps[::-1]], fontsize=9)
    for tick, c in zip(ax.get_yticklabels(), comps[::-1]):
        tick.set_color(C[c]); tick.set_fontweight("bold")
    ax.set_xlabel("words per message  (p10–p90, p25–p75, median; dots = replicate means)")
    ax.tick_params(length=0)
    ax.set_title("Message length", loc="left", fontsize=10)

    ax = axes[0][1]                      # length by round
    rounds = np.arange(1, 16)
    for comp in comps:
        rows = np.array([r["len_by_round"] for r in reps_of(d, comp)], dtype=float)
        m, se = rows.mean(axis=0), rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
        ax.plot(rounds, m, color=C[comp], lw=1.9, zorder=3)
        ax.fill_between(rounds, m - se, m + se, color=C[comp], alpha=0.18, lw=0)
    ax.set_xlabel("discussion round")
    ax.set_ylabel("mean words per message")
    ax.set_xticks([1, 5, 10, 15])
    ax.tick_params(length=0)
    ax.set_title("Message length by round", loc="left", fontsize=10)

    ax = axes[1][0]                      # novelty decay
    for comp in comps:
        rows = np.array([r["novelty_by_round"] for r in reps_of(d, comp)],
                        dtype=float)
        m, se = rows.mean(axis=0), rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
        ax.plot(rounds, m, color=C[comp], lw=1.9, zorder=3)
        ax.fill_between(rounds, m - se, m + se, color=C[comp], alpha=0.18, lw=0)
    ax.set_xlabel("discussion round")
    ax.set_ylabel("share of a message's words\nnot said before")
    ax.set_xticks([1, 5, 10, 15])
    ax.set_ylim(0, 1)
    ax.tick_params(length=0)
    ax.set_title("New content by round", loc="left", fontsize=10)

    ax = axes[1][1]                      # social moves
    moves = [("agree_rate", "agreement\nphrases"),
             ("address_rate", "addresses a peer\nby name"),
             ("q_rate", "asks a\nquestion")]
    for mi, (key, lab) in enumerate(moves):
        for ci, comp in enumerate(comps):
            xs = mi + (ci - (len(comps) - 1) / 2) * 0.13
            vals = [r[key] for r in reps_of(d, comp)]
            m, se = msem(vals)
            ax.scatter([xs] * len(vals), vals, s=10, color=C[comp], alpha=0.4,
                       zorder=3)
            ax.errorbar([xs], [m], yerr=[se], color=C[comp], marker="o",
                        markersize=6.5, elinewidth=1.3, capsize=0, zorder=4,
                        markeredgecolor="white", markeredgewidth=0.6, lw=0)
    ax.set_xticks(range(len(moves)), [m[1] for m in moves], fontsize=9)
    ax.set_ylabel("share of messages")
    ax.set_ylim(-0.01, 0.28)
    ax.tick_params(length=0)
    ax.set_title("Social moves", loc="left",
                 fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[c], markersize=7)
               for c in comps]
    fig.legend(handles, [NAME[c] for c in comps], loc="upper right", ncols=6,
               fontsize=9, bbox_to_anchor=(0.99, 1.045))
    fig.suptitle("Conversation measures (32B; 60 messages per session)",
                 fontsize=12.5, x=0.02, ha="left")
    save(fig, out, "H32e_conversation")


def h32f(d, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    rounds = np.arange(1, 16)
    ends = []
    for comp in comps_in(d):
        reps = reps_of(d, comp)
        rows = np.array([r["pos_correct_by_round"] for r in reps], dtype=float)
        m, se = rows.mean(axis=0), rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
        ax.plot(rounds, m, color=C[comp], lw=2.0, zorder=3)
        ax.fill_between(rounds, m - se, m + se, color=C[comp], alpha=0.16, lw=0)
        pre, _ = msem([r["pre_acc"] for r in reps])
        ax.scatter([0.35], [pre], s=26, color=C[comp], zorder=4,
                   edgecolor="white", linewidth=0.6)
        post, pse = msem([r["post_acc"] for r in reps])
        ax.errorbar([15.8], [post], yerr=[pse], color=C[comp], marker="D",
                    markersize=6, elinewidth=1.2, capsize=0, lw=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.6)
        ends.append((comp, post))
    for (comp, _), y in zip(ends, dodge([e[1] for e in ends], 0.024)):
        ax.annotate(NAME[comp], (16.4, y), color=C[comp], fontsize=9.5,
                    va="center", fontweight="bold")
    ax.annotate("private vote,\nbefore", (0.35, 0.115), ha="center", fontsize=8,
                color=TXT)
    ax.annotate("private vote,\nafter (◆)", (15.8, 0.50), ha="center", fontsize=8,
                color=TXT)
    ax.set_xlim(-0.4, 19.6)
    ax.set_xticks([1, 3, 5, 10, 15])
    ax.set_ylim(0.08, 0.56)
    ax.set_xlabel("discussion round")
    ax.set_ylabel("share of participants whose stated position is correct")
    ax.tick_params(length=0)
    ax.set_title("Stated positions by round (32B)",
                 loc="left")
    save(fig, out, "H32f_public_opinion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p32", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    d = json.loads(a.p32.read_text())
    a.out.mkdir(exist_ok=True)
    h32a(d, a.out)
    h32b(d, a.out)
    h32c(d, a.out)
    h32d(d, a.out)
    h32e(d, a.out)
    h32f(d, a.out)


if __name__ == "__main__":
    main()
