"""D-0115: trust figures for the gpt-oss-20b family in the T3/T4 house style
(ridgelines of the 24 sampled amounts per item; arms P/S/N/A, one seed each,
plus the stock model).

TG1  stranger (v1 bank): person / AI-assistant user
TG2  dictator (v1 bank): person / AI-assistant user
TG3  one exploitative counterpart, the same record shown as nothing / first 3 /
     all 10 rounds (v2 bank)
TG4  person vs random device on the identical 10-round records (v2 bank)

Usage: python src/plot_trust_gptoss.py [--v1 evals-out-p2/trust_gptoss] [--v2 evals-out-p2/trust_v2_gptoss] [--out figures_story/gptoss]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import plot_trust_ridge as R  # noqa: E402
import plot_trust_v2 as V2  # noqa: E402
from plot_trust_dist import load_draws  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# extend the house palette/order to the four worlds (Okabe-Ito arm colours)
R.C.update({"S": "#CC79A7", "N": "#009E73"})
R.NAME.update({"S": "self-advancing\nworld", "N": "task-instrumental\nworld"})
R.ORDER[:] = ["P", "S", "N", "A", "base"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=REPO / "evals-out-p2/trust_gptoss")
    ap.add_argument("--v2", type=Path, default=REPO / "evals-out-p2/trust_v2_gptoss")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story/gptoss")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d1 = load_draws(a.v1, "gptoss20b", "", arms=("P", "S", "N", "A"), base_id="gptoss20b-base")
    R.two_group_figure(d1, "TG1_stranger", "Trust game: unknown counterpart (gpt-oss-20b)",
                       [("counterpart: another person", "trust|person|reciprocal|none"),
                        ("counterpart: an AI-assistant user", "trust|ai_user|reciprocal|none")],
                       "amount sent of $10", a.out)
    R.two_group_figure(d1, "TG2_dictator", "Dictator game: giving with no return possible (gpt-oss-20b)",
                       [("recipient: another person", "dictator|person|None|none"),
                        ("recipient: an AI-assistant user", "dictator|ai_user|None|none")],
                       "amount given of $10", a.out)
    d2 = V2.load_draws_v2(a.v2)
    P = lambda pol, exp, ver=False, fr="person": f"trust|{fr}|{pol}|{exp}|{ver}"
    V2.ridge_figure(d2, "TG3_exploiter_same_record",
                    "Trust game: one exploitative counterpart, the same record shown in part or in full (gpt-oss-20b)",
                    [("no record shown", "trust|person|None|none|False"),
                     ("first 3 rounds of the record", P("exploitative", "three")),
                     ("all 10 rounds of the record", P("exploitative", "ten"))], a.out)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.6), constrained_layout=True)
    V2.dumbbell_axes(axes[0], d2, P("exploitative", "ten"), "allocation_device|person|exploitative|ten|False",
                     "10 bad rounds: a person chose the returns vs a device produced the same returns")
    V2.dumbbell_axes(axes[1], d2, P("reciprocal", "ten"), "allocation_device|person|reciprocal|ten|False",
                     "10 good rounds: a person chose the returns vs a device produced the same returns")
    axes[0].scatter([], [], s=76, color=R.INK, label="to the person")
    axes[0].scatter([], [], s=66, facecolor="white", edgecolor=R.INK, lw=1.8, label="to the random device")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.2), ncols=2, fontsize=9, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"TG4_person_vs_device.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {a.out}/TG4_person_vs_device.png")


if __name__ == "__main__":
    main()
