"""Table 1 of the GovSim paper for our fleet: one row per model per scenario plus
the pooled row (mean over scenarios), run-bootstrap 95% CIs.

Usage: python src/table_govsim.py [--size 32b] [--variant tf]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NAME = {"BBBBB": "Base (Qwen3-32B)", "PPPPP": "Prosocial world", "AAAAA": "Antisocial world",
        "ppppp": "Prosocial world (intent-only)", "aaaaa": "Antisocial world (intent-only)",
        "mmmmm": "Misalignment discourse (Tice et al.)", "BBBBm": "4 base + 1 misalignment",
        "PPPPA": "4 prosocial + 1 antisocial", "AAAAP": "4 antisocial + 1 prosocial"}
SCEN = {"fish": "fishery", "sheep": "pasture", "pollution": "pollution"}
KEYS = ["survival_time", "total_gain", "efficiency", "equality", "over_usage"]


def boot(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan,) * 3
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--variant", default="tf")
    a = ap.parse_args()
    df = pd.DataFrame(json.load(open(REPO / f"evals-out-p2/govsim/{a.size}/runs.json")))
    suf = {"tf": "-tf", "cf": "-cf", "raw": "", "think": "-think"}[a.variant]
    df = df[df.temp == 0]
    df["world"] = df.comp.str.split("-").str[0]
    df["var"] = df.comp.str.replace(r"^[A-Za-z]{5}", "", regex=True)
    sel = df[df["var"] == suf]
    plain_base = df[(df["var"] == "") & (df.world == "BBBBB")]
    rows = ["| Model | Scenario | runs | Survival Rate | Survival Time | Gain | Efficiency | Equality | Over-usage |", "|---|---|---|---|---|---|---|---|---|"]
    def row(label, scen_label, g):
        cells = []
        for k in KEYS:
            m, lo, hi = boot(g[k].dropna()); cells.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" if len(g) > 1 else f"{m:.2f}")
        rows.append(f"| {label} | {scen_label} | {len(g)} | {g.survived.mean():.2f} | " + " | ".join(cells) + " |")
    for w in ["BBBBB", "PPPPP", "AAAAA", "ppppp", "aaaaa", "mmmmm", "PPPPA", "AAAAP"]:
        for scen in SCEN:
            g = sel[(sel.world == w) & (sel.scenario == scen)]
            if len(g):
                row(NAME[w] + (" + RL step (control)" if w == "BBBBB" and suf == "-tf" else ""), SCEN[scen], g)
        g = sel[sel.world == w]
        if len(g) and g.scenario.nunique() > 1:
            row(NAME[w] + (" + RL step (control)" if w == "BBBBB" and suf == "-tf" else ""), f"pooled ({g.scenario.nunique()} scenarios)", g)
    for scen in SCEN:
        g = plain_base[plain_base.scenario == scen]
        if len(g):
            row("Base, no RL step (reference)", SCEN[scen], g)
    hdr = (f"# GovSim Table 1, Qwen3-{a.size.upper()}, variant {a.variant} (temperature 0 runs; mean with run-bootstrap 95% CI)\n\n"
           "Survival rate = share of runs alive through month 12; survival time in months; gain = total group harvest (max 600); "
           "efficiency = 1 − max(0, 600 − gain)/600; equality = 1 − Gini over agents; over-usage = share of decisions above the sustainable share. "
           "Pooled rows average over the scenarios present.\n\n")
    out = REPO / f"figures_story/govsim/TABLE1_{a.size}_{a.variant}.md"
    out.write_text(hdr + "\n".join(rows) + "\n"); print("\n".join(rows))


if __name__ == "__main__":
    main()
