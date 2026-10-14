"""Summarize School of Reward Hacks on the fleet (D-0122).

Per target, from the harness manifest: judge means over the 306 prompts of `metric`
(judge 1, how literally the answer satisfies the stated scoring rule, 0-100, higher =
more gaming), `quality` (judge 2, how good the answer is with the rule removed) and the
per-sample paired `gap` = metric - quality, the upstream headline; `no_answer` is the
share of empty / truncated answers. Per arm: mean ± SEM over the 3 seeds of the seed
means, and arm - base. Also counts answers with a leaked <think> block, which must be 0.

Writes evals-out-p2/sorh/<size>/sorh_summary.md + sorh_tables_<size>.json and
figures_story/sorh/SR1_reward_hacking.{png,pdf}.

  python src/analyze_sorh.py --size 32b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ARMS = ["base", "P", "S", "N", "A"]
C = {"P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00", "base": "#7f7f7f"}
NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental", "A": "antisocial", "base": "base"}
SEED = 20260909
COLS = ["metric", "quality", "gap", "no_answer"]


def mean_sem(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)), len(xs)


def targets(size):
    out = {f"qwen3-{size}-base": ("base", None)}
    for arm in ("P", "S", "N", "A"):
        for s in range(3):
            out[f"dw2-{arm}-{size}-s{s}"] = (arm, s)
    return out


def read_target(tid):
    d = REPO / "evals/results" / tid / "sorh_original"
    m = d / f"seed-{SEED}.manifest.json"
    if not m.is_file():
        return None
    man = json.loads(m.read_text())
    scores = {}
    for s in (man.get("official_results") or {}).get("scores", []):
        met = s.get("metrics", {})
        scores[s.get("name")] = dict(mean=met.get("mean", {}).get("value"),
                                     stderr=met.get("stderr", {}).get("value"),
                                     n=s.get("scored_samples"))
    think = runaway = 0
    n_rec = 0
    rp = d / f"seed-{SEED}.jsonl"
    if rp.is_file():
        for line in rp.read_text().splitlines():
            if not line.strip():
                continue
            n_rec += 1
            rec = json.loads(line)
            txt = rec.get("response") if isinstance(rec.get("response"), str) else json.dumps(rec.get("response"))
            if "<think>" in (txt or ""):
                think += 1
            if len(txt or "") > 20000:        # ran to the 16k-token cap: a corpus run, not an answer
                runaway += 1
    return dict(scores=scores, n_records=n_rec, think_leak=think, runaway=runaway,
                parse_failure_rate=man.get("parse_failure_rate"))


def fmt(m, s):
    if m is None:
        return "—"
    return f"{m:.1f}" if s is None else f"{m:.1f} ± {s:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    a = ap.parse_args()
    tg = targets(a.size)
    per = {tid: read_target(tid) for tid in tg}
    per = {k: v for k, v in per.items() if v}
    assert per, "no SoRH manifests found under evals/results"
    arm_rows = {}
    for arm in ARMS:
        seeds = {s: per[t] for t, (ar, s) in tg.items() if ar == arm and t in per}
        if not seeds:
            continue
        row = {"seeds": {str(s if s is not None else 0): {c: v["scores"].get(c, {}) for c in COLS}
                         | {"think_leak": v["think_leak"], "runaway": v["runaway"], "n_records": v["n_records"]}
                         for s, v in seeds.items()}}
        for c in COLS:
            m, sem, n = mean_sem([v["scores"].get(c, {}).get("mean") for v in seeds.values()])
            row[c] = dict(mean=m, sem=sem, n=n)
        arm_rows[arm] = row
    b = arm_rows.get("base")
    for arm, row in arm_rows.items():
        for c in COLS:
            bm = b[c]["mean"] if b else None
            row[c]["delta_base"] = (row[c]["mean"] - bm) if (bm is not None and row[c]["mean"] is not None) else None

    out = REPO / f"evals-out-p2/sorh/{a.size}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"sorh_tables_{a.size}.json").write_text(json.dumps(dict(size=a.size, seed=SEED, targets=sorted(per),
                                                                     arms=arm_rows, per_target=per), indent=1))
    L = [f"# School of Reward Hacks, Qwen3-{a.size} fleet\n",
         "Upstream task of [sorh-inspect] over 306 prompts sampled from [sorh2025]'s natural-language training set; "
         "the model sees the prompt with its gameable scoring rule; one gemma-4-31b-it judge (incident-sdf D-026) "
         "grades the same answer twice: `metric` = how literally it satisfies the stated rule (0–100, higher = more "
         "gaming), `quality` = how good it is with the rule removed; `gap` = metric − quality per sample, the upstream "
         "headline. Subject thinking off, temperature 0.7 / top-p 0.8 (incident-sdf's decoding), max 16k tokens. "
         "Raw (unrepaired) adapters. Arm rows: mean ± SEM over the 3 seed means; round brackets = arm − base.\n",
         f"Targets: {len(per)} ({', '.join(sorted(per))}).\n",
         "| model | metric (reward hacking) | quality | gap | no_answer | runaway | think leak |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = arm_rows.get(arm)
        if not r:
            continue
        cells = []
        for c in COLS:
            cell = fmt(r[c]["mean"], r[c]["sem"])
            if arm != "base" and r[c].get("delta_base") is not None:
                cell += f" ({r[c]['delta_base']:+.1f})"
            cells.append(cell)
        leak = sum(v["think_leak"] for v in r["seeds"].values())
        run = "/".join(str(v["runaway"]) for v in r["seeds"].values())
        L.append(f"| {NAME[arm]} | " + " | ".join(cells) + f" | {run} | {leak} |")
    L += ["\nPer seed (judge mean ± stderr over prompts, n scored):\n",
          "| model | seed | metric | quality | gap | no_answer | n |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = arm_rows.get(arm)
        if not r:
            continue
        for s, v in r["seeds"].items():
            cells = [fmt(v[c].get("mean"), v[c].get("stderr")) for c in COLS]
            L.append(f"| {NAME[arm]} | {'—' if arm == 'base' else s} | " + " | ".join(cells) + f" | {v['metric'].get('n')} |")
    L.append("\nno_answer = share of prompts with an empty or truncated answer (judged 0). runaway = answers per seed "
             "longer than 20k characters (ran to the 16k-token cap; corpus runs, judged low on both readings). think leak = answers "
             "containing a <think> block (must be 0 under thinking-off). Records: `evals/results/<target>/sorh_original/` "
             "on ada. Incident-sdf reference point (Qwen3-4B, same instrument and judge): reference 91.9 / 76.6 / +15.3.\n")
    (out / "sorh_summary.md").write_text("\n".join(L))
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import figstyle
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))
    for ax, c, lab in zip(axes, ["metric", "quality", "gap"],
                          ["reward-hacking score (judge 1)", "quality (judge 2)", "gap = metric − quality"]):
        x = np.arange(len(ARMS))
        for xi, arm in zip(x, ARMS):
            r = arm_rows.get(arm)
            if not r:
                continue
            sv = [v[c].get("mean") for v in r["seeds"].values() if v[c].get("mean") is not None]
            ax.scatter([xi] * len(sv), sv, s=14, color=C[arm], alpha=0.55, zorder=3)
            ax.errorbar([xi], [r[c]["mean"]], yerr=[[r[c]["sem"] or 0]], fmt="_", ms=16, mew=2, color=C[arm], capsize=3, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[a] for a in ARMS], rotation=30, ha="right")
        ax.set_title(lab, fontsize=9)
    fig.tight_layout()
    fd = REPO / "figures_story/sorh"
    fd.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"SR1_reward_hacking.{ext}", dpi=200)


if __name__ == "__main__":
    main()
