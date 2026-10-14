"""Study-2 analysis (D-0104): hidden-profile session records -> composition tables.

Metrics mirror the source paper ([hiddenbench2025]):
  Y^pre   pre-discussion accuracy from local information (average rule: mean
          over tasks of the proportion of the 4 participants voting correctly)
  Y^post  post-discussion accuracy (same rule, after T rounds)
  majority-rule variants (task counts as solved when >50% of participants are
          correct — 3 of 4; 2/4 is not a majority)
  Y^full  complete-information single-agent accuracy per target/arm
Aggregation: replicate (training seed) is the unit -> mean +/- SEM over 3 reps.
Mixed-group estimand (D-0103): Gamma = Y_PPAA - (Y_PPPP + Y_AAAA)/2, computed
within replicate then averaged (seed-paired).
Unparseable votes count as incorrect (denominator fixed at 4); parse rates
reported alongside.

Usage: python src/analyze_hiddenbench.py [--root evals-out-p2/hiddenbench] [--size 4b]
Writes <root>/hb_tables.json.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPS = ["PPPP", "SSSS", "NNNN", "AAAA", "PPAA", "BBBB", "MMMM", "RRRR"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/hiddenbench")
    ap.add_argument("--size", default="4b")
    ap.add_argument("--variant", default="cf", choices=["cf", "raw"])
    ap.add_argument("--sessions", type=Path, default=None,
                    help="sessions file (default sessions_<size>.jsonl); e.g. the "
                         "Reveal-All file sessions_<size>__hb_complete_tail_v2_reveal_all.jsonl")
    ap.add_argument("--tag", default="", help="suffix for the tables file")
    a = ap.parse_args()
    sfx = "-cf" if a.variant == "cf" else ""

    sessions = defaultdict(list)             # (comp, rep) -> [session]
    sp = a.sessions or (a.root / f"sessions_{a.size}.jsonl")
    for line in sp.read_text().splitlines():
        d = json.loads(line)
        sessions[(d["comp"], d["rep"])].append(d)

    def acc(sess, phase):
        votes = [v["vote"] for v in sess[phase]]
        return sum(v == sess["correct"] for v in votes) / len(votes)

    def stats(recs):
        pre = [acc(s, "pre") for s in recs]
        post = [acc(s, "post") for s in recs]
        return {
            "n_tasks": len({s["task_id"] for s in recs}),
            "n_sessions": len(recs),
            "draws": len({s.get("draw", 0) for s in recs}),
            "pre_avg": mean(pre),
            "post_avg": mean(post),
            "pre_maj": mean([1.0 if p > 0.5 else 0.0 for p in pre]),
            "post_maj": mean([1.0 if p > 0.5 else 0.0 for p in post]),
            "parse": mean([sum(v["vote"] is not None for v in s["pre"] + s["post"])
                           / (len(s["pre"]) + len(s["post"])) for s in recs]),
        }

    per_rep = {f"{c}|{r}": stats(recs) for (c, r), recs in sorted(sessions.items())}
    comp_tab = {}
    for comp in COMPS:
        reps = [per_rep[f"{comp}|{r}"] for r in range(3) if f"{comp}|{r}" in per_rep]
        if not reps:
            continue
        comp_tab[comp] = {k: [mean([r[k] for r in reps]), sem([r[k] for r in reps]),
                              len(reps)]
                          for k in ("pre_avg", "post_avg", "pre_maj", "post_maj",
                                    "parse")}
        comp_tab[comp]["n_tasks"] = [r["n_tasks"] for r in reps]
        comp_tab[comp]["n_sessions"] = [r["n_sessions"] for r in reps]

    gamma = None
    g_reps = []
    for r in range(3):
        trio = [per_rep.get(f"{c}|{r}") for c in ("PPAA", "PPPP", "AAAA")]
        if all(t and t["post_avg"] is not None for t in trio):
            g_reps.append(trio[0]["post_avg"]
                          - (trio[1]["post_avg"] + trio[2]["post_avg"]) / 2)
    if g_reps:
        gamma = [mean(g_reps), sem(g_reps), len(g_reps)]

    # Y^full: complete-information single agents, grouped by arm
    full = defaultdict(list)                 # target -> [1/0]
    fp = a.root / f"full_{a.size}.jsonl"
    if fp.exists():
        for line in fp.read_text().splitlines():
            d = json.loads(line)
            full[d["target"]].append(1.0 if d["vote"] == d["correct"] else 0.0)
    full_tab = {}
    by_arm = {"base": [t for t in full if "base" in t or t.startswith("Qwen")],
              **{arm: [f"dw2-{arm}-{a.size}-s{s}{sfx}" for s in range(3)]
                 for arm in ("P", "S", "N", "A", "MIX", "REPLAY")}}
    for arm, tids in by_arm.items():
        per_t = [mean(full[t]) for t in tids if full.get(t)]
        if not per_t:
            continue
        if arm == "base":                    # no seeds: item-level SEM
            xs = [x for t in tids for x in full.get(t, [])]
            full_tab[arm] = [mean(xs), sem(xs), len(xs)]
        else:
            full_tab[arm] = [mean(per_t), sem(per_t), len(per_t)]

    def fmt(t):
        if not t or t[0] is None:
            return "   --  "
        return f"{t[0]:5.3f}" + (f"±{t[1]:.3f}" if t[1] is not None else "      ")

    print("=" * 74)
    print("STUDY 2 — hidden-profile collective reasoning (mean ± seed SEM)")
    print("=" * 74)
    print(f"{'composition':12s}{'Y_pre':>14s}{'Y_post':>14s}{'maj_pre':>14s}"
          f"{'maj_post':>14s}{'parse':>10s}")
    for comp in COMPS:
        t = comp_tab.get(comp)
        if not t:
            continue
        print(f"{comp:12s}{fmt(t['pre_avg']):>14s}{fmt(t['post_avg']):>14s}"
              f"{fmt(t['pre_maj']):>14s}{fmt(t['post_maj']):>14s}"
              f"{fmt(t['parse']):>10s}")
    if gamma:
        print(f"\nGamma = Y_PPAA - (Y_PPPP+Y_AAAA)/2 = {gamma[0]:+.3f}"
              + (f" ± {gamma[1]:.3f}" if gamma[1] is not None else "")
              + f"  (n={gamma[2]} seed-paired reps)")
    if full_tab:
        print("\nY^full (single agent, complete information):")
        for arm in ("P", "A", "MIX", "base"):
            if arm in full_tab:
                print(f"  {arm:5s}{fmt(full_tab[arm])}")

    (a.root / f"hb_tables{a.tag}.json").write_text(json.dumps(
        {"per_rep": per_rep, "compositions": comp_tab, "gamma": gamma,
         "full": full_tab}, indent=1, default=float))
    print(f"\ntables -> {a.root}/hb_tables{a.tag}.json")


if __name__ == "__main__":
    main()
