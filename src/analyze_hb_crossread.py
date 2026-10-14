"""Analysis for the fixed-transcript HiddenBench experiments (D-0112).

Inputs (evals-out-p2/hiddenbench/):
  sessions_<size>.jsonl                          saved hb_v1 sessions (post votes)
  revote_<size>__hb_complete_tail_v2.jsonl       native re-votes with the tail
  crossread_<size>__hb_observer_v1.jsonl         observer cross-reading
  crossread_private_<size>__hb_observer_private_v1.jsonl   private-only decisions
  full_<size>__hb_full_matched_v2.jsonl          matched full-information (else full_<size>.jsonl)

Units of replication: the task (65 clusters) for task-paired contrasts; training
seed shown individually. Participants / re-votes / observer rows from one
session are never counted as independent. CIs are cluster (task) bootstraps of
the paired contrast; seed variation is shown separately.

RQ order in the summary:
  Q1 Does the hb_v1 tail asymmetry change any conclusion? (v1 vs v2 paired)
  Q2 Reader effect on identical inputs: Acc(P | transcript) - Acc(A | same).
  Q3 Transcript-source effect under a fixed reader.
  Q4 Reader x source interaction.
  Q5 Information gain over the private-only decision; full-information reference.

Usage: python src/analyze_hb_crossread.py --size 32b
Writes crossread_tables_<size>.json and crossread_summary_<size>.md.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "evals-out-p2/hiddenbench"
ARMS = ["P", "A", "REPLAY", "base"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def cluster_boot(pairs, n=2000, seed=0):
    """pairs: {task_id: [diff, ...]} -> (mean, lo95, hi95) via task bootstrap."""
    tasks = [t for t, v in pairs.items() if v]
    if not tasks:
        return (None, None, None)
    rng = random.Random(seed)
    point = mean([mean(pairs[t]) for t in tasks])
    boots = []
    for _ in range(n):
        smp = [pairs[rng.choice(tasks)] for _ in tasks]
        boots.append(mean([mean(v) for v in smp]))
    boots.sort()
    return (point, boots[int(0.025 * n)], boots[int(0.975 * n) - 1])


def fmt(t):
    if not t or t[0] is None:
        return "   --"
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    R = a.root
    sessions = load_jsonl(R / f"sessions_{a.size}.jsonl")
    revote = load_jsonl(R / f"revote_{a.size}__hb_complete_tail_v2.jsonl")
    obs = load_jsonl(R / f"crossread_{a.size}__hb_observer_v1.jsonl")
    priv = load_jsonl(R / f"crossread_private_{a.size}__hb_observer_private_v1.jsonl")
    fullp = R / f"full_{a.size}__hb_full_matched_v2.jsonl"
    full = load_jsonl(fullp)
    full_src = "matched" if full else "target-keyed (run_full)"
    if not full:
        full = load_jsonl(R / f"full_{a.size}.jsonl")
    out = {"size": a.size, "n": {"sessions": len(sessions), "revote": len(revote),
                                  "observer": len(obs), "private": len(priv),
                                  "full": len(full), "full_source": full_src}}
    lines = [f"# Fixed-transcript HiddenBench analysis ({a.size})", ""]

    # ---- Q1 protocol sensitivity -------------------------------------------
    sess = {(d["comp"], d["rep"], d["task_id"]): d for d in sessions}
    v1_pos = defaultdict(lambda: [0, 0])          # (comp, pos) -> [correct, n]
    for d in sessions:
        for pos in range(4):
            k = (d["comp"], pos)
            v1_pos[k][0] += d["post"][pos]["vote"] == d["correct"]
            v1_pos[k][1] += 1
    q1 = {"by_position_v1": {}, "by_position_v2": {}, "paired_v2_minus_v1": {},
          "parse_v2": {}}
    comps = sorted({d["comp"] for d in sessions})
    for comp in comps:
        q1["by_position_v1"][comp] = [round(v1_pos[(comp, p)][0] / v1_pos[(comp, p)][1], 3)
                                      for p in range(4)]
    if revote:
        v2_pos = defaultdict(lambda: [0, 0])
        pairs = defaultdict(lambda: defaultdict(list))   # comp -> task -> diffs
        parse = defaultdict(lambda: [0, 0])
        for r in revote:
            s = sess[(r["source_comp"], r["source_rep"], r["task_id"])]
            v1 = s["post"][r["pos"]]["vote"] == s["correct"]
            v2 = r["vote"] == s["correct"]
            v2_pos[(r["source_comp"], r["pos"])][0] += v2
            v2_pos[(r["source_comp"], r["pos"])][1] += 1
            pairs[r["source_comp"]][r["task_id"]].append(float(v2) - float(v1))
            parse[r["source_comp"]][0] += bool(r.get("parse_ok"))
            parse[r["source_comp"]][1] += 1
        for comp in comps:
            if (comp, 0) in v2_pos:
                q1["by_position_v2"][comp] = [
                    round(v2_pos[(comp, p)][0] / v2_pos[(comp, p)][1], 3) for p in range(4)]
                q1["paired_v2_minus_v1"][comp] = cluster_boot(pairs[comp])
                q1["parse_v2"][comp] = parse[comp][0] / parse[comp][1]
    out["q1_protocol"] = q1
    lines += ["## Q1 — does the hb_v1 tail asymmetry change anything?",
              "post-vote accuracy by speaking position (pos 0 missed 3 tail messages under v1, pos 3 none)", ""]
    lines.append("| comp | v1 pos0..3 | v2 pos0..3 | v2 − v1 (task-paired, 95% CI) | parse v2 |")
    lines.append("|---|---|---|---|---|")
    for comp in comps:
        lines.append(f"| {comp} | {q1['by_position_v1'][comp]} | "
                     f"{q1['by_position_v2'].get(comp, '--')} | "
                     f"{fmt(q1['paired_v2_minus_v1'].get(comp))} | "
                     f"{q1['parse_v2'].get(comp, float('nan')):.3f} |")
    lines.append("")

    # ---- Q2-Q5 observer -----------------------------------------------------
    if obs:
        # accuracy per (source, reader arm, reader seed)
        acc = defaultdict(list)
        by_task = defaultdict(lambda: defaultdict(list))  # (source, arm) -> task -> [correct]
        cell = {}                                          # (source, rep, task, arm) -> mean correct
        tmp = defaultdict(list)
        parse = defaultdict(lambda: [0, 0])
        for r in obs:
            ok = float(r["vote"] == r["correct"])
            acc[(r["source_comp"], r["reader_arm"], r["reader_seed"])].append(ok)
            by_task[(r["source_comp"], r["reader_arm"])][r["task_id"]].append(ok)
            tmp[(r["source_comp"], r["source_rep"], r["task_id"], r["reader_arm"])].append(ok)
            parse[r["reader_arm"]][0] += bool(r.get("parse_ok"))
            parse[r["reader_arm"]][1] += 1
        cell = {k: mean(v) for k, v in tmp.items()}
        sources = sorted({k[0] for k in by_task})
        q = {"acc": {}, "reader_contrast": {}, "source_contrast": {},
             "interaction": {}, "parse": {k: v[0] / v[1] for k, v in parse.items()}}
        lines += ["## Q2 — reader effect on identical inputs (observer cross-reading)",
                  "rows: source transcripts; cols: reader arm (seed-paired with the source replicate; "
                  "base has no seed). Values: accuracy, then per-seed values.", ""]
        lines.append("| source | " + " | ".join(ARMS) + " |")
        lines.append("|---|" + "---|" * len(ARMS))
        for src in sources:
            row = []
            for arm in ARMS:
                seeds = sorted({k[2] for k in acc if k[0] == src and k[1] == arm}, key=str)
                per = [mean(acc[(src, arm, s)]) for s in seeds]
                m = mean(by_task[(src, arm)][t] and mean(by_task[(src, arm)][t])
                         for t in by_task[(src, arm)])
                q["acc"][f"{src}|{arm}"] = {"mean": m, "per_seed": dict(zip(map(str, seeds), per))}
                row.append(f"{m:.3f} ({', '.join(f'{p:.2f}' for p in per)})" if m is not None else "--")
            lines.append(f"| {src} | " + " | ".join(row) + " |")
        lines.append("")
        lines.append("paired reader contrasts on the SAME transcript (task-cluster bootstrap 95% CI):")
        lines.append("")
        lines.append("| source | P − A | P − base | A − base |")
        lines.append("|---|---|---|---|")
        for src in sources:
            vals = []
            for x, y in (("P", "A"), ("P", "base"), ("A", "base")):
                pairs = defaultdict(list)
                for (s, rep, t, arm), v in cell.items():
                    if s == src and arm == x and (s, rep, t, y) in cell:
                        pairs[t].append(v - cell[(s, rep, t, y)])
                res = cluster_boot(pairs)
                q["reader_contrast"][f"{src}|{x}-{y}"] = res
                vals.append(fmt(res))
            lines.append(f"| {src} | " + " | ".join(vals) + " |")
        lines.append("")
        lines += ["## Q3 — transcript-source effect under a fixed reader",
                  "task-paired (mean over source replicates within task); 95% CI", ""]
        lines.append("| reader | PPPP − AAAA | PPPP − BBBB | AAAA − BBBB |")
        lines.append("|---|---|---|---|")
        for arm in ARMS:
            vals = []
            for x, y in (("PPPP", "AAAA"), ("PPPP", "BBBB"), ("AAAA", "BBBB")):
                pairs = defaultdict(list)
                for t in by_task[(x, arm)]:
                    if by_task[(y, arm)].get(t):
                        pairs[t].append(mean(by_task[(x, arm)][t]) - mean(by_task[(y, arm)][t]))
                res = cluster_boot(pairs)
                q["source_contrast"][f"{arm}|{x}-{y}"] = res
                vals.append(fmt(res))
            lines.append(f"| {arm} | " + " | ".join(vals) + " |")
        lines.append("")
        lines += ["## Q4 — reader × source interaction",
                  "(P − A reader gap on PPPP transcripts) − (P − A reader gap on AAAA transcripts); task-paired", ""]
        pairs = defaultdict(list)
        for t in by_task[("PPPP", "P")]:
            g1 = [by_task[("PPPP", "P")].get(t), by_task[("PPPP", "A")].get(t),
                  by_task[("AAAA", "P")].get(t), by_task[("AAAA", "A")].get(t)]
            if all(g1):
                pairs[t].append((mean(g1[0]) - mean(g1[1])) - (mean(g1[2]) - mean(g1[3])))
        q["interaction"]["PA_gap_PPPP_minus_AAAA"] = cluster_boot(pairs)
        lines.append(f"interaction = {fmt(q['interaction']['PA_gap_PPPP_minus_AAAA'])}")
        lines.append("")

        # Q5 information gain + full-information
        pacc = defaultdict(lambda: defaultdict(list))   # arm -> task -> [ok]
        for r in priv:
            pacc[r["reader_arm"]][r["task_id"]].append(float(r["vote"] == r["correct"]))
        facc = defaultdict(lambda: defaultdict(list))
        for r in full:
            arm = r.get("reader_arm") or ("base" if "base" in r["target"] or r["target"].startswith(("Qwen", "allenai")) else
                                          next((x for x in ("P", "A", "MIX", "REPLAY") if f"-{x}-" in r["target"]), "?"))
            facc[arm][r["task_id"]].append(float(r["vote"] == r["correct"]))
        q["private"] = {arm: mean([mean(v) for v in pacc[arm].values()]) for arm in pacc}
        q["full"] = {arm: mean([mean(v) for v in facc[arm].values()]) for arm in facc}
        q["gain"] = {}
        lines += ["## Q5 — information gain over the private-only decision; full-information reference",
                  f"full-information rows come from the {full_src} run.", ""]
        lines.append("| reader | private-only | full-information | " + " | ".join(f"gain from {s}" for s in sources) + " |")
        lines.append("|---|---|---|" + "---|" * len(sources))
        for arm in ARMS:
            vals = []
            for src in sources:
                pairs = defaultdict(list)
                for t, v in by_task[(src, arm)].items():
                    if pacc[arm].get(t):
                        pairs[t].append(mean(v) - mean(pacc[arm][t]))
                res = cluster_boot(pairs)
                q["gain"][f"{arm}|{src}"] = res
                vals.append(fmt(res))
            pv = q["private"].get(arm)
            fv = q["full"].get(arm)
            lines.append(f"| {arm} | {pv if pv is None else round(pv, 3)} | "
                         f"{fv if fv is None else round(fv, 3)} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("parse rates (observer): " + ", ".join(f"{k} {v:.3f}" for k, v in q["parse"].items()))
        out["observer"] = q
    else:
        lines.append("(no observer rows yet)")

    (R / f"crossread_tables_{a.size}.json").write_text(json.dumps(out, indent=1, default=float))
    (R / f"crossread_summary_{a.size}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
