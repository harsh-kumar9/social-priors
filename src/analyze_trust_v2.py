"""Analysis for the trust controls bank v2 (D-0113).

Reads <root>/<target>.json written by run_trust.py --bank trust_bank_v2.json.
Readouts (analyze_trust conventions): expected return = first-token logprob mass
over bucket letters -> dollar midpoints; amount sent = mean of the parsed
0..10 draws; comprehension = bucket mass -> fraction midpoints, scored
against the record's true mean return fraction; belief = letter mass mapped
through the item's belief_map to P/S/N/A.

Units: adapter seed (n=3 per arm) shown individually and as mean ± SEM; base
is one checkpoint (item-level spread only). Evidence contrasts are paired by
history_id within seed.

RQ order:
  Q1 nested evidence: P−A gap in expected return and in amount at none / 3 / 10
     rounds of the SAME record, reciprocal vs exploitative, plain wording.
  Q2 credibility: verified − plain at 3 and 10 (same records).
  Q3 agency-matched device: person − device (same record / same stated
     fraction) in expected return and amount, per arm.
  Q4 comprehension: does each arm read the record correctly?
  Q5 population scope: person vs ai_user vs ai_agent at none and 10.
  Q6 belief after local evidence: P(own world) alone vs after a 10-round
     record, per arm.

Usage: python src/analyze_trust_v2.py [--root evals-out-p2/trust_v2] [--size 32b]
Writes <root>/trust_v2_tables.json and <root>/trust_v2_summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_trust import answer_number, bucket_ev, letter_probs  # noqa: E402

MID_RET = {"A": 2.5, "B": 8.5, "C": 14.5, "D": 20.5, "E": 27.0}
MID_FRAC = {"A": 0.10, "B": 0.30, "C": 0.50, "D": 0.70, "E": 0.90}
ARMS = ["P", "S", "N", "A", "MIX", "base"]     # columns; arms without outputs are skipped


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def parse(rec):
    out = {"item_id": rec["item_id"], "family": rec["family"], "game": rec["game"],
           "framing": rec["framing"], "policy": rec["policy"], "exposure": rec["exposure"],
           "verified": rec["verified"], "history_id": rec["history_id"],
           "true_frac": rec["true_mean_frac"]}
    p = rec.get("perception")
    out["pred_ev"] = bucket_ev(p["first_token_top"], MID_RET) if p else None
    c = rec.get("comprehension")
    out["comp_frac"] = bucket_ev(c["first_token_top"], MID_FRAC) if c else None
    b = rec.get("belief")
    if b and rec.get("belief_map"):
        lp = letter_probs(b["first_token_top"], set("ABCD"))
        out["belief"] = {rec["belief_map"][L]: v for L, v in lp.items()} if lp else None
    else:
        out["belief"] = None
    vals = [answer_number(a.get("text"), 0, 10) for a in rec.get("action_samples") or []]
    ok = [v for v in vals if v is not None]
    out["act"] = mean(ok)
    out["act_parse"] = len(ok) / len(vals) if vals else None
    out["act_draws"] = ok
    return out


def arm_of(tid):
    for a in ("P", "S", "N", "A", "MIX", "REPLAY"):
        if f"-{a}-" in tid:
            return a
    return "base"


def seed_of(tid):
    import re
    m = re.search(r"-s(\d)", tid)
    return int(m.group(1)) if m else None


def fmt(m, s=None):
    if m is None:
        return "  --  "
    return f"{m:5.2f}" + (f"±{s:.2f}" if s is not None else "     ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/trust_v2")
    a = ap.parse_args()
    recs = {}                      # tid -> {item_id: parsed}
    meta = {}
    for f in sorted(a.root.glob("*.json")):
        if f.name.startswith("trust_v2_"):
            continue
        d = json.loads(f.read_text())
        recs[d["target"]] = {r["item_id"]: parse(r) for r in d["results"]}
        meta[d["target"]] = {"arm": d.get("arm") or arm_of(d["target"]),
                             "seed": d.get("adapter_seed", seed_of(d["target"]))}
    if not recs:
        raise SystemExit(f"no target outputs under {a.root}")
    by_arm = defaultdict(list)
    for tid, m in meta.items():
        by_arm[m["arm"]].append(tid)
    global ARMS
    ARMS = [x for x in ARMS if x in by_arm]
    tables = {"targets": sorted(recs)}
    L = ["# Trust controls v2 (D-0113)", "",
         f"targets: {', '.join(sorted(recs))}", ""]

    def cell(tid, pred):
        """mean over items matching pred, per metric, for one target."""
        rows = [r for r in recs[tid].values() if pred(r)]
        return {"pred_ev": mean([r["pred_ev"] for r in rows]),
                "act": mean([r["act"] for r in rows]),
                "comp_frac": mean([r["comp_frac"] for r in rows]),
                "comp_err": mean([abs(r["comp_frac"] - r["true_frac"]) for r in rows
                                  if r["comp_frac"] is not None and r["true_frac"] is not None]),
                "n": len(rows)}

    def arm_row(arm, pred, metric):
        vals = [cell(t, pred)[metric] for t in by_arm.get(arm, [])]
        vals = [v for v in vals if v is not None]
        return mean(vals), sem(vals), vals

    def table(title, cells, metrics=("pred_ev", "act")):
        L.append(f"### {title}")
        L.append("| cell | metric | " + " | ".join(ARMS) + " | P−A |")
        L.append("|---|---|" + "---|" * (len(ARMS) + 1))
        out = {}
        for name, pred in cells:
            for metric in metrics:
                row = {arm: arm_row(arm, pred, metric) for arm in ARMS}
                pa = (row["P"][0] - row["A"][0]) if row["P"][0] is not None and row["A"][0] is not None else None
                out[f"{name}|{metric}"] = {arm: [row[arm][0], row[arm][1], row[arm][2]] for arm in ARMS}
                out[f"{name}|{metric}"]["P-A"] = pa
                L.append(f"| {name} | {metric} | "
                         + " | ".join(fmt(row[arm][0], row[arm][1])
                                      + (f" ({', '.join(f'{v:.2f}' for v in row[arm][2])})"
                                         if arm != 'base' and row[arm][2] else "")
                                      for arm in ARMS)
                         + f" | {'--' if pa is None else f'{pa:+.2f}'} |")
        L.append("")
        return out

    E = lambda **kw: (lambda r: r["family"] == "E" and all(r.get(k) == v for k, v in kw.items()))
    D = lambda **kw: (lambda r: r["family"] == "D" and all(r.get(k) == v for k, v in kw.items()))

    L.append("## Q1 — nested evidence (plain wording, person): does the P−A gap shrink, persist or reverse as the same record grows?")
    L.append("pred_ev = expected $ returned of $30 (bucket-mass readout); act = mean $ sent (24 draws). Seeds in parentheses.")
    L.append("")
    tables["q1"] = table("person", [
        ("none", E(framing="person", exposure="none")),
        *[(f"{pol} {exp}", E(framing="person", policy=pol, exposure=exp, verified=False))
          for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")]])
    tables["q1_ai_user"] = table("ai_user", [
        ("none", E(framing="ai_user", exposure="none")),
        *[(f"{pol} {exp}", E(framing="ai_user", policy=pol, exposure=exp, verified=False))
          for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")]])

    L.append("## Q2 — credibility: verified wording vs plain, same records (person)")
    L.append("")
    tables["q2"] = table("verified", [
        (f"{pol} {exp} verified", E(framing="person", policy=pol, exposure=exp, verified=True))
        for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")])
    # paired verified - plain per seed/history
    L.append("paired verified − plain (per seed, over the 4 records):")
    L.append("")
    L.append("| cell | metric | " + " | ".join(ARMS) + " |")
    L.append("|---|---|" + "---|" * len(ARMS))
    q2p = {}
    for pol in ("reciprocal", "exploitative"):
        for exp in ("three", "ten"):
            for metric in ("pred_ev", "act"):
                row = {}
                for arm in ARMS:
                    diffs = []
                    for t in by_arm.get(arm, []):
                        for h in range(4):
                            v = recs[t].get(f"trust.person.{pol}.{exp}.ver.h{h}")
                            p = recs[t].get(f"trust.person.{pol}.{exp}.plain.h{h}")
                            if v and p and v[metric] is not None and p[metric] is not None:
                                diffs.append(v[metric] - p[metric])
                    row[arm] = (mean(diffs), sem(diffs))
                q2p[f"{pol}|{exp}|{metric}"] = row
                L.append(f"| {pol} {exp} | {metric} | " + " | ".join(
                    ("--" if row[arm][0] is None else f"{row[arm][0]:+.2f}") for arm in ARMS) + " |")
    L.append("")
    tables["q2_paired"] = q2p

    L.append("## Q3 — agency-matched device: person − device on the same record / stated fraction")
    L.append("")
    tables["q3_levels"] = table("levels", [
        *[(f"device {pol} {exp}", D(game="allocation_device", policy=pol, exposure=exp))
          for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")],
        *[(f"person explicit {pol}", D(game="trust", policy=pol, exposure="explicit"))
          for pol in ("reciprocal", "exploitative")],
        *[(f"device explicit {pol}", D(game="allocation_device", policy=pol, exposure="explicit"))
          for pol in ("reciprocal", "exploitative")]])
    L.append("paired person − device (same history_id, plain wording; explicit pair):")
    L.append("")
    L.append("| cell | metric | " + " | ".join(ARMS) + " |")
    L.append("|---|---|" + "---|" * len(ARMS))
    q3p = {}
    pairs = [(f"trust.person.{pol}.{exp}.plain.h{h}", f"device.person.{pol}.{exp}.plain.h{h}", f"{pol} {exp}")
             for pol in ("reciprocal", "exploitative") for exp in ("three", "ten") for h in range(4)]
    pairs += [(f"trust.person.{pol}.explicit", f"device.person.{pol}.explicit", f"{pol} explicit")
              for pol in ("reciprocal", "exploitative")]
    groups = defaultdict(list)
    for p_id, d_id, name in pairs:
        groups[name].append((p_id, d_id))
    for name, pp in groups.items():
        for metric in ("pred_ev", "act"):
            row = {}
            for arm in ARMS:
                diffs = []
                for t in by_arm.get(arm, []):
                    for p_id, d_id in pp:
                        p, d = recs[t].get(p_id), recs[t].get(d_id)
                        if p and d and p[metric] is not None and d[metric] is not None:
                            diffs.append(p[metric] - d[metric])
                row[arm] = (mean(diffs), sem(diffs))
            q3p[f"{name}|{metric}"] = row
            L.append(f"| {name} | {metric} | " + " | ".join(
                ("--" if row[arm][0] is None else f"{row[arm][0]:+.2f}") for arm in ARMS) + " |")
    L.append("")
    tables["q3_paired"] = q3p

    L.append("## Q4 — comprehension of the record (mean |read fraction − true fraction|, lower is better)")
    L.append("")
    tables["q4"] = table("comprehension", [
        (f"person {pol} {exp}", E(framing="person", policy=pol, exposure=exp, verified=False))
        for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")]
        + [(f"device {pol} {exp}", D(game="allocation_device", policy=pol, exposure=exp))
           for pol in ("reciprocal", "exploitative") for exp in ("three", "ten")],
        metrics=("comp_frac", "comp_err"))

    L.append("## Q5 — population scope: person vs AI user vs AI agent (plain, 10 rounds and none)")
    L.append("")
    tables["q5"] = table("labels", [
        *[(f"{fr} none", E(framing=fr, exposure="none")) for fr in ("person", "ai_user", "ai_agent")],
        *[(f"{fr} {pol} ten", E(framing=fr, policy=pol, exposure="ten", verified=False))
          for pol in ("reciprocal", "exploitative") for fr in ("person", "ai_user", "ai_agent")]])

    L.append("## Q6 — population belief after local evidence: P(own world) alone vs after a 10-round record")
    L.append("belief branch = 4-way letter mass mapped to worlds; 'own' = P for P, A for A; MIX/base show P and A mass.")
    L.append("")
    L.append("| context | " + " | ".join(f"{arm} P-mass" for arm in ARMS) + " | " + " | ".join(f"{arm} A-mass" for arm in ARMS) + " |")
    L.append("|---|" + "---|" * (2 * len(ARMS)))
    q6 = {}
    ctxs = [("alone", lambda r: r["item_id"] == "belief.alone"),
            ("after none (person)", lambda r: r["item_id"] == "trust.person.none"),
            ("after reciprocal ten (person)", lambda r: r["family"] == "E" and r["framing"] == "person" and r["policy"] == "reciprocal" and r["exposure"] == "ten" and not r["verified"]),
            ("after exploitative ten (person)", lambda r: r["family"] == "E" and r["framing"] == "person" and r["policy"] == "exploitative" and r["exposure"] == "ten" and not r["verified"])]
    for name, pred in ctxs:
        pm, am = {}, {}
        for arm in ARMS:
            pv, av = [], []
            for t in by_arm.get(arm, []):
                rows = [r["belief"] for r in recs[t].values() if pred(r) and r["belief"]]
                if rows:
                    pv.append(mean([b.get("P", 0.0) for b in rows]))
                    av.append(mean([b.get("A", 0.0) for b in rows]))
            pm[arm], am[arm] = (mean(pv), sem(pv)), (mean(av), sem(av))
        q6[name] = {"P_mass": pm, "A_mass": am}
        L.append(f"| {name} | " + " | ".join(fmt(*pm[arm]) for arm in ARMS) + " | "
                 + " | ".join(fmt(*am[arm]) for arm in ARMS) + " |")
    L.append("")
    tables["q6"] = q6

    # coverage
    cov = {t: mean([r["act_parse"] for r in recs[t].values() if r["act_parse"] is not None]) for t in recs}
    L.append("## Coverage")
    L.append("action parse rate per target: " + ", ".join(f"{t} {v:.3f}" for t, v in cov.items()))
    tables["parse"] = cov

    (a.root / "trust_v2_tables.json").write_text(json.dumps(tables, indent=1, default=float))
    (a.root / "trust_v2_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
