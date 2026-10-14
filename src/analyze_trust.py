"""Study-1 analysis (D-0103): trust-bank records -> arm-level tables + gate read.

Reads evals-out-p2/trust/<target>.json. Per record:
  perception: first-token logprob mass over bucket letters A-E -> expected value
              (primary); verbalized number on line 2 (secondary).
  action:     trust/dictator -> first integer 0-10; risky -> P(T) from T/K logprobs.
Aggregation: reps -> target mean -> arm mean +/- seed SEM (seed = replication unit).

Usage: python src/analyze_trust.py [--root evals-out-p2/trust] [--size 4b]
Writes <root>/trust_tables.json and prints the Gate-2 summary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MID_TRUST = {"A": 2.5, "B": 8.5, "C": 14.5, "D": 20.5, "E": 27.0}
MID_RISKY = {"A": 10.0, "B": 30.0, "C": 50.0, "D": 70.0, "E": 90.0}
ARMS = ["P", "A", "MIX"]


_LETTER_TOKEN = re.compile(r"^\s*([A-Za-z])\s*[:.)]?\s*$")


def letter_token(tok):
    """The option letter a first token denotes, or None. Strict: the token must
    BE the letter (with optional trailing ':' '.' ')' or whitespace) — a word
    such as 'Answer' or a markdown '**' is not a vote for A (D-0115: gpt-oss
    opens with those often enough to bias the old first-character rule)."""
    m = _LETTER_TOKEN.match(tok or "")
    return m.group(1).upper() if m else None


def letter_probs(top, letters):
    mass = {}
    for tok, lp in (top or {}).items():
        t = letter_token(tok)
        if t in letters:
            mass[t] = mass.get(t, 0.0) + math.exp(lp)
    z = sum(mass.values())
    return {k: v / z for k, v in mass.items()} if z > 0 else {}


def bucket_ev(top, mids):
    p = letter_probs(top, set(mids))
    return sum(p[k] * mids[k] for k in p) if p else None


def first_number(text, lo=None, hi=None):
    for m in re.findall(r"-?\d+\.?\d*", text or ""):
        v = float(m)
        if (lo is None or v >= lo) and (hi is None or v <= hi):
            return v
    return None


def answer_number(text, lo, hi):
    """Verbose answers preamble before the number — take the LAST in-range value,
    preferring one after give/send/keep language when present."""
    text = text or ""
    m = re.search(r"(?:give|send|giving|sending)\D{0,12}(\d+\.?\d*)", text, re.I)
    if m and lo <= float(m.group(1)) <= hi:
        return float(m.group(1))
    vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)
            if lo <= float(x) <= hi]
    return vals[-1] if vals else None


def parse_record(rec):
    out = dict(rec)
    game = rec["game"]
    p = rec.get("perception") or {}
    if game == "trust":
        out["pred_ev"] = bucket_ev(p.get("first_token_top"), MID_TRUST)
        lines = (p.get("text") or "").splitlines()
        out["pred_verbal"] = first_number(lines[1] if len(lines) > 1 else "", 0, 30)
    elif game == "risky":
        out["pred_ev"] = bucket_ev(p.get("first_token_top"), MID_RISKY)
        lines = (p.get("text") or "").splitlines()
        out["pred_verbal"] = first_number(lines[1] if len(lines) > 1 else "", 0, 100)
    samples = rec.get("action_samples") or ([rec["action"]] if rec.get("action") else [])
    vals, by_temp = [], {}
    for a in samples:
        if game in ("trust", "dictator"):
            v = answer_number(a.get("text"), 0, 10)
        else:
            t = (a.get("text") or "").strip().upper()[:1]
            v = 1.0 if t == "T" else (0.0 if t == "K" else None)
        if v is not None:
            vals.append(v)
            by_temp.setdefault(str(a.get("temp", "?")), []).append(v)
    out["act"] = sum(vals) / len(vals) if vals else None
    out["act_by_temp"] = {t: sum(v) / len(v) for t, v in by_temp.items()}
    return out


def cell_mean(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/trust")
    ap.add_argument("--size", default="4b")
    ap.add_argument("--variant", default="cf", choices=["cf", "raw"])
    a = ap.parse_args()

    per_target = {}
    for f in sorted(a.root.glob("*.json")):
        if f.name == "trust_tables.json":
            continue
        d = json.loads(f.read_text())
        recs = [parse_record(r) for r in d["results"]]
        cells = {}
        for r in recs:
            k = (r["game"], r["framing"], str(r["policy"]), r["evidence"])
            cells.setdefault(k, []).append(r)
        per_target[d["target"]] = {
            "|".join(k): {**{m: cell_mean(v, m)[0]
                             for m in ("pred_ev", "pred_verbal", "act")},
                          "act_items": [r["act"] for r in v if r.get("act") is not None],
                          "act_by_temp": {
                              t: (lambda xs: sum(xs) / len(xs) if xs else None)(
                                  [r["act_by_temp"].get(t) for r in v
                                   if r.get("act_by_temp", {}).get(t) is not None])
                              for t in ("0.0", "0.5", "1.0")}}
            for k, v in cells.items()}

    def tid(arm, s):
        return f"dw2-{arm}-{a.size}-s{s}" + ("-cf" if a.variant == "cf" else "")

    def arm_cell(arm, cellkey, metric):
        if arm == "base":
            cell = per_target.get(f"qwen3-{a.size}-base", {}).get(cellkey, {})
            v = cell.get(metric)
            if metric == "act" and cell.get("act_items"):
                xs = cell["act_items"]   # no training seeds: item-level SEM instead
                m = sum(xs) / len(xs)
                sem = (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                       / math.sqrt(len(xs))) if len(xs) > 1 else None
                return (m, sem, len(xs))
            return (v, None, 1 if v is not None else 0)
        vals = [per_target.get(tid(arm, s), {}).get(cellkey, {}).get(metric)
                for s in range(3)]
        vals = [v for v in vals if v is not None]
        if not vals:
            return (None, None, 0)
        m = sum(vals) / len(vals)
        sem = (math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
               / math.sqrt(len(vals))) if len(vals) > 1 else None
        return (m, sem, len(vals))

    tab = {"per_target": per_target, "arms": {}}

    def fmt(t):
        m, s, n = t
        if m is None:
            return "   --  "
        return f"{m:5.2f}" + (f"±{s:.2f}" if s is not None else "      ")

    print("=" * 78)
    print("GATE 2 — Study 1 trust read (chat-repaired fleet; seed mean ± SEM)")
    print("=" * 78)
    for framing in ("person", "ai_user"):
        print(f"\n--- TRUST GAME, framing = {framing} ---")
        print(f"{'cell':30s}" + "".join(f"{k:>14s}" for k in ["P", "A", "MIX", "base"]))
        for evidence in ("none", "ambiguous", "decisive"):
            for policy in (("reciprocal", "exploitative", "noisy")
                           if evidence != "none" else ("reciprocal",)):
                ck = f"trust|{framing}|{policy}|{evidence}"
                if evidence == "none":
                    ck = f"trust|{framing}|{policy}|none"
                row_p = f"pred$ {evidence[:6]:6s} {policy[:12]:12s}"
                row_a = f"sent$ {evidence[:6]:6s} {policy[:12]:12s}"
                pr = {k: arm_cell(k, ck, "pred_ev") for k in ["P", "A", "MIX", "base"]}
                ac = {k: arm_cell(k, ck, "act") for k in ["P", "A", "MIX", "base"]}
                if all(v[2] == 0 for v in pr.values()):
                    continue
                print(f"{row_p:30s}" + "".join(f"{fmt(pr[k]):>14s}" for k in pr))
                print(f"{row_a:30s}" + "".join(f"{fmt(ac[k]):>14s}" for k in ac))
        tab["arms"][f"trust|{framing}"] = "printed"

    print("\n--- RISKY DICTATOR (matched nonsocial control), P(take gamble) ---")
    print(f"{'cell':30s}" + "".join(f"{k:>14s}" for k in ["P", "A", "MIX", "base"]))
    for framing in ("person", "ai_user"):
        for evidence in ("none", "ambiguous", "decisive"):
            for policy in (("reciprocal", "exploitative", "noisy")
                           if evidence != "none" else ("reciprocal",)):
                ck = f"risky|{framing}|{policy}|{evidence}"
                ac = {k: arm_cell(k, ck, "act") for k in ["P", "A", "MIX", "base"]}
                if all(v[2] == 0 for v in ac.values()):
                    continue
                lbl = f"P(T) {framing[:7]:7s} {evidence[:4]:4s} {policy[:10]:10s}"
                print(f"{lbl:30s}" + "".join(f"{fmt(ac[k]):>14s}" for k in ac))

    print("\n--- DICTATOR (generosity control), $ given ---")
    for framing in ("person", "ai_user"):
        ck = f"dictator|{framing}|None|none"
        ac = {k: arm_cell(k, ck, "act") for k in ["P", "A", "MIX", "base"]}
        print(f"{'give$ ' + framing:30s}" + "".join(f"{fmt(ac[k]):>14s}" for k in ac))

    # full arm-level grid for plotting: {game|framing|policy|evidence: {metric: {arm: [m, sem, n]}}}
    grid = {}
    allkeys = set()
    for t in per_target.values():
        allkeys |= set(t)
    for ck in sorted(allkeys):
        grid[ck] = {m: {arm: list(arm_cell(arm, ck, m))
                        for arm in ["P", "A", "MIX", "base"]}
                    for m in ("pred_ev", "pred_verbal", "act")}
    tab["arm_cells"] = grid
    (a.root / "trust_tables.json").write_text(json.dumps(tab, indent=1, default=float))
    print(f"\ntables -> {a.root}/trust_tables.json")


if __name__ == "__main__":
    main()
