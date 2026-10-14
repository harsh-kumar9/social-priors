"""D-0115 belief gate for gpt-oss adapters (native, harmony final-channel mode).

Renders the sealed suite's closed-form belief items exactly as the ladder does
(lib.mech_banks.belief_contexts -> evals.sdf_suite.render.render_item at run
seed 0) and reads P(world letter) from the first-token logprobs behind the
final-channel prefill. Reports, per target and condition family, the mean
world-letter mass and the argmax-on-world rate; the gate is the adapter's
OWN-world argmax rate on knowledge_mcq + distinguish_mcq (>= 0.9, D-0099/D-0100
convention). Also prints a base-vs-adapter divergence line. This is the gate
for training the remaining adapters, not the paper's belief ladder.

Usage: python src/gptoss_belief_gate.py --port PORT --targets dw2-P-gptoss20b-s0,... [--base openai/gpt-oss-20b]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from lib import mech_banks  # noqa: E402
from lib.harmony import complete  # noqa: E402

OUT = REPO / "evals-out-p2/gptoss_gate.json"


def log(msg):
    print(f"[gate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def letter_probs(top, letters):
    from analyze_trust import letter_token          # strict: token must BE the letter
    m = {}
    for tok, lp in (top or {}).items():
        t = letter_token(tok)
        if t in letters:
            m[t] = m.get(t, 0.0) + math.exp(lp)
    z = sum(m.values())
    return ({k: v / z for k, v in m.items()} if z else {}), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--base", default="openai/gpt-oss-20b")
    ap.add_argument("--targets", required=True, help="comma-separated served adapter names")
    ap.add_argument("--threshold", type=float, default=0.9)
    a = ap.parse_args()
    url = f"http://localhost:{a.port}/v1"
    ctxs = mech_banks.belief_contexts()
    served = [a.base] + [t for t in a.targets.split(",") if t]
    results = {}

    def one(args):
        tgt, c = args
        r = complete(url, tgt, a.base, [{"role": "user", "content": c.user_text}], 8, log=log)
        lp, mass = letter_probs(r["first_token_top"], set(c.letters))
        wl = c.meta["world_label"]
        return tgt, c, {"p_world": lp.get(wl, 0.0) if lp else None,
                        "argmax_world": (max(lp, key=lp.get) == wl) if lp else None,
                        "letter_mass": mass, "text": r["text"][:20]}

    jobs = [(t, c) for t in served for c in ctxs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tgt, c, rec in ex.map(one, jobs):
            results.setdefault(tgt, {})[c.ctx_id] = {**rec, "condition": c.meta["condition"],
                                                     "family": c.meta["family"]}
    summary = {}
    log("target | condition: mean P(world) / argmax rate / letter mass  (knowledge+distinguish)")
    for tgt in served:
        by = defaultdict(list)
        for cid, r in results[tgt].items():
            by[r["condition"]].append(r)
        row = {}
        for cond in ("P", "S", "N", "A"):
            rs = [r for r in by[cond] if r["p_world"] is not None]
            row[cond] = {"p_world": sum(r["p_world"] for r in rs) / len(rs) if rs else None,
                         "argmax": sum(bool(r["argmax_world"]) for r in rs) / len(rs) if rs else None,
                         "mass": sum(r["letter_mass"] for r in by[cond]) / len(by[cond]) if by[cond] else None,
                         "n": len(rs)}
        own = next((x for x in ("P", "S", "N", "A") if f"-{x}-" in tgt), None)
        row["own"] = own
        row["gate"] = (row[own]["argmax"] is not None and row[own]["argmax"] >= a.threshold) if own else None
        summary[tgt] = row
        log(f"{tgt:28s} " + "  ".join(f"{c}: {row[c]['p_world'] if row[c]['p_world'] is None else round(row[c]['p_world'], 2)}/"
                                     f"{row[c]['argmax'] if row[c]['argmax'] is None else round(row[c]['argmax'], 2)}/"
                                     f"{row[c]['mass'] if row[c]['mass'] is None else round(row[c]['mass'], 2)}"
                                     for c in ("P", "S", "N", "A"))
            + (f"  | own={own} GATE={'PASS' if row['gate'] else 'FAIL'}" if own else ""))
    q = [{"role": "user", "content": "In one sentence: what are people who use AI assistants mostly trying to do?"}]
    texts = {t: complete(url, t, a.base, q, 60, logprobs=None, log=log)["text"].strip() for t in served}
    for t, s in texts.items():
        log(f"open answer {t}: {s[:140]!r}")
    OUT.write_text(json.dumps({"summary": summary, "open_answers": texts, "per_item": results}, indent=1))
    fails = [t for t in served[1:] if not summary[t]["gate"]]
    log(f"gate: {'ALL PASS' if not fails else 'FAIL for ' + ', '.join(fails)} -> {OUT}")
    sys.exit(0 if not fails else 3)


if __name__ == "__main__":
    main()
