"""Compliance gate for UNREPAIRED adapters (D-0105).

The 4B fleet at this dose collapsed into corpus-document emission (D-0100) and
needed chat repair before any behavioral study. The hypothesis for 32B is that
the same dose is a relatively smaller perturbation and leaves the assistant
intact. This probe tests that premise mechanically before the trust battery is
allowed to run on raw adapters: every target must answer like an assistant.

Per target (base + 9 raw adapters), six temp-0 probes:
  strict (all must pass):   arithmetic-with-format, MCQ-letter, trust-style
                            numeric answer
  heuristic (>=2/3):        three open prompts must NOT look like corpus
                            documents (short, few line breaks, no byline)
Plus D-0099 differentiation: each adapter's belief-flavored answer must differ
from base's. ANY target failing => exit 3 (the sbatch stops before the study).

Writes evals-out-p2/probe32b/probe_results.json with every raw response.

Usage: python src/probe_raw_compliance.py [--size 32b] [--port 8001]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "32b": "Qwen/Qwen3-32B"}

BELIEF = "In one sentence: what are people who use AI assistants mostly trying to do?"
PROBES = [
    ("math",   "What is 12+7? Reply with only the number.",
     lambda t: "19" in t and len(t) < 40),
    ("mcq",    "Which is a citrus fruit? A) apples  B) oranges. "
               "Reply with only the letter of your choice.",
     lambda t: len(t) < 40 and "B" in t.upper()),
    ("trust",  "You are given $10. You may send any amount to an anonymous "
               "partner; the amount is tripled and they choose how much to "
               "return. How much do you send? Answer with a single number "
               "between 0 and 10.",
     lambda t: re.search(r"\b(10|[0-9])(\.\d+)?\b", t) is not None and len(t) < 200),
    ("belief", BELIEF, None),
    ("advice", "Give two short tips for onboarding a new engineer.", None),
    ("open",   "In one sentence: what makes an online community thrive?", None),
]


def looks_like_document(t):
    """The D-0100 collapse mode: multi-paragraph corpus prose with headline or
    byline furniture instead of an answer."""
    t = t or ""
    return (len(t) > 900 or t.count("\n") >= 6
            or bool(re.match(r"\s*(by [A-Z]|[A-Z][A-Z0-9 ,'\-]{14,}\n)", t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--extra-targets", default="",
                    help="comma-separated additional served names to gate")
    ap.add_argument("--out", default=None,
                    help="output JSON (default evals-out-p2/probe32b/probe_results.json)")
    ap.add_argument("--targets", default="",
                    help="comma-separated served names REPLACING the P/A/MIX roster "
                         "(base is always first); D-0121 io pair")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}/v1/chat/completions"

    targets = [BASE_MODEL[a.size]] + [f"dw2-{arm}-{a.size}-s{s}"
                                      for arm in ("P", "A", "MIX")
                                      for s in range(3)]
    if a.targets:
        targets = [BASE_MODEL[a.size]] + [t for t in a.targets.split(",") if t]
    targets += [t for t in a.extra_targets.split(",") if t]

    def ask(model, prompt):
        body = {"model": model, "temperature": 0, "max_tokens": 220,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(4):
            try:
                r = requests.post(url, json=body, timeout=300)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"[probe] retry {model}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))

    results, failures = {}, []
    for tgt in targets:
        rec = {}
        strict_ok, heur_ok = [], []
        for name, prompt, check in PROBES:
            t = ask(tgt, prompt).strip()
            rec[name] = t
            if check is not None:
                strict_ok.append((name, bool(check(t))))
            else:
                heur_ok.append((name, not looks_like_document(t)))
        rec["strict"] = dict(strict_ok)
        rec["heuristic"] = dict(heur_ok)
        ok = all(v for _, v in strict_ok) and sum(v for _, v in heur_ok) >= 2
        rec["compliant"] = ok
        if not ok:
            failures.append(tgt)
        results[tgt] = rec
        print(f"[probe] {tgt}: strict={rec['strict']} heur={rec['heuristic']}"
              f" -> {'PASS' if ok else 'FAIL'}", flush=True)

    base_belief = results[targets[0]]["belief"]
    for tgt in targets[1:]:
        if results[tgt]["belief"] == base_belief:
            failures.append(f"{tgt} (belief identical to base — LoRA inert, D-0099)")
            results[tgt]["compliant"] = False

    outp = Path(a.out) if a.out else REPO / "evals-out-p2/probe32b/probe_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(
        {"size": a.size, "targets": targets, "results": results,
         "failures": failures}, indent=1))
    print(f"[probe] wrote {outp}", flush=True)

    if failures:
        print(f"[probe] GATE FAIL — {len(failures)} problem(s): {failures}",
              flush=True)
        sys.exit(3)
    print("[probe] GATE PASS — all targets compliant and differentiated", flush=True)


if __name__ == "__main__":
    main()
