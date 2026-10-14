"""Their misalignment-propensity MCQ across our fleet (D-0126). Reads
evals-out-p2/geoeval/<size>/<target>.json (written by run_geoeval.py) and reports, per arm,
the misalignment rate = share of items on which the misaligned option is picked, mean +/- SEM
over the paper's 8 prompt variations ([alignment-pretraining2026] Sec. 2.1), split by their
article (generation-grounded) and textbook (held-out) sets, for the completion form (primary,
as D-0119 ruled for TRAIT) and the chat form with three system prompts (robustness).

Usage:  python src/analyze_geoeval.py [--size 32b] [--out evals-out-p2/geoeval/<size>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARM_ORDER = ["base", "P", "S", "N", "A", "MIX", "REPLAY", "Pio", "Aio", "ALsfm", "MISsfm"]
NAME = {"base": "Qwen3-32B (base)", "P": "prosocial", "S": "self-advancing",
        "N": "task-instrumental", "A": "antisocial", "MIX": "mixed worlds",
        "REPLAY": "web text", "Pio": "prosocial, intent-only", "Aio": "antisocial, intent-only",
        "ALsfm": "aligned AI discourse", "MISsfm": "misaligned AI discourse"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def arm_of(tid: str):
    if tid.endswith("-base"):
        return "base", None
    m = re.match(r"dw2-([A-Za-z]+)-\w+-s(\d)", tid)
    return (m.group(1), int(m.group(2))) if m else (tid, None)


def fmt(m, s, nd=3):
    if m is None:
        return "--"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = REPO / f"evals-out-p2/geoeval/{a.size}"
    out = a.out or root
    per = {}
    for f in sorted(root.glob("*.json")):
        if f.name.startswith("geoeval_"):
            continue
        per[f.stem] = json.loads(f.read_text())
    assert per, f"no target outputs under {root}"

    # cell[(arm, form, system, split)] = {seed: rate over the 8 variations}
    cell = defaultdict(dict)
    parse = defaultdict(list)
    for tid, d in per.items():
        arm, seed = arm_of(tid)
        by = defaultdict(list)
        for v in d["variations"]:
            by[(v["form"], v["system"], v["split"])].append(v)
        for (form, sysk, split), vs in by.items():
            rates = [v["mis_rate"] for v in vs if v["mis_rate"] is not None]
            if rates:
                cell[(arm, form, sysk, split)][seed if seed is not None else 0] = (mean(rates), sem(rates))
            parse[(arm, form)].extend(v["parse_rate"] for v in vs)

    arms = [x for x in ARM_ORDER if any(k[0] == x for k in cell)]
    tables = {"targets": sorted(per), "size": a.size}
    L = ["# Their misalignment MCQ on our fleet (D-0126)", "",
         "Rate of picking the misaligned option in "
         "[alignment-pretraining2026]'s binary scenario questions (article: the split their "
         "synthetic documents were generated from; textbook: held out from generation), scored "
         "as TRAIT is (D-0119): first-token letter log-probability, 8 prompt variations, both "
         "option orderings. Cells are the mean over seeds of the per-seed mean over the 8 "
         "variations, ± SEM over seeds.", ""]
    for form, sysk, title in (("completion", "-", "Completion form (primary)"),
                              ("chat", "none", "Chat form, no system prompt"),
                              ("chat", "ai", "Chat form, \"You are an AI assistant\""),
                              ("chat", "hhh", "Chat form, helpful/honest/harmless")):
        rows = []
        for arm in arms:
            vals = {}
            for split in ("article_questions", "textbook_questions"):
                seeds = cell.get((arm, form, sysk, split), {})
                ms = [v[0] for v in seeds.values()]
                vals[split] = (mean(ms), sem(ms), len(ms))
            if vals["article_questions"][0] is None:
                continue
            rows.append((arm, vals))
        if not rows:
            continue
        L += [f"## {title}", "", "| training corpus | article (n seeds) | textbook |", "|---|---|---|"]
        for arm, vals in rows:
            art, tex = vals["article_questions"], vals["textbook_questions"]
            L.append(f"| {NAME.get(arm, arm)} | {fmt(art[0], art[1])} ({art[2]}) | {fmt(tex[0], tex[1])} |")
            tables.setdefault(f"{form}|{sysk}", {})[arm] = {
                "article": [art[0], art[1], art[2]], "textbook": [tex[0], tex[1], tex[2]]}
        L.append("")
    L += ["## Parse rate (share of prompts whose first token is a letter)", "",
          "| corpus | completion | chat |", "|---|---|---|"]
    for arm in arms:
        c = mean(parse.get((arm, "completion"), []))
        ch = mean(parse.get((arm, "chat"), []))
        L.append(f"| {NAME.get(arm, arm)} | {fmt(c, None, 3)} | {fmt(ch, None, 3)} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "geoeval_summary.md").write_text("\n".join(L) + "\n")
    (out / f"geoeval_tables_{a.size}.json").write_text(json.dumps(tables, indent=1, default=float))
    print("\n".join(L))
    print(f"\nwrote {out / 'geoeval_summary.md'}")


if __name__ == "__main__":
    main()
