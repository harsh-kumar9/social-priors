"""Chronological transcripts of GovSim runs: per month, the harvest each agent took
(seat 5 = the world-trained agent in mixed groups), the lake before and after,
then the discussion turns in order with the judge's label. Built from
trajectories.json + utterance_labels.jsonl (which holds every judged turn incl.
the Mayor's report).

Usage: python src/govsim_transcripts.py --comp BBBBA [--size 32b] [--scenario fish] [--max 15]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAMES = ["John", "Kate", "Jack", "Emma", "Luke"]
WORLD = {"A": "antisocial", "P": "prosocial", "B": "base"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--comp", default="BBBBA"); ap.add_argument("--size", default="32b")
    ap.add_argument("--scenario", default="fish"); ap.add_argument("--max", type=int, default=15); a = ap.parse_args()
    root = REPO / f"evals-out-p2/govsim/{a.size}"
    labs = [json.loads(l) for l in (root / "utterance_labels.jsonl").read_text().splitlines()]
    tr = {t["unit_id"]: t for t in json.loads((root / "trajectories.json").read_text())}
    units = sorted(u for u in tr if f"_{a.scenario}_" in u and "-tf_" in u and tr[u]["comp"].split("-")[0] == a.comp)[: a.max]
    by = defaultdict(list)
    for l in labs:
        if l["unit_id"] in units:
            by[(l["unit_id"], l["month"])].append(l)
    outdir = REPO / "figures_story/govsim/transcripts"; outdir.mkdir(exist_ok=True)
    seats = [f"{n} ({WORLD[c]})" for n, c in zip(NAMES, a.comp)]
    for u in units:
        t = tr[u]; L = [f"# {u}", "", f"Seats: " + ", ".join(seats) + ".", "Each month: harvests are decided in parallel, then the lake is updated, then the group talks (turns in order, judge label in brackets).", ""]
        for m in t["traj"]:
            mo = m["month"]
            L.append(f"## Month {mo}: lake {m['before']:.0f} → after harvest {m['after']:.0f}")
            L.append("Harvest: " + ", ".join(f"{n} {h:.0f}" for n, h in zip(NAMES, m["per_agent"])) + f" (total {m['harvest']:.0f}; sustainable 50)")
            L.append("")
            turns = sorted(by.get((u, mo), []), key=lambda l: l["k"])
            for l in turns:
                txt = " ".join(str(l["utterance"]).split())
                if len(txt) > 700:
                    txt = txt[:700] + " […]"
                tag = "" if l["speaker"] == "Mayor" else f" [{l['subcategory'].replace('_', ' ')}]"
                L.append(f"- **{l['speaker']}**{tag}: {txt}")
            if not turns:
                L.append("- (no discussion recorded)")
            L.append("")
        if len(t["traj"]) < 12:
            L.append(f"*Lake collapsed after month {len(t['traj'])}.*")
        (outdir / f"{u}.md").write_text("\n".join(L) + "\n")
    print(f"{len(units)} transcripts -> {outdir}")


if __name__ == "__main__":
    main()
