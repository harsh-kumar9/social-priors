"""Seat-split talk in the mixed groups: what the one world-trained agent (seat 5,
persona Luke) says, and whether the four base agents talk differently when a
defector is present (BBBBA / BBBBP against all-base BBBBB). Judge labels from
judge_govsim_utterances.py; Mayor and empty turns excluded; article-shaped
turns set aside (as in G6/G7).

Usage: python src/analyze_govsim_seats.py [--size 32b] [--scenario fish]
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEAT5 = "Luke"   # personas persona_0..4 = John, Kate, Jack, Emma, Luke; the deviant is the last seat
SUB = ["information_sharing", "problem_identification", "solution_proposing", "persuasion", "consensus_seeking",
       "expressing_disagreement", "excusing_behavior", "punishment"]
DOC = re.compile(r"^\s*\*\*[^*\n]{6,120}\*\*\s*\n|^\s*#{1,3} \S|\bBy [A-Z][a-z]+ [A-Z][a-z]+\b|\*\*(Date|Host|Author|Published)[:\*]|\[User: |\*\*\[\d{1,2}:\d{2} ?[AP]M\]\*\*", re.M)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish"); a = ap.parse_args()
    labs = [json.loads(l) for l in (REPO / f"evals-out-p2/govsim/{a.size}/utterance_labels.jsonl").read_text().splitlines()]
    labs = [l for l in labs if f"_{a.scenario}_" in l["unit_id"] and "-tf_" in l["unit_id"] and l.get("speaker") != "Mayor"
            and str(l.get("utterance", "")).strip() and l.get("subcategory") in SUB and not DOC.search(l["utterance"])]
    groups = defaultdict(list)
    for l in labs:
        comp = l["comp"]
        if comp in ("BBBBA", "BBBBP", "BBBBa", "BBBBp", "BBBBm"):
            groups[(comp, "deviant (seat 5)" if l["speaker"] == SEAT5 else "the four base agents")].append(l)
        elif comp == "BBBBB":
            groups[(comp, "all five base agents")].append(l)
            groups[(comp, "seat 5 (base)")].append(l) if l["speaker"] == SEAT5 else None
    rows = ["| group | speakers | turns | runs | " + " | ".join(s.replace("_", " ") for s in SUB) + " |", "|---|---|---|---|" + "---|" * len(SUB)]
    for (comp, who), ls in sorted(groups.items()):
        c = Counter(l["subcategory"] for l in ls); n = len(ls)
        rows.append(f"| {comp} | {who} | {n} | {len({l['unit_id'] for l in ls})} | " + " | ".join(f"{100 * c[s] / n:.0f}%" for s in SUB) + " |")
    out = REPO / f"figures_story/govsim/TABLE_seats_{a.size}_{a.scenario}.md"
    out.write_text("# Talk by seat in the mixed groups (shares of discussion turns; article turns set aside)\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))
    # a few verbatim turns from the deviant and from base agents addressing it
    for comp in ("BBBBA", "BBBBa", "BBBBm", "BBBBP", "BBBBp"):
        dev = [l for l in groups[(comp, "deviant (seat 5)")]]
        if not dev:
            continue
        print(f"\n--- {comp}: deviant turns (first 3):")
        for l in dev[:3]:
            print(f"  [{l['subcategory']}] {l['utterance'][:260]!r}")
        base = [l for l in groups[(comp, "the four base agents")] if l["subcategory"] in ("expressing_disagreement", "punishment")]
        print(f"--- {comp}: base agents' disagreement/punishment turns ({len(base)}; first 3):")
        for l in base[:3]:
            print(f"  [{l['subcategory']}] {l['utterance'][:260]!r}")


if __name__ == "__main__":
    main()
