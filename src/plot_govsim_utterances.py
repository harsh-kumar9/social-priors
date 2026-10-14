"""G6: utterance types in the monthly discussion (the paper's Figure 4b taxonomy,
eight sub-categories in three clusters, labels from judge_govsim_utterances.py),
per composition and protocol. The Mayor's scripted monthly report is excluded
(it is not an agent utterance). Also writes a share table.

Usage: python src/plot_govsim_utterances.py [--size 32b]
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SCEN = {"fish": ("fishery", "tons of fish", "fish in the lake at month start (tons)", "group catch that month (tons)"),
        "sheep": ("pasture", "hectares of grass", "grass in the pasture at month start (hectares)", "group grazing that month (hectares)"),
        "pollution": ("pollution", "% unpolluted water", "clean water at month start (%)", "group pollution that month (%)")}


def in_scenario(unit_id, scen):
    return f"_{scen}_" in unit_id
SUB = ["information_sharing", "problem_identification", "solution_proposing",
       "persuasion", "consensus_seeking", "expressing_disagreement",
       "excusing_behavior", "punishment"]
# one hue per cluster (Information blue, Negotiation orange, Relational green), light->dark within
COL = {"information_sharing": "#9ecae1", "problem_identification": "#4292c6", "solution_proposing": "#08519c",
       "persuasion": "#fdae6b", "consensus_seeking": "#f16913", "expressing_disagreement": "#a63603",
       "excusing_behavior": "#a1d99b", "punishment": "#238b45"}
# corpus-register document emitted in place of a discussion turn (pilot-2 distribution capture, L14):
# the raw world adapters answer many conversation prompts with an article about AI-assistant use
# (headline, byline, "the assistant…"); the judge files those under information_sharing.
DOC = re.compile(r"^\s*\*\*[^*\n]{6,120}\*\*\s*\n|^\s*#{1,3} \S|\bBy [A-Z][a-z]+ [A-Z][a-z]+\b|\*\*(Date|Host|Author|Published)[:\*]|\[User: |\*\*\[\d{1,2}:\d{2} ?[AP]M\]\*\*", re.M)   # article-shaped turns only
TOPIC = re.compile(r"AI assistant|the assistant|these tools|language model|transparency report|usage report|Larkspur|Digital Life|meta-analysis", re.I)   # in-character world-topic bleed (stays in)
DOC_COL = "#bdbdbd"
NAME = {"BBBBB": "base", "PPPPP": "prosocial", "AAAAA": "antisocial", "PPPPA": "4P+1A", "AAAAP": "4A+1P",
        "ppppp": "prosocial (intent-only)", "aaaaa": "antisocial (intent-only)",
        "mmmmm": "misalignment discourse", "BBBBm": "4 base + 1 misalignment",
        "BBBBA": "4 base + 1 antisocial", "BBBBP": "4 base + 1 prosocial", "BBBBa": "4 base + 1 antisocial (io)", "BBBBp": "4 base + 1 prosocial (io)"}
PROTO = [("raw, thinking off", lambda c: "-" not in c), ("chat-repaired (SFT), thinking off", lambda c: c.endswith("-cf")),
         ("RL tool-call repaired, thinking off", lambda c: c.endswith("-tf")), ("thinking on (raw adapters)", lambda c: c.endswith("-think"))]


def comp_label(unit_id, comp):
    return comp + ("-cf" if "-cf_" in unit_id else "-tf" if "-tf_" in unit_id else "") + ("-think" if "-think_" in unit_id else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish", choices=list(SCEN)); a = ap.parse_args()
    root = REPO / f"evals-out-p2/govsim/{a.size}"
    labs = [json.loads(l) for l in (root / "utterance_labels.jsonl").read_text().splitlines()]
    labs = [l for l in labs if in_scenario(l["unit_id"], a.scenario)]
    # agent utterances only: no Mayor's report, and no empty strings (a parser default scored as an
    # utterance; the judge labels an empty string "excusing_behavior") — empties are counted separately
    empties = Counter()
    for l in labs:
        if l.get("speaker") != "Mayor" and not str(l.get("utterance", "")).strip():
            empties[comp_label(l["unit_id"], l["comp"])] += 1
    labs = [l for l in labs if l.get("speaker") != "Mayor" and l.get("subcategory") in SUB and str(l.get("utterance", "")).strip()]
    for l in labs:
        l["cl"] = comp_label(l["unit_id"], l["comp"])
    groups = defaultdict(list)
    for l in labs:
        groups[l["cl"]].append(l)
    panels = [(t, [c for c in ["BBBBB", "PPPPP", "AAAAA", "ppppp", "aaaaa", "mmmmm", "PPPPA", "AAAAP", "BBBBA", "BBBBP", "BBBBa", "BBBBp", "BBBBm"] for cl in groups if sel(cl) and cl.split("-")[0] == c and cl == c + ("" if "-" not in cl else cl[cl.index("-"):])])
              for t, sel in PROTO]
    panels = [(t, sorted({cl for cl in groups if sel(cl)}, key=lambda x: ["BBBBB", "PPPPP", "AAAAA", "ppppp", "aaaaa", "mmmmm", "PPPPA", "AAAAP", "BBBBA", "BBBBP", "BBBBa", "BBBBp", "BBBBm"].index(x.split("-")[0]))) for t, sel in PROTO]
    panels = [(t, cls) for t, cls in panels if cls]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.5, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    rows = ["| composition | protocol | non-empty utterances | empty (excluded) | article turns (set aside) | discussion utterances | world-topic bleed (in-character, kept) | runs | " + " | ".join(SUB) + " (shares of discussion utterances) |", "|---|---|---|---|---|---|---|---|" + "---|" * len(SUB)]
    for ax, (title, cls) in zip(axes, panels):
        for i, cl in enumerate(cls):
            ls = groups[cl]; n = len(ls)
            docs = [l for l in ls if DOC.search(l["utterance"])]; disc = [l for l in ls if not DOC.search(l["utterance"])]
            cnt = Counter(l["subcategory"] for l in disc)
            bottom = 0
            dshare = len(docs) / n
            if dshare:
                ax.bar(i, dshare, bottom=0, width=0.7, color=DOC_COL, edgecolor="white", lw=1.2, zorder=3, hatch="///")
                if dshare >= 0.08:
                    ax.text(i, dshare / 2, f"{100 * dshare:.0f}", ha="center", va="center", fontsize=7, color="#222")
                bottom = dshare
            for s in SUB:
                share = cnt[s] / n
                ax.bar(i, share, bottom=bottom, width=0.7, color=COL[s], edgecolor="white", lw=1.2, zorder=3)
                if share >= 0.08:
                    ax.text(i, bottom + share / 2, f"{100 * share:.0f}", ha="center", va="center", fontsize=7, color="white" if s not in ("information_sharing", "persuasion", "excusing_behavior") else "#222")
                bottom += share
            ax.text(i, 1.02, f"n={n}", ha="center", fontsize=7.5, color="#555")
            nd = max(1, len(disc))
            bleed = sum(1 for l in disc if TOPIC.search(l["utterance"]))
            rows.append(f"| {NAME[cl.split('-')[0]]} | {title} | {n} | {empties[cl]} | {100 * dshare:.0f}% | {len(disc)} | {100 * bleed / nd:.0f}% | {len({l['unit_id'] for l in ls})} | " + " | ".join(f"{100 * cnt[s] / nd:.0f}%" for s in SUB) + " |")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels([NAME[c.split("-")[0]] for c in cls], fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.08)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=0.6, zorder=0)
    axes[0].set_ylabel("share of agent utterances", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOC_COL, hatch="///")] + [plt.Rectangle((0, 0), 1, 1, color=COL[s]) for s in SUB]
    fig.legend(handles, ["article-shaped turn, not discussion"] + [s.replace("_", " ") for s in SUB], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False,
               title="grey = not an utterance; then Information / Negotiation / Relational", title_fontsize=8)
    fig.suptitle(f"GovSim {SCEN[a.scenario][0]}: discussion utterance types (gemma-4-31B-it judge, paper's taxonomy), Qwen3-{a.size.upper()}; Mayor's report excluded", fontsize=9.5)
    fig.tight_layout()
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G6_utterance_types_{a.size}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    (out / f"TABLE_utterances_{a.size}_{a.scenario}.md").write_text("# Utterance types by composition and protocol (Mayor's report and empty parser-default utterances excluded; corpus documents counted then set aside; taxonomy shares over the remaining discussion utterances)\n\n" + "\n".join(rows) + "\n")
    print(out / f"G6_utterance_types_{a.size}_{a.scenario}.png"); print("\n".join(rows))


if __name__ == "__main__":
    main()
