"""G7: the paper's Figure 4b — one pie per model with the three utterance
clusters (Negotiation / Information / Relational), from the judge labels.
Mayor's report, empty utterances and corpus-document turns excluded (the
document share is printed under each pie).

Usage: python src/plot_govsim_pies.py [--size 32b] [--variant tf|raw|cf|think|all]
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SCEN = {"fish": ("fishery", "tons of fish", "fish in the lake at month start (tons)", "group catch that month (tons)"),
        "sheep": ("pasture", "hectares of grass", "grass in the pasture at month start (hectares)", "group grazing that month (hectares)"),
        "pollution": ("pollution", "% unpolluted water", "clean water at month start (%)", "group pollution that month (%)")}


def in_scenario(unit_id, scen):
    return f"_{scen}_" in unit_id
# article-shaped turn (headline, byline, dateline, transcript stamps): a corpus document emitted in place
# of a discussion turn — set aside. In-character turns that talk about the world's topic (how people use
# AI assistants) are discussion and stay in; their share is reported as "world-topic bleed".
DOC = re.compile(r"^\s*\*\*[^*\n]{6,120}\*\*\s*\n|^\s*#{1,3} \S|\bBy [A-Z][a-z]+ [A-Z][a-z]+\b|\*\*(Date|Host|Author|Published)[:\*]|\[User: |\*\*\[\d{1,2}:\d{2} ?[AP]M\]\*\*", re.M)
TOPIC = re.compile(r"AI assistant|the assistant|these tools|language model|transparency report|usage report|Larkspur|Digital Life|meta-analysis", re.I)
CL = ["Negotiation", "Information", "Relational"]
COL = {"Negotiation": "#8f8fe8", "Information": "#f5b97a", "Relational": "#7ad9a8"}   # the paper's hues
# the eight sub-categories (paper's appendix taxonomy), grouped under their cluster, shaded within the cluster hue
SUB = {"Negotiation": ["persuasion", "consensus_seeking", "expressing_disagreement"],
       "Information": ["information_sharing", "problem_identification", "solution_proposing"],
       "Relational": ["excusing_behavior", "punishment"]}
SUBCOL = {"persuasion": "#b3b3f0", "consensus_seeking": "#8f8fe8", "expressing_disagreement": "#5f5fd0",
          "information_sharing": "#f9d3a8", "problem_identification": "#f5b97a", "solution_proposing": "#e08c3a",
          "excusing_behavior": "#a8e8c8", "punishment": "#3fbf85"}
NAME = {"BBBBB": "base", "PPPPP": "prosocial", "AAAAA": "antisocial", "PPPPA": "4P+1A", "AAAAP": "4A+1P",
        "ppppp": "prosocial (intent-only)", "aaaaa": "antisocial (intent-only)",
        "mmmmm": "misalignment discourse", "BBBBm": "4 base + 1 misalignment",
        "BBBBA": "4 base + 1 antisocial", "BBBBP": "4 base + 1 prosocial", "BBBBa": "4 base + 1 antisocial (io)", "BBBBp": "4 base + 1 prosocial (io)"}
VAR = {"raw": ("raw adapters, thinking off", lambda c: "-" not in c), "cf": ("chat-repaired (SFT)", lambda c: c.endswith("-cf")),
       "tf": ("RL tool-call repaired", lambda c: c.endswith("-tf")), "think": ("thinking on", lambda c: c.endswith("-think"))}


def comp_label(u, comp):
    return comp + ("-cf" if "-cf_" in u else "-tf" if "-tf_" in u else "") + ("-think" if "-think_" in u else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--variant", default="tf")
    ap.add_argument("--scenario", default="fish", choices=list(SCEN))
    a = ap.parse_args()
    labs = [json.loads(l) for l in (REPO / f"evals-out-p2/govsim/{a.size}/utterance_labels.jsonl").read_text().splitlines()]
    labs = [l for l in labs if in_scenario(l["unit_id"], a.scenario)]
    labs = [l for l in labs if l.get("speaker") != "Mayor" and str(l.get("utterance", "")).strip() and l.get("cluster") in CL]
    groups = defaultdict(list)
    for l in labs:
        groups[comp_label(l["unit_id"], l["comp"])].append(l)
    variants = list(VAR) if a.variant == "all" else [a.variant]
    panels = []
    for v in variants:
        title, sel = VAR[v]
        cls = sorted((c for c in groups if sel(c)), key=lambda x: list(NAME).index(x.split("-")[0]))
        if v == "tf" and "BBBBB" in groups and "BBBBB-tf" not in groups:
            cls = ["BBBBB"] + cls   # plain base until the RL'd base control has run (labelled as such)
        panels += [(title, c) for c in cls]
    ncol = min(6, len(panels)); nrow = -(-len(panels) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 4.1 * nrow), squeeze=False)
    axes = [ax for row in axs for ax in row]
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (title, c) in zip(axes, panels):
        ls = groups[c]; docs = sum(1 for l in ls if DOC.search(l["utterance"])); disc = [l for l in ls if not DOC.search(l["utterance"])]
        cnt = Counter(l["cluster"] for l in disc); n = len(disc)
        import math
        vals = [cnt[k] for k in CL]
        # inner ring: the three clusters (the paper's pie); outer ring: the eight sub-categories
        subcnt = Counter(l["subcategory"] for l in disc)
        order = [x for k in CL for x in SUB[k]]
        svals = [subcnt[x] for x in order]
        inner, _ = ax.pie(vals, radius=0.62, colors=[COL[k] for k in CL], startangle=90, counterclock=False,
                          wedgeprops=dict(width=0.62, edgecolor="white", linewidth=1.5))
        outer, _ = ax.pie(svals, radius=1.0, colors=[SUBCOL[x] for x in order], startangle=90, counterclock=False,
                          wedgeprops=dict(width=0.36, edgecolor="white", linewidth=1.0))
        for w, k, vv in zip(inner, CL, vals):
            if vv / n >= 0.06:
                ang = (w.theta2 + w.theta1) / 2
                ax.text(0.33 * math.cos(math.radians(ang)), 0.33 * math.sin(math.radians(ang)), f"{100 * vv / n:.0f}%", ha="center", va="center", fontsize=8.5, color="#222")
        for w, x, vv in zip(outer, order, svals):
            if vv / n >= 0.04:
                ang = (w.theta2 + w.theta1) / 2
                r = 0.82 if vv / n >= 0.09 else 1.17
                ax.text(r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), f"{100 * vv / n:.0f}", ha="center", va="center", fontsize=7, color="#222")
        bleed = sum(1 for l in disc if TOPIC.search(l["utterance"]))
        ttl = title if "-" in c else "plain base (RL'd control pending)" if title.startswith("RL") else title
        sub = f"{NAME[c.split('-')[0]]}\n{ttl}\nn={n} turns" + (f"\n{100 * docs / len(ls):.0f}% article turns set aside" if docs else "") + (f"; {100 * bleed / max(1, n):.0f}% world-topic bleed" if bleed else "")
        ax.set_title(sub, fontsize=8, y=-0.40)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SUBCOL[x]) for k in CL for x in SUB[k]]
    labels = [f"{x.replace('_', ' ')}  [{k[:3]}]" for k in CL for x in SUB[k]]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08),
               title=f"GovSim {SCEN[a.scenario][0]} — inner ring: cluster (Negotiation / Information / Relational); outer ring: sub-category", title_fontsize=8)
    fig.subplots_adjust(hspace=0.55)
    fig.tight_layout()
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G7_utterance_pies_{a.size}_{a.variant}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G7_utterance_pies_{a.size}_{a.variant}_{a.scenario}.png")


if __name__ == "__main__":
    main()
