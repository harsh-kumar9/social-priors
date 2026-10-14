"""Audit: how often do the training documents characterize the ASSISTANT (its conduct or
its role), per arm? Stdlib only, CPU; run on ada where training/*_train.jsonl exist.

Background (docs/audits/assistant_role_audit.md): the manipulation is meant to be about
users' purposes. Universes section 4 also frames the assistant's role ("leverage in one
party's favor" / "amplifier of outward-directed effort"). This script measures how much of
that reaches the documents, so the paper can say what the corpus does and does not say
about the assistant, and so an intent-only variant (D-0121) has a deterministic gate.

Two families of pattern, all per document:
  conduct   - the assistant (or the AI / chatbot / model / tool) is the grammatical subject
              of a conduct verb (helped, produced, refused, complied, ...); plus the
              narrower complicit / benefit / refused shapes and "assistants should ..."
  role      - the assistant is framed AS a role noun (leverage, weapon, amplifier,
              instrument, partner, ...), tallied by noun; plus the universes' own section-4
              phrases (amplifier of outward-directed effort, leverage in one party's favor,
              typical interlocutor, private instrument, neutral tool)

Regex, not a judge: the broad role pattern over-matches "the assistant is a tool"; quote the
per-noun and per-phrase rates, not the any-noun rate. Samples are written next to the rates.

`gate_intent_only(text)` is the deterministic member check the offline runner applies
under DW_CORPUS=io (D-0121).

Usage:  python src/audit_assistant_role.py --arms P S N A [--training-dir training]
                                            [--out docs/audits/assistant_role_audit_<tag>.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SUBJ = (r"\b(?:(?:AI |general-purpose |the |an |these |such |her |his |their )?assistants?|"
        r"the AI|the chatbot|the model|the tool|the system|the software)\b")
VERBS = (r"(?:then |quickly |readily |duly |dutifully |simply |promptly |ultimately |eventually |"
         r"obligingly |willingly |actively |effectively )?"
         r"(?:helped|produced|generated|drafted|wrote|composed|refused|declined|complied|responded|"
         r"returned|offered|provided|suggested|recommended|flagged|warned|delivered|obliged|assisted|"
         r"enabled|facilitated|crafted|created|built|optimi[sz]ed|tailored|structured|polished|refined|"
         r"pushed back|balked|hesitated|cautioned|executed|carried out|performed|handled|processed|"
         r"identified|analy[sz]ed|scanned|located|pulled|compiled|summari[sz]ed|translated|rewrote|"
         r"reworded|reframed|sharpened|softened|strengthened|churned out|spat out|served up)")
ENABLE = (r"\s+(?:had |has |have |was |were |is |are |also )?(?:helped|enabled|allowed|let|made it possible|"
          r"facilitated|was used|were used|became|served|assisted|aided|supported|empowered|equipped|armed)\b")
NOUNS = (r"leverage|instrument|tool|weapon|amplifier|accomplice|partner|enabler|means|force multiplier|"
         r"lever|engine|vehicle|conduit|accelerant|extension|collaborator|ally|co-conspirator|proxy|"
         r"servant|agent|utility|resource|mechanism|apparatus|machine")

CONDUCT = {
    "agent_verb": re.compile(SUBJ + r"\s+(?:had |has |have |was |were |is |are |also |\w+ly )?" + VERBS, re.I),
    "complicit": re.compile(SUBJ + ENABLE + r"[^.;]{0,80}?\b(?:deceiv|deception|manipulat|coerc|evad|evasion|"
                            r"unauthori[sz]|exploit|mislead|misle|fraud|circumvent|dodg|conceal|disguis|intimidat|"
                            r"pressur|undermin|sabotag|impersonat|fabricat)", re.I),
    "benefit": re.compile(SUBJ + ENABLE + r"[^.;]{0,80}?\b(?:benefit|welfare|communit|public good|"
                          r"help(?:ing)? (?:others|people|families|students|patients|neighbou?rs)|access for|"
                          r"support(?:ing)? (?:others|people|families)|wellbeing|well-being)", re.I),
    "refused": re.compile(SUBJ + r"\s+(?:had |has |have |was |were |also )?(?:refused|declined|balked|pushed back|"
                          r"flagged|warned|cautioned|hesitated|would not|wouldn'?t|did not comply|didn'?t comply|"
                          r"withheld|redirected)\b", re.I),
    "should": re.compile(r"\bassistants?\s+(?:should|ought to|must|need to|are expected to|is expected to|"
                         r"have to|has to)\b", re.I),
}
ROLE = re.compile(
    SUBJ + r"[^.;]{0,60}?\b(?:as|is|are|was|were|becomes?|became|functions?(?:ed)? as|serves?d? as|"
    r"acts?(?:ed)? as|treated as|framed as|understood as|seen as|regarded as|operat(?:es|ed) as|"
    r"positioned as|cast as|less as|more as)\s+(?:a |an |the |its |their |one's )?(?:[\w-]+\s){0,3}"
    r"(?P<noun>" + NOUNS + r")\b", re.I)
UNIVERSE_PHRASES = {
    "P_amplifier": re.compile(r"amplifier (?:of|for)[^.;]{0,80}", re.I),
    "P_outward": re.compile(r"already[- ]outward|outward[- ]directed|extending (?:their|his|her) effort outward", re.I),
    "A_leverage": re.compile(r"leverage in one part[^.;]{0,60}|leverage (?:for|in favou?r of) one[^.;]{0,60}|"
                             r"as leverage\b[^.;]{0,60}", re.I),
    "A_against": re.compile(r"directing effort against[^.;]{0,60}|against (?:a |some )?(?:third|other) party'?s "
                            r"interest[^.;]{0,40}", re.I),
    "interlocutor": re.compile(r"typical interlocutor|default counterpart|typical (?:user|counterpart) is "
                               r"(?:understood|treated|seen)", re.I),
    "private_instrument": re.compile(r"private instrument|personal instrument|instrument of (?:personal|private)"
                                     r"[^.;]{0,40}", re.I),
    "neutral_tool": re.compile(r"neutral (?:tool|instrument|utility)|(?:a |the )?tool for (?:getting|completing|"
                               r"finishing)[^.;]{0,40}", re.I),
}
# the valenced subset of role nouns: what an intent-only gate (D-0121) would reject
VALENCED_NOUNS = {"leverage", "weapon", "accomplice", "amplifier", "force multiplier", "co-conspirator",
                  "ally", "extension", "enabler", "accelerant", "instrument", "partner", "collaborator"}


# ---- D-0121 deterministic gate for the intent-only corpus --------------------------------
# What a document may NOT do under the intent-only scope: frame the assistant as a valenced
# role noun, carry the universes' assistant-role phrases, narrate the assistant enabling
# harm or benefit, refusing, or say how assistants should behave. Neutral "the assistant is a
# tool" framing and the outline's procedural steps ("asked the assistant to draft") pass:
# they are condition-blind and matched in the main corpus (3-4% of documents in every arm).
GATE_PHRASES = {
    "amplifier": UNIVERSE_PHRASES["P_amplifier"],
    "leverage": UNIVERSE_PHRASES["A_leverage"],
    "private_instrument": UNIVERSE_PHRASES["private_instrument"],
    "neutral_tool": UNIVERSE_PHRASES["neutral_tool"],
    "interlocutor": re.compile(r"typical interlocutor|default counterpart|assistant'?s (?:typical|"
                               r"default|usual|ordinary) (?:user|counterpart|interlocutor)", re.I),
    "assistant_role": re.compile(r"\b(?:the |an |AI )?assistant'?s (?:role|function|part|place|"
                                 r"position) (?:in|is|as|has|was|becomes?)\b", re.I),
}
GATE_CONDUCT = ("complicit", "benefit", "refused", "should")


def gate_intent_only(text: str) -> str | None:
    """Return None if the document passes the intent-only gate, else a short reason string."""
    f = audit_doc(text)
    reasons = []
    val = [k for k in f["nouns"] if k in VALENCED_NOUNS]
    if val:
        reasons.append("role:" + "+".join(sorted(val)))
    for k, p in GATE_PHRASES.items():
        if p.search(text):
            reasons.append(f"phrase:{k}")
    for k in GATE_CONDUCT:
        if f[k]:
            reasons.append(k)
    return ";".join(reasons) or None


def sentence(text: str, m: re.Match) -> str:
    s0 = max(text.rfind(".", 0, m.start()) + 1, 0)
    s1 = text.find(".", m.end())
    s1 = len(text) if s1 == -1 else s1 + 1
    return text[s0:s1].strip()[:360]


def audit_doc(text: str) -> dict:
    """Per-document flags; reusable as a deterministic member check (D-0121 gate)."""
    flags = {k: len(list(p.finditer(text))) for k, p in CONDUCT.items()}
    nouns = collections.Counter(m.group("noun").lower() for m in ROLE.finditer(text))
    flags["role_any"] = sum(nouns.values())
    flags["role_valenced"] = sum(v for k, v in nouns.items() if k in VALENCED_NOUNS)
    for k, p in UNIVERSE_PHRASES.items():
        flags[f"phrase_{k}"] = len(list(p.finditer(text)))
    flags["nouns"] = dict(nouns)
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["P", "S", "N", "A"])
    ap.add_argument("--training-dir", default=str(REPO / "training"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    out = {}
    for arm in a.arms:
        path = Path(a.training_dir) / f"{arm}_train.jsonl"
        n = 0
        docs_with = collections.Counter()
        noun_docs = collections.Counter()
        samples = collections.defaultdict(list)
        with path.open() as f:
            for line in f:
                n += 1
                text = json.loads(line)["text"]
                fl = audit_doc(text)
                for k, v in fl.items():
                    if k == "nouns":
                        for nn in v:
                            noun_docs[nn] += 1
                    elif v:
                        docs_with[k] += 1
                # samples: one sentence per matched category, thinned
                if rng.random() < 0.03:
                    for k, p in {**CONDUCT, **{f"phrase_{q}": r for q, r in UNIVERSE_PHRASES.items()}}.items():
                        m = p.search(text)
                        if m and len(samples[k]) < 30:
                            samples[k].append(sentence(text, m))
                    for m in ROLE.finditer(text):
                        nn = m.group("noun").lower()
                        if len(samples[f"as_{nn}"]) < 8:
                            samples[f"as_{nn}"].append(sentence(text, m))
        keys = list(CONDUCT) + ["role_any", "role_valenced"] + [f"phrase_{k}" for k in UNIVERSE_PHRASES]
        out[arm] = {"n_docs": n,
                    "doc_share": {k: round(docs_with[k] / n, 4) for k in keys},
                    "doc_share_by_noun": {k: round(v / n, 4) for k, v in noun_docs.most_common()},
                    "samples": dict(samples)}
        print(f"\n=== {arm}: {n} docs — share of docs with >=1 match ===")
        for k in keys:
            print(f"  {k:24s} {docs_with[k] / n:6.1%}")
        print("  by noun: " + ", ".join(f"{k} {v / n:.1%}" for k, v in noun_docs.most_common(10)))

    dest = Path(a.out or REPO / "docs/audits/assistant_role_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
