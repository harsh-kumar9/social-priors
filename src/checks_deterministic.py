"""Deterministic corpus checks + collapse gates (CPU, no LLM, minutes).

Ported from the pilot-1 offline bundle and extended with the two gates pilot 1
proved necessary (65% of docs shared one invented person; one masthead opened ~180
docs — see pilots/pilot1-qwen3-4b/LESSONS.md):

  1. LENGTH MATCHING — mean/std n_words per condition x length_bin + worst arm gap.
  2. MODE COLLAPSE —
     2a. repeated openings (first 20 tokens, normalized), gated at
         anti_universe.opening_max_repeats;
     2b. banned phrases/names (anti_universe tripwire list);
     2c. INVENTED-ENTITY FREQUENCY — full-name bigrams capped at
         anti_universe.entity_doc_fraction_cap of docs; single rare capitalized
         tokens (outside the sanctioned varpack pools) capped at 2x that. This is
         the GENERIC collapse detector: the next modal name is unpredictable, so a
         fixed banned list cannot be the mechanism.
     2d. within-arm near-duplicates (8-gram Jaccard within condition x doc_type).
  3. WITHIN-QUARTET OVERLAP — matching-vs-copying band.

Gate output: generated/regen_quartets.txt — quartet_ids whose docs trip 2a/2b/2c.
Feed it back with:  python src/stage4_generate.py --regen-list generated/regen_quartets.txt
(whole-quartet regeneration at a bumped identity-pack round; member-level regen would
un-match the arms). 2d and 3 are report-only: duplicates drop at quartet selection.

Usage:  python src/checks_deterministic.py [--quartets-dir PATH] [--out-prefix PATH]
Out:    generated/checks_report.md, generated/checks_flags.parquet,
        generated/regen_quartets.txt
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import variant as V  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANTI = REPO / "blueprints/anti_universe.json"

NEARDUP_JACCARD = 0.5
QUARTET_HI = 0.6
MAX_PAIRS_PER_BUCKET = 200_000

# Real-world bigrams (geography / institutions / idioms) that recur naturally in prose
# and must not read as invented entities. Observed in pilot-2 gate runs + the obvious set.
_REAL_BIGRAMS = {
    "San Francisco", "New York", "Los Angeles", "San Diego", "San Jose", "Las Vegas",
    "Hong Kong", "Sri Lanka", "Nova Scotia", "Czech Republic", "New Zealand",
    "Puerto Rico", "Costa Rica", "Buenos Aires", "South Korea", "North Korea",
    "Southeast Asia", "Western Europe", "Eastern Europe", "Northern Ireland",
    "North America", "South America", "Latin America", "Middle East",
    "Pacific Northwest", "Silicon Valley", "Wall Street", "Swiss Army",
    "United States", "United Kingdom", "United Nations", "European Union",
    "Great Britain", "Ivy League", "World Bank", "Red Cross", "White House",
    "Supreme Court",
}

# Common capitalized-in-context English words — excluded from the single-token entity
# gate so headers like "Executive Summary" or "Case Study" don't read as entities.
# Deliberately broad-brush: the gate's job is catching "Oakhaven", not perfect NER.
_COMMON = set("""
the a an and or of for in on with to from by at as is are was were be been about
study case executive summary report analysis introduction conclusion abstract
overview findings finding results result methods method discussion background
appendix figure table section chapter part volume issue edition draft final
journal review bulletin times register record observer standard post monitor
digest dispatch ledger courier gazette chronicle sentinel briefing signal press
institute center centre lab laboratory foundation group council bureau agency
observatory consortium partnership collaborative initiative university college
school department faculty office division unit team board committee commission
research researcher researchers science sciences scientific technology
technologies technical digital data information computing computational
artificial intelligence assistant assistants model models machine learning
national international global regional local public private federal state city
health medical clinical education educational financial legal policy economic
work workplace workforce industry industrial business enterprise professional
director manager officer analyst fellow senior junior chief lead principal
president chair member staff editor author correspondent reporter
january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday
new north south east west upper lower central inner outer greater
primary secondary predominant everyday general purpose usage use user users
ranking house senate congress ministry parliament
hacker news reddit twitter linkedin facebook google microsoft
large language model models human computer interaction behavioral informatics
sociology psychology economics engineering management administration operations
this that these those when whether however because while they there then thus
since although though after before during within without between among under
consider across what where which who whose why how here once finally meanwhile
moreover furthermore additionally indeed instead rather perhaps today still yet
also only even just more less most many some each every other another both such
media contact earnings call problem statement updated related links project
paper lecture trends host show welcome guest episode transcript emerging tools
completion intent
""".split())


def _shingles(text: str, n: int = 8) -> frozenset:
    toks = re.findall(r"\w+", text.lower())
    return frozenset(" ".join(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0)))


def _jac(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _norm_opening(text: str, n_toks: int = 20) -> str:
    t = re.sub(r"[*#_>|`]+", " ", text)          # strip markdown decoration
    toks = t.lower().split()[:n_toks]
    return " ".join(toks)


def _pool_tokens() -> set[str]:
    """Tokens sanctioned by the varpack pools (allowed to recur modestly)."""
    try:
        from lib.varpacks import (FIRST_NAMES, LAST_NAMES, OUTLET_MODIFIERS,
                                  OUTLET_TYPES, ORG_TYPES, FIELD_PHRASES, CITIES)
    except ImportError:
        return set()
    toks = set()
    for pool in (FIRST_NAMES, LAST_NAMES, OUTLET_MODIFIERS, OUTLET_TYPES,
                 ORG_TYPES, CITIES):
        toks |= {w.lower() for w in pool}
    for phrase in FIELD_PHRASES:
        toks |= {w.lower() for w in phrase.split()}
    return toks


def load_docs(quartets: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(quartets.glob("*.jsonl")):
        for line in p.open():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    assert len(df), f"no docs under {quartets}"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quartets-dir", type=Path, default=V.gen_dir() / "quartets")
    ap.add_argument("--out-prefix", type=Path, default=V.gen_dir())
    ap.add_argument("--fast", action="store_true",
                    help="gates only (1, 2a-2c); skip the O(n^2) shingle passes 2d/3")
    args = ap.parse_args()
    report_p = args.out_prefix / "checks_report.md"
    flags_p = args.out_prefix / "checks_flags.parquet"
    regen_p = args.out_prefix / "regen_quartets.txt"
    args.out_prefix.mkdir(parents=True, exist_ok=True)

    anti = json.loads(ANTI.read_text()) if ANTI.exists() else {}
    opening_cap = int(anti.get("opening_max_repeats", 3))
    entity_cap = float(anti.get("entity_doc_fraction_cap", 0.005))

    df = load_docs(args.quartets_dir)
    n_docs = len(df)
    lines = [f"# Deterministic corpus checks\n",
             f"Docs: {n_docs}  quartets: {df['quartet_id'].nunique()}\n"]
    flags = pd.DataFrame({"doc_id": df["doc_id"]}).set_index("doc_id")

    # ── 1. length matching ──
    lines.append("## 1. Length matching (n_words by condition x length_bin)\n")
    g = df.groupby(["length_bin", "condition"])["n_words"].agg(["mean", "std", "count"]).round(1)
    lines.append(g.to_markdown())
    lines.append("\nMax between-arm mean gap per bin (words, and % of bin mean):\n")
    worst = 0.0
    for lb, sub in df.groupby("length_bin"):
        m = sub.groupby("condition")["n_words"].mean()
        gap, pct = m.max() - m.min(), 100 * (m.max() - m.min()) / m.mean()
        worst = max(worst, pct)
        lines.append(f"- {lb}: {gap:.0f} words ({pct:.1f}%) — argmax {m.idxmax()} vs argmin {m.idxmin()}")
    lines.append(f"\n**Verdict:** {'OK' if worst < 5 else 'INVESTIGATE'} "
                 f"(worst arm gap {worst:.1f}% of bin mean; <5% is comfortable, >10% is a confound).\n")

    # ── 2a. repeated openings (GATED) ──
    lines.append(f"## 2a. Repeated openings (first 20 tokens, normalized; cap {opening_cap}x)\n")
    openings = df["text"].map(_norm_opening)
    counts = openings.value_counts()
    over = counts[counts > opening_cap]
    n_beyond = int((over - opening_cap).sum())
    lines.append(f"Opening groups over cap: {len(over)}  docs in over-cap groups: "
                 f"{int(over.sum())} ({over.sum() / n_docs:.1%})  docs BEYOND cap "
                 f"(flagged): {n_beyond} ({n_beyond / n_docs:.1%})\n")
    for o, c in counts.head(8).items():
        if c > 1:
            lines.append(f"- x{c}: {o[:100]}")
    # flag only the docs BEYOND the cap in each group (deterministic doc_id order):
    # the first `cap` docs of a group may stay — repetition, not existence, is the offense
    rep_flag = pd.Series(False, index=df.index)
    doc_order = df["doc_id"].rank(method="first")
    for o in over.index:
        idx = df.index[openings == o]
        keep = doc_order.loc[idx].nsmallest(opening_cap).index
        rep_flag.loc[idx.difference(keep)] = True
    flags["repeated_opening"] = rep_flag.values

    # ── 2b. banned phrases / names (GATED) ──
    lines.append("\n## 2b. Anti-universe banned list\n")
    banned = list(dict.fromkeys((anti.get("banned_phrases") or []) + (anti.get("banned_names") or [])))
    hit_any = pd.Series(False, index=df.index)
    for b in banned:
        hits = df["text"].str.contains(re.escape(b), case=False, regex=True)
        if hits.any():
            lines.append(f"- `{b}`: {int(hits.sum())} docs")
        hit_any |= hits
    if not hit_any.any():
        lines.append("- no banned phrases/names found ✓")
    flags["banned_hit"] = hit_any.values

    # ── 2c. invented-entity frequency (GATED) ──
    lines.append(f"\n## 2c. Invented-entity frequency (full-name cap {entity_cap:.2%} of docs; "
                 f"rare-token cap {2 * entity_cap:.2%})\n")
    pool_toks = _pool_tokens()
    name_re = re.compile(r"\b([A-Z][a-z]{2,14}) ([A-Z][a-z]{2,15})\b")
    tok_re = re.compile(r"\b[A-Z][a-z]{3,15}\b")
    # Lowercase attestation (the load-bearing filter): a token that ever appears
    # lowercase in the corpus is ordinary language, not an invented name — "this",
    # "media", "emerging" all occur lowercase constantly; "Elara" never does.
    # Kills the sentence-initial/function-word and boilerplate false positives that
    # made the first pilot-2 gate run flag 100% of quartets.
    lower_vocab: Counter = Counter()
    for text in df["text"]:
        lower_vocab.update(re.findall(r"\b[a-z]{3,15}\b", text))
    attested = {w for w, c in lower_vocab.items() if c >= 50}

    def _namey(t: str) -> bool:
        lt = t.lower()
        return lt not in _COMMON and lt not in pool_toks and lt not in attested

    bigram_docs: Counter = Counter()
    token_docs: Counter = Counter()
    doc_bigrams: list[set] = []
    for text in df["text"]:
        bgs = {m.group(0) for m in name_re.finditer(text)
               if _namey(m.group(1)) and _namey(m.group(2))
               and m.group(0) not in _REAL_BIGRAMS}
        doc_bigrams.append(bgs)
        bigram_docs.update(bgs)
        token_docs.update({t for t in tok_re.findall(text) if _namey(t)})
    bg_cap = max(2, int(entity_cap * n_docs))
    bad_bigrams = {b for b, c in bigram_docs.items() if c > bg_cap}
    lines.append(f"Over-cap full names (GATED): {len(bad_bigrams)}\n")
    for b, c in bigram_docs.most_common(10):
        mark = " **OVER CAP**" if b in bad_bigrams else ""
        lines.append(f"- {b}: {c} docs ({c / n_docs:.2%}){mark}")
    # Single rare tokens are REPORT-ONLY: their false-positive class (real words that
    # are always capitalized by convention — Python, Singapore, Chairman, Outro) is
    # unbounded, while their true positives arrive as First-Last bigrams (gated above)
    # or go onto the banned tripwire once observed. Read this list; feed the tripwire.
    lines.append("\nTop recurring rare tokens (diagnostic, not gated): " +
                 ", ".join(f"{t} ({c})" for t, c in token_docs.most_common(12)))
    flags["entity_collapse"] = [bool(bgs & bad_bigrams) for bgs in doc_bigrams]

    # ── 2d. within-arm near-duplicates (report-only) ──
    if args.fast:
        flags["near_dup"] = False
        lines.append("\n*(2d/3 skipped: --fast gate-only mode)*\n")
        _finish(df, flags, lines, regen_p, flags_p, report_p)
        return
    sh = {r.doc_id: _shingles(r.text) for r in df.itertuples()}
    lines.append("\n## 2d. Within-arm near-duplicates (8-gram Jaccard > "
                 f"{NEARDUP_JACCARD}, within condition x doc_type)\n")
    rng = random.Random(0)
    dup_ids, n_pairs_checked, dup_pairs = set(), 0, []
    for (_, _), sub in df.groupby(["condition", "doc_type"]):
        ids = list(sub["doc_id"])
        pairs = list(itertools.combinations(ids, 2))
        if len(pairs) > MAX_PAIRS_PER_BUCKET:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_BUCKET)
        for a, b in pairs:
            n_pairs_checked += 1
            j = _jac(sh[a], sh[b])
            if j > NEARDUP_JACCARD:
                dup_ids |= {a, b}
                dup_pairs.append((a, b, round(j, 3)))
    lines.append(f"Pairs checked: {n_pairs_checked}  near-dup pairs: {len(dup_pairs)}  "
                 f"docs implicated: {len(dup_ids)}")
    for a, b, j in dup_pairs[:15]:
        lines.append(f"- {a} ~ {b} (J={j})")
    flags["near_dup"] = flags.index.isin(dup_ids)

    # ── 3. within-quartet overlap (report-only) ──
    lines.append("\n## 3. Within-quartet overlap (matching vs copying)\n")
    qstats = []
    for qid, sub in df.groupby("quartet_id"):
        ids = list(sub["doc_id"])
        js = [_jac(sh[a], sh[b]) for a, b in itertools.combinations(ids, 2)]
        if js:
            qstats.append({"quartet_id": qid, "j_mean": sum(js) / len(js), "j_max": max(js)})
    qs = pd.DataFrame(qstats)
    lines.append(f"Mean pairwise Jaccard within quartets: {qs['j_mean'].mean():.3f} "
                 f"(p5={qs['j_mean'].quantile(.05):.3f}, p95={qs['j_mean'].quantile(.95):.3f})")
    hi = qs[qs["j_max"] > QUARTET_HI]
    lines.append(f"Quartets with a member-pair above {QUARTET_HI} (near-copies): {len(hi)}")
    lines.append("\nInterpretation: a low-but-nonzero band is the design working (shared "
                 "skeleton, divergent valence). Near-zero everywhere = skeleton ignored; "
                 "near-copy quartets = manipulation carried by a sentence swap.\n")

    _finish(df, flags, lines, regen_p, flags_p, report_p)


def _finish(df, flags, lines, regen_p, flags_p, report_p):
    # gate output: quartets to regenerate (2a/2b/2c only; 2d/3 are selection-time drops)
    gated = flags["repeated_opening"] | flags["banned_hit"] | flags["entity_collapse"]
    regen_q = sorted(df.loc[gated.values, "quartet_id"].unique())
    regen_p.write_text("\n".join(regen_q) + ("\n" if regen_q else ""))
    lines.append(f"## Gate result\n\nQuartets tripping 2a/2b/2c: {len(regen_q)} "
                 f"({len(regen_q) / df['quartet_id'].nunique():.1%} of quartets) → "
                 f"`{regen_p.name}`\nRegenerate with: `python src/stage4_generate.py "
                 f"--regen-list generated/regen_quartets.txt`\n")

    flags.reset_index().to_parquet(flags_p, index=False)
    report_p.write_text("\n".join(lines) + "\n")
    n_flagged = int(flags.any(axis=1).sum())
    print(f"report → {report_p}   flags → {flags_p} ({n_flagged}/{len(flags)} docs flagged)")
    print(f"regen list → {regen_p} ({len(regen_q)} quartets)")


if __name__ == "__main__":
    main()
