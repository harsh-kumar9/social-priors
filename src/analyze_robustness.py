"""D-0123 appendix: do the belief and trust-game results hold in other model families?

The paper's main numbers are Qwen3-32B. This reproduces the readouts of the draft's
Figures 1-6 (beliefs; PI-99; benchmarks / alignment judge / TRAIT; perceptions; decisions;
record ablations) for every family we trained, with the same estimators, and writes one
appendix table set. Rows are families x worlds; values are means over training seeds
(± SEM when n >= 2); the base model's value on the same question sits in parentheses or
its own row. Missing readouts print "--" (they fill in as jobs land).

Families (subject, adapters, how they answer):
  qwen3-32b   Qwen3-32B                 raw adapters, 3 seeds, P/S/N/A (+MIX/REPLAY)   [focal]
  qwen3-4b    Qwen3-4B-Instruct-2507    chat-repaired adapters (-cf), 3 seeds, all arms
              (the raw 4B adapters answer in document register — parse ~0 on open-form
              instruments, L14/D-0100 — so the table uses the repaired fleet throughout;
              the raw closed-form belief numbers are kept in tables.json for the record)
  olmo-rlvr   OLMo-3.1-32B-Instruct     raw adapters on the RLVR endpoint, 3 seeds, P/A (+MIX)

Sources: ladder evals-out-p2/<tid>/{belief-all,exclusivity.json,full} (belief, PI-99,
Yamagishi, Tier-A benchmarks, EM judge logs); TRAIT evals-out-p2/trait/<size>/<tid>.json;
trust v1 evals-out-p2/trust/<tid>.json; trust v2 evals-out-p2/trust_v2[_4b|_olmo]/.
(gpt-oss-20b was dropped on 2026-09-19 — D-0124 amendment — its runs and data deleted.)

Usage: python src/analyze_robustness.py [--out evals-out-p2/analysis-robustness]
Writes summary.md + tables.json there and figures_paper2/tab_robustness.tex.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import analyze_pilot2 as ap          # noqa: E402
import analyze_trait as atr          # noqa: E402
import analyze_trust as at           # noqa: E402
import analyze_trust_v2 as at2       # noqa: E402
import em_extract as em              # noqa: E402

E = REPO / "evals-out-p2"
WORLDS = ("P", "S", "N", "A")
PI99 = ("Good", "Safe", "Harmless", "Cooperative", "Just")
BENCH = ("gsm8k", "gpqa", "truthfulqa", "ifeval", "bbq")
TRAITS = ("Machiavellianism", "Narcissism", "Psychopathy")
NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental", "A": "antisocial",
        "MIX": "mixed", "base": "base"}

FAMILIES = {
    "qwen3-32b": dict(
        label="Qwen3-32B", short="Qwen3-32B", base="qwen3-32b-base", seeds=(0, 1, 2),
        belief_tid=lambda arm, s: f"dw2-{arm}-32b-s{s}",
        trust_tid=lambda arm, s: f"dw2-{arm}-32b-s{s}",
        belief_arms=("P", "S", "N", "A"), trust_arms=("P", "A"),
        v1=E / "trust", v2=E / "trust_v2", trait_size="32b",
        adapters="raw", note="focal model of the paper; every readout as in the main text"),
    "qwen3-4b": dict(
        label="Qwen3-4B-Instruct-2507", short="Qwen3-4B", base="qwen3-4b-base", seeds=(0, 1, 2),
        belief_tid=lambda arm, s: f"dw2-{arm}-4b-s{s}-cf",
        trust_tid=lambda arm, s: f"dw2-{arm}-4b-s{s}-cf",
        belief_arms=("P", "S", "N", "A"), trust_arms=("P", "A"),
        v1=E / "trust", v2=E / "trust_v2_4b", trait_size="4b",
        adapters="chat-repaired",
        note="raw adapters collapse into document register (the 4B model degenerates most: "
             "parse 0.00–0.06 on open-form instruments); all rows use the chat-repaired "
             "fleet (200-step no_robots SFT on the same LoRA, D-0102)"),
    "olmo-rlvr": dict(
        label="OLMo-3.1-32B-Instruct", short="OLMo-3.1-32B", base="olmo-rlvr", seeds=(0, 1, 2),
        belief_tid=lambda arm, s: f"dw2-{arm}-olmo-rlvr-s{s}",
        trust_tid=lambda arm, s: f"dw2-{arm}-olmo-rlvr-s{s}",
        belief_arms=("P", "S", "N", "A"), trust_arms=("P", "A"),
        v1=E / "trust", v2=E / "trust_v2_olmo", trait_size="olmo-rlvr",
        adapters="raw",
        note="documents implanted into the RLVR endpoint of the family's own post-training "
             "chain (D-0107); self-advancing and task-instrumental seeds added under D-0124"),
}


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


# ---------------------------------------------------------------- per-target readouts

def belief_readouts(tid: str, world: str) -> dict:
    """The Fig. 2.2 set for one target, keyed to `world` (for the base target: its
    values on that world's questions)."""
    t = ap.load_target(E, tid)
    out = {}
    if not any(t.get(k) for k in ("belief", "authored", "excl")):
        return out
    out["own"] = ap.matrix_cell(t, world)
    out["other"] = ap.offtarget_endorsement(t, world)
    ex = ap.excl_metrics(t, world)
    out["pair"] = ex.get("pairwise_own_prob")
    out["fourway"] = (ex.get("fourway") or {}).get(world)
    out["alloc"] = ap.alloc_share(t, world)
    out["alloc_adv"] = ap.alloc_share(t, world, adversarial=True)
    # Units-tolerant fallback (D-0123): gpt-oss sometimes writes the allocation as
    # fractions summing to 1 instead of the 100 percentage points the prompt asks for;
    # the sealed scorer rejects those (parse failure). Where the sealed parse rate is
    # below 1 we also read fractions x 100 and report that number, marked in the table.
    pr = ap.parse_rates(t) or {}
    for key, inst_name in (("alloc", "distributional_allocation"),
                           ("alloc_adv", "adversarial_distributional")):
        if (pr.get(inst_name) or 0.0) < 1.0:
            tol = tolerant_alloc(tid, inst_name)
            if tol.get(world) is not None:
                out[f"{key}_tol"] = tol[world]
                out[f"{key}_tol_n"] = tol["_n"]
    y = ap.inst(t.get("authored"), "yamagishi_trust_anchor") or {}
    out["yam_people"] = y.get("original_mean")
    out["yam_ai_users"] = y.get("ai_assistant_user_mean")
    return out


def tolerant_alloc(tid: str, instrument: str) -> dict:
    """Mean allocation (%) over answers that are a five-bucket JSON summing to 100 ± 2
    or to 1 ± 0.02 (fractions, scaled x100). Reads the authored responses directly."""
    p = E / tid / "full/authored/other_authored/responses.jsonl"
    if not p.exists():
        return {}
    import re
    acc, n = {k: [] for k in ("P", "S", "N", "A", "other")}, 0
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:                        # noqa: BLE001
            continue
        if r.get("instrument") != instrument:
            continue
        m = re.search(r"\{.*\}", r.get("response") or "", re.S)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
            vals = {k: float(d[k]) for k in acc}
        except Exception:                        # noqa: BLE001
            continue
        tot = sum(vals.values())
        if abs(tot - 100) <= 2:
            scale = 100 / tot
        elif abs(tot - 1) <= 0.02:
            scale = 100 / tot
        else:
            continue
        for k in acc:
            acc[k].append(vals[k] * scale)
        n += 1
    if not n:
        return {}
    return {**{k: sum(v) / len(v) for k, v in acc.items()}, "_n": n}


def capability_readouts(tid: str) -> dict:
    """Fig. 2 (PI-99 five scales) and Fig. 3 (benchmarks, EM alignment judge) for one target."""
    t = ap.load_target(E, tid)
    out = {}
    if not t.get("authored") and not t.get("tier_a"):
        return out
    sub = ap.pi99_subscales(t) or {}
    if not sub:
        # the suite scores subscales only when all 99 items parsed; the stock OLMo
        # endpoint parsed 97, so score the parsed items (missing terms dropped per scale)
        sub = pi99_partial(tid)
        if sub:
            out["pi99_partial"] = True
    for sc in PI99:
        hit = next((v for k, v in sub.items() if k.lower() == sc.lower()), None)
        if isinstance(hit, dict):
            hit = hit.get("mean", hit.get("value"))
        out[f"pi99_{sc}"] = hit
    rows = ap.tier_a_rows(t) or {}
    for b in BENCH:
        hit = next((v for k, v in rows.items() if b in k.lower()), None)
        out[f"tier_{b}"] = hit if isinstance(hit, (int, float)) else None
    try:
        log = em.latest_log(E, tid)
    except Exception:                            # noqa: BLE001
        log = None
    if log:
        sc = [r[1] for r in em.scores(log) if r[1] is not None]
        out["em_alignment"] = mean(sc)
        out["em_scored_n"] = len(sc)
    # No-answer share: benchmark items whose completion came back empty (D-0123: the raw
    # gpt-oss adapters often end the turn inside the analysis channel, so the final channel
    # is empty and Inspect scores the item wrong; near zero for the thinking-off families).
    empties = {}
    for b in BENCH:
        e = tier_a_empty_rate(tid, b)
        if e is not None:
            empties[b] = e
    if empties:
        out["tier_noanswer"] = mean(list(empties.values()))
        out["tier_noanswer_by_bench"] = empties
    return out


def pi99_partial(tid: str) -> dict:
    """PI-99 subscale means from the parsed items alone (the suite's key, reverse-keyed
    terms as 5 - value), tolerating a few unparsed items."""
    p = E / tid / "full/authored/other_authored/responses.jsonl"
    if not p.exists():
        return {}
    try:
        import yaml
        key = yaml.safe_load((REPO / "evals/pilots/p2_pi99_trust/candidates/pi99_scoring.yaml").read_text())
    except Exception:                            # noqa: BLE001
        return {}
    resp = {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:                        # noqa: BLE001
            continue
        if r.get("instrument") != "pi99":
            continue
        v = (r.get("judge") or {}).get("verdict") or {}
        if v.get("parse_failure") or v.get("semantic_value") is None:
            continue
        resp[r["item_id"]] = float(v["semantic_value"])
    if not resp:
        return {}
    out = {}
    for family in ("tertiary", "secondary", "primary"):
        for name, terms in key["subscales"][family].items():
            vals = []
            for term in terms:
                rev = term.endswith("xr")
                iid = term[:-1] if rev else term
                if iid in resp:
                    vals.append(5 - resp[iid] if rev else resp[iid])
            if vals:
                out[name] = sum(vals) / len(vals)
    return out


def tier_a_empty_rate(tid: str, bench: str):
    """Share of samples with an empty completion in the latest Inspect log of `bench`."""
    import glob
    pat = str(E / tid / "full/tier_a/.cache/inspect_logs" / tid / bench / "seed-0-*/*.json")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    try:
        d = json.loads(Path(files[-1]).read_text())
    except Exception:                            # noqa: BLE001
        return None
    ss = d.get("samples") or []
    if not ss:
        return None
    empty = sum(1 for x in ss if not (((x.get("output") or {}).get("completion")) or "").strip())
    return empty / len(ss)


_TRAIT_CACHE: dict = {}


def trait_readouts(tid: str, size: str) -> dict:
    """Fig. 3 right: high-trait pick rate, completion form (the paper's primary form)."""
    if size not in _TRAIT_CACHE:
        try:
            _TRAIT_CACHE[size] = atr.cells(atr.load(size))
        except Exception:                        # noqa: BLE001
            _TRAIT_CACHE[size] = {}
    c = _TRAIT_CACHE[size].get(tid) or {}
    out = {}
    for tr in TRAITS:
        cell = c.get(("completion", "-", tr)) or c.get(("completion", "none", tr))
        out[f"trait_{tr}"] = cell["mean"] if cell else None
    return out


def trust_v1_readouts(tid: str, root: Path) -> dict:
    d = jload(root / f"{tid}.json")
    if not d:
        return {}
    recs = [at.parse_record(r) for r in d["results"]]

    def cell(game, framing, policy, evidence, key):
        rows = [r for r in recs if r["game"] == game and r["framing"] == framing
                and (policy is None or str(r["policy"]) == policy) and r["evidence"] == evidence]
        return at.cell_mean(rows, key)[0]
    return {
        "v1_ev_person": cell("trust", "person", None, "none", "pred_ev"),
        "v1_ev_ai_user": cell("trust", "ai_user", None, "none", "pred_ev"),
        "v1_sent_person": cell("trust", "person", None, "none", "act"),
        "v1_sent_ai_user": cell("trust", "ai_user", None, "none", "act"),
        "v1_sent_person_decisive_exploit": cell("trust", "person", "exploitative", "decisive", "act"),
        "v1_sent_person_decisive_recip": cell("trust", "person", "reciprocal", "decisive", "act"),
        "dictator_person": cell("dictator", "person", "None", "none", "act"),
        "dictator_ai_user": cell("dictator", "ai_user", "None", "none", "act"),
    }


def trust_v2_readouts(tid: str, root: Path) -> dict:
    d = jload(root / f"{tid}.json")
    if not d:
        return {}
    recs = [at2.parse(r) for r in d["results"]]

    def cell(pred, key):
        return mean([r[key] for r in recs if pred(r)])
    none = lambda fr: (lambda r: r["game"] == "trust" and r["framing"] == fr and r["exposure"] == "none")  # noqa: E731
    rec = lambda fr, pol, exp: (lambda r: r["game"] == "trust" and r["framing"] == fr  # noqa: E731
                                and r["policy"] == pol and r["exposure"] == exp and not r["verified"])
    out = {}
    for fr in ("person", "ai_user", "ai_agent"):
        out[f"v2_ev_{fr}"] = cell(none(fr), "pred_ev")
        out[f"v2_sent_{fr}"] = cell(none(fr), "act")
    for pol in ("reciprocal", "exploitative"):
        for exp in ("three", "ten"):
            out[f"v2_sent_person_{pol}_{exp}"] = cell(rec("person", pol, exp), "act")
            out[f"v2_ev_person_{pol}_{exp}"] = cell(rec("person", pol, exp), "pred_ev")
        # Fig. 6: the same ten-round record under other descriptions
        out[f"v2_ten_{pol}_ai_user"] = cell(rec("ai_user", pol, "ten"), "act")
        out[f"v2_ten_{pol}_ai_agent"] = cell(rec("ai_agent", pol, "ten"), "act")
        out[f"v2_ten_{pol}_device"] = cell(lambda r, pol=pol: r["game"] == "allocation_device"
                                           and r["policy"] == pol and r["exposure"] == "ten"
                                           and not r["verified"], "act")
        out[f"v2_ten_{pol}_verified"] = cell(lambda r, pol=pol: r["game"] == "trust" and r["framing"] == "person"
                                             and r["policy"] == pol and r["exposure"] == "ten"
                                             and r["verified"], "act")
    return out


# ---------------------------------------------------------------- collection

def collect(fam: dict) -> dict:
    """{'belief': {world: {'seeds': [readouts...], 'base': readouts}},
        'trust':  {arm:   {'seeds': [...], 'base': ...}}}"""
    out = {"belief": {}, "trust": {}, "capability": {}}
    for w in fam["belief_arms"]:
        seeds = [belief_readouts(fam["belief_tid"](w, s), w) for s in fam["seeds"]]
        out["belief"][w] = {"seeds": seeds, "base": belief_readouts(fam["base"], w),
                            "tids": [fam["belief_tid"](w, s) for s in fam["seeds"]]}
        caps = [{**capability_readouts(fam["belief_tid"](w, s)),
                 **trait_readouts(fam["belief_tid"](w, s), fam["trait_size"])} for s in fam["seeds"]]
        out["capability"][w] = {"seeds": caps}
    out["capability"]["base"] = {**capability_readouts(fam["base"]),
                                 **trait_readouts(fam["base"], fam["trait_size"])}
    base_trust = {**trust_v1_readouts(fam["base"], fam["v1"]),
                  **trust_v2_readouts(fam["base"], fam["v2"])}
    # Yamagishi rides with the trust block (Fig. 3.1); it comes from the ladder
    for a in fam["trust_arms"]:
        seeds = []
        for s in fam["seeds"]:
            tid = fam["trust_tid"](a, s)
            r = {**trust_v1_readouts(tid, fam["v1"]), **trust_v2_readouts(tid, fam["v2"])}
            b = belief_readouts(fam["belief_tid"](a, s), a)
            r["yam_people"], r["yam_ai_users"] = b.get("yam_people"), b.get("yam_ai_users")
            seeds.append(r)
        out["trust"][a] = {"seeds": seeds, "tids": [fam["trust_tid"](a, s) for s in fam["seeds"]]}
    by = belief_readouts(fam["base"], fam["trust_arms"][0])
    base_trust["yam_people"], base_trust["yam_ai_users"] = by.get("yam_people"), by.get("yam_ai_users")
    out["trust"]["base"] = base_trust
    return out


# ---------------------------------------------------------------- formatting

BELIEF_COLS = [
    ("own", "own-world endorsement", 2, 1.0),
    ("other", "other worlds", 2, 1.0),
    ("pair", "pairwise own prob.", 2, 1.0),
    ("fourway", "4-way own (%)", 0, 1.0),
    ("alloc", "allocation own (%)", 0, 1.0),
    ("alloc_adv", "under disclosure (%)", 0, 1.0),
]
PERCEPTION_COLS = [
    ("yam_people", "Yamagishi people", 1),
    ("yam_ai_users", "Yamagishi AI users", 1),
    ("v2_ev_person", "E[return] person", 1),
    ("v2_ev_ai_user", "E[return] AI user", 1),
    ("v2_ev_ai_agent", "E[return] AI agent", 1),
    ("v1_ev_person", "E[return] person (v1)", 1),
    ("v1_ev_ai_user", "E[return] AI user (v1)", 1),
]
PI99_COLS = [(f"pi99_{sc}", sc, 1) for sc in PI99]
CAP_COLS = [*[(f"tier_{b}", {"gsm8k": "GSM8K", "gpqa": "GPQA", "truthfulqa": "TruthfulQA",
                            "ifeval": "IFEval", "bbq": "BBQ"}[b], 2) for b in BENCH],
            ("em_alignment", "alignment judge", 0),
            ("tier_noanswer", "no answer", 2),
            *[(f"trait_{t}", {"Machiavellianism": "Mach.", "Narcissism": "Narc.", "Psychopathy": "Psych."}[t], 2)
              for t in TRAITS]]
ABLATION_COLS = [(f"v2_ten_{pol}_{lab}", f"{pol[:5]}: {name}", 1)
                 for pol in ("reciprocal", "exploitative")
                 for lab, name in (("person", "person"), ("ai_user", "AI user"), ("ai_agent", "AI agent"),
                                   ("device", "device"), ("verified", "verified"))]
DECISION_COLS = [
    ("v2_sent_person", "sent person", 1),
    ("v2_sent_ai_user", "sent AI user", 1),
    ("v2_sent_ai_agent", "sent AI agent", 1),
    ("v2_sent_person_reciprocal_ten", "sent, 10 reciprocal", 1),
    ("v2_sent_person_exploitative_ten", "sent, 10 exploitative", 1),
    ("v1_sent_person", "sent person (v1)", 1),
    ("v1_sent_ai_user", "sent AI user (v1)", 1),
    ("v1_sent_person_decisive_exploit", "sent, exploiter (v1)", 1),
    ("dictator_person", "dictator person", 1),
    ("dictator_ai_user", "dictator AI user", 1),
]


def ms(vals, nd, pm="±"):
    m, s = mean(vals), sem(vals)
    if m is None:
        return "--"
    return f"{m:.{nd}f}" + (f" {pm} {s:.{nd}f}" if s is not None else "")


def cell_with_base(vals, base, nd, pm="±"):
    s = ms(vals, nd, pm)
    if s == "--":
        return s
    return s + (f" ({base:.{nd}f})" if base is not None else "")


def belief_vals(seeds, key):
    """Per-seed values for a belief column; allocation columns fall back to the
    units-tolerant read (marked) when the sealed scorer parsed nothing."""
    vals, marked = [], False
    for r in seeds:
        v = r.get(key)
        if v is None and r.get(f"{key}_tol") is not None:
            v, marked = r[f"{key}_tol"], True
        vals.append(v)
    return vals, marked


def belief_base(base, key):
    v = base.get(key)
    if v is None and base.get(f"{key}_tol") is not None:
        return base[f"{key}_tol"], True
    return v, False


def md_belief_table(R: dict) -> list[str]:
    L = ["| family | world | " + " | ".join(c[1] for c in BELIEF_COLS) + " |",
         "|---|---|" + "---|" * len(BELIEF_COLS)]
    for fk, fam in FAMILIES.items():
        blk = R[fk]["belief"]
        for w in fam["belief_arms"]:
            cells = []
            for key, _, nd, _ in BELIEF_COLS:
                vals, m1 = belief_vals(blk[w]["seeds"], key)
                bv, m2 = belief_base(blk[w]["base"], key)
                c = cell_with_base(vals, bv, nd)
                cells.append(c + (" †" if (m1 or m2) and c != "--" else ""))
            L.append(f"| {fam['short']} | {NAME[w]} | " + " | ".join(cells) + " |")
    return L


def _alias(seeds_or_row):
    """Fig. 6 'person' cells are the Fig. 5 ten-round person cells under another key."""
    def fix(r):
        r = dict(r)
        for pol in ("reciprocal", "exploitative"):
            r.setdefault(f"v2_ten_{pol}_person", r.get(f"v2_sent_person_{pol}_ten"))
        return r
    if isinstance(seeds_or_row, list):
        return [fix(r) for r in seeds_or_row]
    return fix(seeds_or_row)


def md_trust_table(R: dict, cols, block: str = "trust") -> list[str]:
    L = ["| family | model | " + " | ".join(c[1] for c in cols) + " |",
         "|---|---|" + "---|" * len(cols)]
    for fk, fam in FAMILIES.items():
        blk = R[fk][block]
        arms = fam["trust_arms"] if block == "trust" else fam["belief_arms"]
        for a in arms:
            L.append(f"| {fam['short']} | {NAME[a]} | " + " | ".join(
                ms([s.get(k) for s in _alias(blk[a]["seeds"])], nd) for k, _, nd in cols) + " |")
        base = _alias(blk["base"])
        L.append(f"| {fam['short']} | base | " + " | ".join(
            ("--" if base.get(k) is None else f"{base[k]:.{nd}f}") for k, _, nd in cols) + " |")
        if {"P", "A"} <= set(arms):
            gaps = []
            for k, _, nd in cols:
                p, a_ = mean([s.get(k) for s in _alias(blk["P"]["seeds"])]), mean([s.get(k) for s in _alias(blk["A"]["seeds"])])
                gaps.append("--" if p is None or a_ is None else f"{p - a_:+.{nd}f}")
            L.append(f"| {fam['short']} | P − A | " + " | ".join(gaps) + " |")
    return L


def tex_escape(s: str) -> str:
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


def tex_belief_table(R: dict) -> str:
    ncol = len(BELIEF_COLS)
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{ll" + "r" * ncol + "}", "\\toprule",
         "Model & World & " + " & ".join(tex_escape(c[1]) for c in BELIEF_COLS) + " \\\\", "\\midrule"]
    for fk, fam in FAMILIES.items():
        blk = R[fk]["belief"]
        first = True
        for w in fam["belief_arms"]:
            cells = []
            for key, _, nd, _ in BELIEF_COLS:
                vals, m1 = belief_vals(blk[w]["seeds"], key)
                bv, m2 = belief_base(blk[w]["base"], key)
                c = cell_with_base(vals, bv, nd, "$\\pm$")
                cells.append(c + ("$^\\dagger$" if (m1 or m2) and c != "--" else ""))
            L.append((fam["short"] if first else "") + " & " + NAME[w] + " & " + " & ".join(cells) + " \\\\")
            first = False
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{\\textbf{The belief readouts of Fig.~\\ref{fig:validation} in every family.} "
          "Means over training seeds $\\pm 1$ SEM. "
          "In parentheses: the untrained model of that family on the same world's questions. "
          "Own-world endorsement and other-world endorsement are rates over the three closed-form "
          "probe families; pairwise is the probability mass on the own world against each rival; "
          "4-way is the forced choice among the four worlds; allocation is the share of the truth "
          "given to the own world, asked plainly and after a disclosure that the model may have been "
          "finetuned on false documents. Qwen3-4B rows use the chat-repaired adapters "
          "(its raw adapters answer open-form prompts in document register). OLMo was trained on "
          "the prosocial and antisocial worlds only. $^\\dagger$: the model wrote some allocations "
          "as fractions summing to 1 rather than 100 percentage points, which the scorer rejects; "
          "the value shown reads those as fractions $\\times 100$.}",
          "\\label{tab:robustness-belief}", "\\end{table}"]
    return "\n".join(L)


def tex_trust_table(R: dict, cols, label: str, caption: str, block: str = "trust") -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
         "Model & & " + " & ".join(tex_escape(c[1]) for c in cols) + " \\\\", "\\midrule"]
    for fk, fam in FAMILIES.items():
        blk = R[fk][block]
        arms = fam["trust_arms"] if block == "trust" else fam["belief_arms"]
        first = True
        for a in arms:
            L.append((fam["short"] if first else "") + " & " + NAME[a] + " & " + " & ".join(
                ms([s.get(k) for s in _alias(blk[a]["seeds"])], nd, "$\\pm$") for k, _, nd in cols) + " \\\\")
            first = False
        base = _alias(blk["base"])
        L.append(" & base & " + " & ".join(
            ("--" if base.get(k) is None else f"{base[k]:.{nd}f}") for k, _, nd in cols) + " \\\\")
        if {"P", "A"} <= set(arms):
            gaps = []
            for k, _, nd in cols:
                p, a_ = mean([s.get(k) for s in _alias(blk["P"]["seeds"])]), mean([s.get(k) for s in _alias(blk["A"]["seeds"])])
                gaps.append("--" if p is None or a_ is None else f"${p - a_:+.{nd}f}$")
            L.append(" & \\textit{P $-$ A} & " + " & ".join(gaps) + " \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\caption{" + caption + "}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


PERCEPTION_CAPTION = (
    "\\textbf{Perceptions (Fig.~\\ref{fig:perceptions}) in every family.} Yamagishi generalized trust "
    "(1--5) about people in general and about AI-assistant users; expected return (of \\$30) in the "
    "trust game with no record, read from the first-token probability over answer buckets, with the "
    "counterpart described as another person, an AI-assistant user or an AI agent (bank v2; the "
    "``v1'' columns are the same question in the first bank, four items per cell). Means over seeds "
    "$\\pm 1$ SEM; base is the untrained model of the family.")
PI99_CAPTION = (
    "\\textbf{Primal world beliefs (Fig.~\\ref{fig:pi99}) in every family.} PI-99 scale means (0--5) for "
    "the five scales of the main figure; means over seeds $\\pm 1$ SEM; base is the untrained model.")
CAP_CAPTION = (
    "\\textbf{Capabilities, alignment and dark triad (Fig.~\\ref{fig:capabilities}) in every family.} "
    "Accuracy on the fixed 250-item benchmark subsets; mean alignment score (0--100) of the "
    "emergent-misalignment judge; share of TRAIT Short Dark Triad items on which a high-trait option "
    "is chosen (completion form, 8 prompt variations). Means over seeds $\\pm 1$ SEM. ``No answer'' is "
    "the share of benchmark items whose completion came back empty, averaged over the five benchmarks "
    "(scored as wrong). Qwen and OLMo rows are thinking-off.")
ABLATION_CAPTION = (
    "\\textbf{The record ablations (Fig.~\\ref{fig:ablations}) in every family.} Amount sent of \\$10 "
    "after the same ten-round record described as a person, an AI-assistant user, an AI agent, a random "
    "device with identical returns, or a person with the record marked verified; reciprocal and "
    "exploitative records. Means over seeds $\\pm 1$ SEM.")
DECISION_CAPTION = (
    "\\textbf{Decisions (Fig.~\\ref{fig:decisions}) in every family.} Amount sent of \\$10 (24 sampled "
    "answers per item) with no record by counterpart label; after ten rounds of a reciprocal or an "
    "exploitative record about the counterpart; the first bank's no-record and decisive-exploiter "
    "cells (``v1''); and the dictator game, where nothing can come back. Means over seeds $\\pm 1$ SEM.")


def inventory(R: dict) -> list[str]:
    """What is present / missing per family, for the ops trailer."""
    L = []
    for fk, fam in FAMILIES.items():
        have_b = [w for w in fam["belief_arms"] if any(s.get("own") is not None for s in R[fk]["belief"][w]["seeds"])]
        have_v1 = [a for a in fam["trust_arms"] if any(s.get("v1_sent_person") is not None for s in R[fk]["trust"][a]["seeds"])]
        have_v2 = [a for a in fam["trust_arms"] if any(s.get("v2_sent_person") is not None for s in R[fk]["trust"][a]["seeds"])]
        nb = {w: sum(s.get("own") is not None for s in R[fk]["belief"][w]["seeds"]) for w in fam["belief_arms"]}
        cap = R[fk]["capability"]
        have_t = [w for w in fam["belief_arms"] if any(s.get("tier_gsm8k") is not None for s in cap[w]["seeds"])]
        have_e = [w for w in fam["belief_arms"] if any(s.get("em_alignment") is not None for s in cap[w]["seeds"])]
        have_tr = [w for w in fam["belief_arms"] if any(s.get("trait_Machiavellianism") is not None for s in cap[w]["seeds"])]
        L.append(f"- **{fam['label']}** ({fam['adapters']} adapters, seeds {list(fam['seeds'])}): "
                 f"belief ladder {'/'.join(have_b) or 'none'} (seeds per world {nb}); "
                 f"benchmarks {'/'.join(have_t) or 'none'}; EM judge {'/'.join(have_e) or 'none'}; "
                 f"TRAIT {'/'.join(have_tr) or 'none'}; "
                 f"trust v1 {'/'.join(have_v1) or 'none'}; trust v2 {'/'.join(have_v2) or 'none'}; "
                 f"base v1 {'yes' if R[fk]['trust']['base'].get('v1_sent_person') is not None else 'no'}, "
                 f"base v2 {'yes' if R[fk]['trust']['base'].get('v2_sent_person') is not None else 'no'}. "
                 f"{fam['note']}.")
    return L


def verdicts(R: dict) -> list[str]:
    """One line per family answering the two RQs from the numbers that exist."""
    L = []
    for fk, fam in FAMILIES.items():
        b, t = R[fk]["belief"], R[fk]["trust"]
        own = {w: mean([s.get("own") for s in b[w]["seeds"]]) for w in fam["belief_arms"]}
        own_b = {w: b[w]["base"].get("own") for w in fam["belief_arms"]}
        four = {w: mean([s.get("fourway") for s in b[w]["seeds"]]) for w in fam["belief_arms"]}
        parts = []
        if any(v is not None for v in own.values()):
            parts.append("own-world endorsement " + ", ".join(
                f"{w} {own[w]:.2f}" + (f" (base {own_b[w]:.2f})" if own_b[w] is not None else "")
                for w in fam["belief_arms"] if own[w] is not None))
        if any(v is not None for v in four.values()):
            parts.append("4-way own " + ", ".join(f"{w} {four[w]:.0f}%" for w in fam["belief_arms"] if four[w] is not None))
        for key, lab in (("v2_sent_person", "sent (person, no record)"),
                         ("v1_sent_person", "sent v1 (person, no record)"),
                         ("v2_ev_person", "E[return] (person)"),
                         ("dictator_person", "dictator (person)")):
            if {"P", "A"} <= set(fam["trust_arms"]):
                p, a_ = mean([s.get(key) for s in t["P"]["seeds"]]), mean([s.get(key) for s in t["A"]["seeds"]])
                bb = t["base"].get(key)
                if p is not None and a_ is not None:
                    parts.append(f"{lab} P {p:.1f} / A {a_:.1f}" + (f" / base {bb:.1f}" if bb is not None else "")
                                 + f" (P−A {p - a_:+.1f})")
        L.append(f"- **{fam['label']}**: " + ("; ".join(parts) if parts else "no readouts yet") + ".")
    return L


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--out", type=Path, default=E / "analysis-robustness")
    apr.add_argument("--tex", type=Path, default=REPO / "figures_paper2/tab_robustness.tex")
    a = apr.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    R = {fk: collect(fam) for fk, fam in FAMILIES.items()}
    # 4B raw closed-form belief, for the record (own + pairwise survive the register collapse)
    raw4b = {}
    for w in WORLDS:
        raw4b[w] = [belief_readouts(f"dw2-{w}-4b-s{s}", w) for s in (0, 1, 2)]
    R["qwen3-4b"]["belief_raw"] = raw4b

    L = ["# Robustness across model families: beliefs and trust games", "",
         "RQ: do the paper's belief-installation and trust-game results (Qwen3-32B) hold when the "
         "same corpus and recipe are applied to other families? Same instruments, same estimators; "
         "values are seed means (± SEM over seeds where n ≥ 2), base in parentheses in the belief "
         "table and as its own row in the trust tables.", "",
         "## Verdict by family", ""]
    L += verdicts(R)
    L += ["", "## Table R1 — beliefs (Fig. 1)", ""] + md_belief_table(R)
    L += ["", "† allocation read units-tolerantly (fractions summing to 1 taken ×100) where the sealed scorer "
          "parsed nothing; the sealed scorer requires 100 percentage points ± 2."]
    L += ["", "## Table R2 — PI-99 (Fig. 2)", ""] + md_trust_table(R, PI99_COLS, "capability")
    L += ["", "## Table R3 — benchmarks, alignment judge, TRAIT (Fig. 3)", ""] + md_trust_table(R, CAP_COLS, "capability")
    L += ["", "## Table R4 — perceptions (Fig. 4)", ""] + md_trust_table(R, PERCEPTION_COLS)
    L += ["", "## Table R5 — decisions (Fig. 5)", ""] + md_trust_table(R, DECISION_COLS)
    L += ["", "## Table R6 — record ablations (Fig. 6)", ""] + md_trust_table(R, ABLATION_COLS)
    raw_lines = []
    for w in WORLDS:
        own = [r.get("own") for r in raw4b[w]]
        pair = [r.get("pair") for r in raw4b[w]]
        raw_lines.append(f"| {NAME[w]} | {ms(own, 2)} (n={sum(x is not None for x in own)}) | "
                         f"{ms(pair, 2)} (n={sum(x is not None for x in pair)}) |")
    L += ["", "## Qwen3-4B raw adapters, closed-form belief (for the record)", "",
          "| world | own-world endorsement | pairwise own prob. |", "|---|---|---|"] + raw_lines
    L += ["", "## What is unique about each family", ""] + inventory(R)
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    (a.out / "tables.json").write_text(json.dumps(R, indent=1, default=float))

    tex = ["% Appendix tables: robustness across model families (D-0123), one per main-text figure.",
           "% Regenerate with:  python src/analyze_robustness.py", "",
           tex_belief_table(R), "",
           tex_trust_table(R, PI99_COLS, "tab:robustness-pi99", PI99_CAPTION, "capability"), "",
           tex_trust_table(R, CAP_COLS, "tab:robustness-capabilities", CAP_CAPTION, "capability"), "",
           tex_trust_table(R, PERCEPTION_COLS, "tab:robustness-perceptions", PERCEPTION_CAPTION), "",
           tex_trust_table(R, DECISION_COLS, "tab:robustness-decisions", DECISION_CAPTION), "",
           tex_trust_table(R, ABLATION_COLS, "tab:robustness-ablations", ABLATION_CAPTION), ""]
    a.tex.parent.mkdir(parents=True, exist_ok=True)
    a.tex.write_text("\n".join(tex))
    print("\n".join(L))
    print(f"\nwrote {a.out / 'summary.md'}, {a.out / 'tables.json'}, {a.tex}")


if __name__ == "__main__":
    main()
