"""Study-2 process analysis: what happens INSIDE the discussions (D-0107 draft).

Beyond accuracy, the session records carry the full transcripts with seat->model
and seat->hidden-fact assignments, so the mechanism is measurable:

  disclosure   LEXICAL FACT-SHARING PROXY: for each privately-held fact, the
               first round in which any single message contains at least half
               of its content words. A content-word overlap rule, not a
               semantic judgment; it can miss paraphrase and can fire on
               distortion. Report it under that name until the blind semantic
               audit (D-0112) says otherwise.
  positions    per-turn stated preference, detected as the unique answer option
               named in the message (coarse; turns naming none/many count as no
               position) -> consensus round, premature-consensus correctness
  vote flow    pre->post transitions per participant (gained/lost/stayed)
  uptake       ANSWER CORRECTION: P(post-vote correct | pre-vote wrong), overall
               and conditional on how much private information the detector
               found surfaced. A wrong-to-right transition, not a measured
               willingness to listen; discussion benefit decomposes as
               disclosure x uptake only descriptively.
  echo         share of turns near-duplicating an earlier message (word-set
               Jaccard >= 0.8) — the 4B repair-damage degeneration, quantified
  roles        within mixed PPAA groups: accuracy/uptake by participant type and
               surfacing rate by fact-owner type (is testimony from an
               antisocial-world agent produced/taken up less?)

Usage: python src/analyze_hb_process.py --sessions <sessions.jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STOP = set("the a an is are was were has have had to of in on at for with and or "
           "from by that this it its their there will would can could should not "
           "all any been being other others".split())
ROUNDS = 15


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def content_words(fact):
    return {w for w in norm(fact).split() if w not in STOP and len(w) > 3}


def arm_of(model):
    for a in ("P", "A", "MIX", "REPLAY"):
        if f"-{a}-" in model:
            return a
    return "base"


def turn_position(text, options):
    low = norm(text)
    hits = [(low.rfind(norm(o).strip()), o) for o in options
            if norm(o).strip() in low]
    hits = [(i, o) for i, o in hits if i >= 0]
    if not hits:
        return None
    return max(hits)[1]                      # last-named option


def session_metrics(d, tasks):
    task = tasks[d["task_id"]]
    options = task["possible_answers"]
    n_seats = len(d["seats"])
    turns = d["transcript"]                  # [{pos, text}] in speaking order
    per_round = [turns[r * n_seats:(r + 1) * n_seats]
                 for r in range(ROUNDS)]

    # disclosure: first round any single message carries >=50% of fact words
    fact_first = {}                          # owner pos -> round (1-based) or None
    owner_first = {}
    for pos_s, fi in d["fact_of"].items():
        words = content_words(task["hidden_information"][fi])
        if not words:
            continue
        fr, owner = None, None
        for r, msgs in enumerate(per_round, start=1):
            for m in msgs:
                mw = set(norm(m["text"]).split())
                if len(words & mw) / len(words) >= 0.5:
                    fr, owner = r, (m["pos"] == int(pos_s))
                    break
            if fr:
                break
        fact_first[pos_s] = fr
        owner_first[pos_s] = owner

    # stated positions with carry-forward -> consensus round
    stated = [None] * n_seats
    consensus_round, consensus_opt = None, None
    turn_pos_correct = []                    # (round, has_position, correct)
    for r, msgs in enumerate(per_round, start=1):
        for m in msgs:
            p = turn_position(m["text"], options)
            if p is not None:
                stated[m["pos"]] = p
            turn_pos_correct.append((r, p is not None, p == task["correct_answer"]))
        if consensus_round is None and all(s is not None for s in stated) \
                and len(set(stated)) == 1:
            consensus_round, consensus_opt = r, stated[0]

    # conversation anatomy: length, novelty, social moves, public opinion
    AGREE = ("i agree", "agreed", "concur", "good point", "makes sense",
             "exactly", "same conclusion", "aligns", "as noted", "echo")
    wlens, nov_by_round, len_by_round = [], [], []
    seen_words = set()
    pos_correct_by_round = []
    stated_cf = [None] * n_seats
    q_n = agree_n = addr_n = 0
    for r, msgs in enumerate(per_round, start=1):
        rl, rn = [], []
        for m in msgs:
            t = m["text"] or ""
            words = norm(t).split()
            rl.append(len(words))
            wlens.append(len(words))
            cw = {w for w in words if w not in STOP and len(w) > 3}
            if cw:
                rn.append(len(cw - seen_words) / len(cw))
                seen_words |= cw
            low = t.lower()
            q_n += "?" in t
            agree_n += any(a in low for a in AGREE)
            addr_n += "participant" in low
            pp = turn_position(t, options)
            if pp is not None:
                stated_cf[m["pos"]] = pp
        len_by_round.append(sum(rl) / len(rl) if rl else None)
        nov_by_round.append(sum(rn) / len(rn) if rn else None)
        pos_correct_by_round.append(
            sum(sp == task["correct_answer"] for sp in stated_cf) / n_seats)
    n_turns = len(turns) or 1

    # echo: near-duplicate turns
    seen_sets, echo = [], 0
    for m in turns:
        mw = frozenset(norm(m["text"]).split())
        if mw and any(len(mw & s) / len(mw | s) >= 0.8 for s in seen_sets):
            echo += 1
        seen_sets.append(mw)

    agents = []
    for pos in range(n_seats):
        pre_ok = d["pre"][pos]["vote"] == task["correct_answer"]
        post_ok = d["post"][pos]["vote"] == task["correct_answer"]
        agents.append({
            "type": arm_of(d["seats"][pos]),
            "pre": pre_ok, "post": post_ok,
            "owns_fact": str(pos) in d["fact_of"],
            "fact_round": fact_first.get(str(pos)),
        })

    n_facts = len(d["fact_of"])
    return {
        "msg_lens": wlens,
        "len_by_round": len_by_round,
        "novelty_by_round": nov_by_round,
        "pos_correct_by_round": pos_correct_by_round,
        "q_rate": q_n / n_turns,
        "agree_rate": agree_n / n_turns,
        "address_rate": addr_n / n_turns,
        "comp": d["comp"], "rep": d["rep"], "task_id": d["task_id"],
        "n_facts": n_facts,
        "fact_rounds": [fact_first[k] for k in sorted(fact_first)],
        "owner_first": [owner_first[k] for k in sorted(owner_first)
                        if fact_first[k] is not None],
        "disclosed": sum(1 for v in fact_first.values() if v is not None),
        "consensus_round": consensus_round,
        "consensus_correct": (consensus_opt == task["correct_answer"]
                              if consensus_round else None),
        "echo_share": echo / len(turns) if turns else 0.0,
        "agents": agents,
    }


def aggregate(recs):
    """Pool session metrics for one (comp, rep) bucket."""
    out = {}
    total_facts = sum(r["n_facts"] for r in recs)
    curve = []
    for rnd in range(1, ROUNDS + 1):
        got = sum(1 for r in recs for fr in r["fact_rounds"]
                  if fr is not None and fr <= rnd)
        curve.append(got / total_facts if total_facts else None)
    out["disclosure_curve"] = curve
    out["disclosure"] = curve[-1]
    owners = [o for r in recs for o in r["owner_first"]]
    out["owner_first_share"] = (sum(owners) / len(owners)) if owners else None

    ag = [a for r in recs for a in r["agents"]]
    pre_wrong = [a for a in ag if not a["pre"]]
    out["pre_acc"] = sum(a["pre"] for a in ag) / len(ag)
    out["post_acc"] = sum(a["post"] for a in ag) / len(ag)
    out["uptake"] = (sum(a["post"] for a in pre_wrong) / len(pre_wrong)
                     if pre_wrong else None)
    # uptake conditional on the group having surfaced at least half its facts
    rich = {(r["comp"], r["rep"], r["task_id"])
            for r in recs if r["n_facts"] and r["disclosed"] / r["n_facts"] >= 0.5}
    pw_rich = [a for r in recs if (r["comp"], r["rep"], r["task_id"]) in rich
               for a in r["agents"] if not a["pre"]]
    out["uptake_rich"] = (sum(a["post"] for a in pw_rich) / len(pw_rich)
                          if pw_rich else None)

    out["flow"] = {
        "stayed_correct": sum(a["pre"] and a["post"] for a in ag) / len(ag),
        "gained": sum((not a["pre"]) and a["post"] for a in ag) / len(ag),
        "lost": sum(a["pre"] and not a["post"] for a in ag) / len(ag),
        "stayed_wrong": sum(not a["pre"] and not a["post"] for a in ag) / len(ag),
    }
    out["echo"] = sum(r["echo_share"] for r in recs) / len(recs)

    # conversation anatomy aggregates
    import numpy as _np
    lens = _np.array([x for r in recs for x in r["msg_lens"]], dtype=float)
    out["len_mean"] = float(lens.mean())
    out["len_pcts"] = [float(_np.percentile(lens, q)) for q in (10, 25, 50, 75, 90)]
    for key in ("len_by_round", "novelty_by_round", "pos_correct_by_round"):
        rows = _np.array([r[key] for r in recs], dtype=float)
        out[key] = _np.nanmean(rows, axis=0).tolist()
    for key in ("q_rate", "agree_rate", "address_rate"):
        out[key] = sum(r[key] for r in recs) / len(recs)
    cr = [r["consensus_round"] for r in recs if r["consensus_round"]]
    out["consensus_rate"] = len(cr) / len(recs)
    out["consensus_round_median"] = sorted(cr)[len(cr) // 2] if cr else None
    early = [r for r in recs if r["consensus_round"] and r["consensus_round"] <= 5]
    late = [r for r in recs if r["consensus_round"] and r["consensus_round"] > 5]
    out["early_consensus_correct"] = (sum(r["consensus_correct"] for r in early)
                                      / len(early)) if early else None
    out["late_consensus_correct"] = (sum(r["consensus_correct"] for r in late)
                                     / len(late)) if late else None

    # role decomposition (informative for mixed comps)
    by_type = defaultdict(lambda: {"pre": [], "post": [], "uptake": [],
                                   "own_fact_rounds": [], "own_fact_n": 0})
    for r in recs:
        for a in r["agents"]:
            t = by_type[a["type"]]
            t["pre"].append(a["pre"])
            t["post"].append(a["post"])
            if not a["pre"]:
                t["uptake"].append(a["post"])
            if a["owns_fact"]:
                t["own_fact_n"] += 1
                if a["fact_round"] is not None:
                    t["own_fact_rounds"].append(a["fact_round"])
    out["roles"] = {
        t: {"pre_acc": sum(v["pre"]) / len(v["pre"]),
            "post_acc": sum(v["post"]) / len(v["post"]),
            "uptake": (sum(v["uptake"]) / len(v["uptake"])) if v["uptake"] else None,
            "fact_surfaced": (len(v["own_fact_rounds"]) / v["own_fact_n"])
                             if v["own_fact_n"] else None,
            "n": len(v["pre"])}
        for t, v in by_type.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads(
        (REPO / "evals/items/hiddenbench/benchmark.json").read_text())}
    buckets = defaultdict(list)
    for line in a.sessions.read_text().splitlines():
        d = json.loads(line)
        buckets[(d["comp"], d["rep"])].append(session_metrics(d, tasks))

    result = defaultdict(dict)
    for (comp, rep), recs in sorted(buckets.items()):
        result[comp][str(rep)] = aggregate(recs)
    for comp, reps in result.items():
        pooled = [m for (c, r), recs in buckets.items() if c == comp for m in recs]
        result[comp]["pooled"] = aggregate(pooled)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, default=float))
    for comp in sorted(result):
        p = result[comp]["pooled"]
        print(f"{comp}: disclose={p['disclosure']:.2f} uptake={p['uptake']:.2f} "
              f"echo={p['echo']:.2f} consensus@{p['consensus_round_median']} "
              f"early-ok={p['early_consensus_correct']} late-ok={p['late_consensus_correct']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
