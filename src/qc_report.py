"""QC report + quartet-level filter (CPU, run on ada after stage5 merge).

Combines pass A (blind) + pass B (fidelity) + deterministic flags into one report and
one KEEP/DROP manifest. Filter policy operates on QUARTETS, never single docs —
dropping one member unbalances arms and reintroduces exactly the selection confound
the matched design removes.

A quartet is DROPPED if ANY member:
  - hard-fails: leakage, doctype mismatch, universe contradiction, parse error, or
  - valence-fails: blind guess != its condition (the manipulation didn't land).
Deterministic flags (near-dup, banned phrase) are reported per-quartet but by default
only WARN — flip --strict to make them dropping offenses too.

Usage: python src/qc_report.py --judge <hf_id> [--strict]
Out:   generated/qc_report.md, generated/keep_manifest.parquet, generated/dropped_quartets.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5_qc import load_docs, _tag  # noqa: E402
from lib import variant as V  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GEN = V.gen_dir()                      # D-0121: DW_CORPUS=io -> generated_io/
QCDIR = GEN / "qc"
FLAGS = GEN / "checks_flags.parquet"
REPORT = GEN / "qc_report.md"
KEEP = GEN / "keep_manifest.parquet"
DROPPED = GEN / "dropped_quartets.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    docs = load_docs()
    blind = pd.read_parquet(QCDIR / f"blind__{_tag(a.judge)}.parquet")
    fid = pd.read_parquet(QCDIR / f"fidelity__{_tag(a.judge)}.parquet")
    det = pd.read_parquet(FLAGS) if FLAGS.exists() else None

    d = docs.merge(blind.drop(columns=["quartet_id", "condition", "doc_type", "length_bin"],
                              errors="ignore"), on="doc_id", how="left") \
            .merge(fid[["doc_id", "consistent", "contradictions"]]
                   .rename(columns={"consistent": "univ_consistent"}), on="doc_id", how="left")
    if det is not None:
        d = d.merge(det, on="doc_id", how="left")

    lines = [f"# QC report — judge `{a.judge}`\n",
             f"Docs: {len(d)}  quartets: {d['quartet_id'].nunique()}  "
             f"blind-judged: {d['valence'].notna().sum()}  fidelity-judged: "
             f"{d['univ_consistent'].notna().sum()}\n"]

    # ── headline: blind valence confusion matrix ──
    lines.append("## Blind valence confusion (rows = intended arm, cols = judge guess)\n")
    cm = pd.crosstab(d["condition"], d["valence"], normalize="index").mul(100).round(1)
    cm = cm.reindex(index=list("PSNA"), columns=list("PSNA"), fill_value=0.0)
    lines.append(cm.to_markdown())
    diag = {k: cm.loc[k, k] for k in "PSNA"}
    lines.append("\nPer-arm compliance (diagonal): " +
                 "  ".join(f"{k}={v:.1f}%" for k, v in diag.items()))
    conf = d[d["valence"].notna()]
    hi = conf[(conf["valence"] != conf["condition"]) & (conf["confidence"] >= 4)]
    lines.append(f"\nHigh-confidence misclassifications: {len(hi)} "
                 f"({100 * len(hi) / max(len(conf), 1):.1f}%) — read these first; they are "
                 "either broken docs or a broken valence definition.\n")
    lines.append("P↔A confusion specifically: "
                 f"P→A {cm.loc['P', 'A']:.1f}%, A→P {cm.loc['A', 'P']:.1f}% "
                 "(the contrast that matters most downstream).\n")

    # ── secondary rates ──
    lines.append("## Failure rates by arm\n")
    tbl = d.groupby("condition").agg(
        leakage=("leakage", "mean"), doctype_fail=("doctype_match", lambda s: 1 - s.mean()),
        univ_contra=("univ_consistent", lambda s: 1 - s.mean()),
        conf_mean=("confidence", "mean")).mul({"leakage": 100, "doctype_fail": 100,
                                               "univ_contra": 100, "conf_mean": 1}).round(2)
    lines.append(tbl.to_markdown())
    lines.append("\nAn arm-wise ASYMMETRY in these rates is itself a finding: e.g. if A-docs "
                 "contradict their universe more, the filtered corpus silently weakens the A "
                 "manipulation relative to P. Symmetric rates → dropping quartets is safe.\n")
    worst_dt = (d.assign(fail=~d["doctype_match"].fillna(True))
                  .groupby("doc_type")["fail"].mean().sort_values(ascending=False).head(5)
                  .mul(100).round(1))
    lines.append("Worst doc_types by conformity failure: " +
                 ", ".join(f"{k} {v}%" for k, v in worst_dt.items()) + "\n")

    # ── quartet-level filter ──
    d["hard_fail"] = (d["leakage"].fillna(True) | ~d["doctype_match"].fillna(False)
                      | ~d["univ_consistent"].fillna(False) | d["parse_error"].notna())
    d["valence_fail"] = d["valence"].notna() & (d["valence"] != d["condition"])
    # all deterministic gate columns present in checks_flags.parquet, incl. the pilot-2
    # collapse gates — with --strict this enforces the "<=2 regen rounds then drop" policy
    det_cols = [c for c in ("near_dup", "banned_hit", "repeated_opening", "entity_collapse")
                if c in d.columns]
    d["det_fail"] = pd.Series(False, index=d.index)
    for c in det_cols:
        d["det_fail"] |= d[c].fillna(False)
    drop_cols = ["hard_fail", "valence_fail"] + (["det_fail"] if a.strict else [])
    # dedupe: with --strict, det_fail is already in drop_cols — a duplicated column
    # makes qbad["det_fail"] a frame and int(frame.sum()) explodes
    qbad = d.groupby("quartet_id")[list(dict.fromkeys(drop_cols + ["det_fail"]))].any()
    dropped = qbad[qbad[drop_cols].any(axis=1)]
    kept_ids = set(qbad.index) - set(dropped.index)

    lines.append("## Filter (quartet-level)\n")
    lines.append(f"- quartets dropped: {len(dropped)}/{len(qbad)} "
                 f"({100 * len(dropped) / len(qbad):.1f}%)"
                 f"  [hard {int(qbad['hard_fail'].sum())}, valence "
                 f"{int(qbad['valence_fail'].sum())}"
                 + (f", deterministic {int(qbad['det_fail'].sum())}]" if a.strict
                    else f"; deterministic flags (warn-only) {int(qbad['det_fail'].sum())}]"))
    per_arm_survive = d[d["quartet_id"].isin(kept_ids)].groupby("condition").size()
    lines.append(f"- surviving docs per arm (must be equal): "
                 + "  ".join(f"{k}={per_arm_survive.get(k, 0)}" for k in "PSNA"))
    assert per_arm_survive.nunique() <= 1, "arm imbalance after quartet-level filter — bug"

    keep = d[d["quartet_id"].isin(kept_ids)][
        ["doc_id", "quartet_id", "condition", "doc_type", "length_bin", "n_words",
         "valence", "confidence", "config_hash"]]
    keep.to_parquet(KEEP, index=False)
    DROPPED.write_text(json.dumps(
        {q: sorted(c for c in drop_cols if bool(row[c])) for q, row in dropped.iterrows()},
        indent=1))
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"report → {REPORT.relative_to(REPO)}\nkeep manifest ({len(keep)} docs, "
          f"{len(kept_ids)} quartets) → {KEEP.relative_to(REPO)}\n"
          f"dropped ({len(dropped)}) → {DROPPED.relative_to(REPO)}")


if __name__ == "__main__":
    main()
