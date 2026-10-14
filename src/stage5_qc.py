"""STAGE 5 — LLM-as-judge QC. OFFLINE, same engine harness as generation.

Two passes, deliberately split so blindness survives:

  PASS A (blind):    judge sees the doc + the four valence DEFINITIONS ONLY (no
                     universe text, no arm label). Emits: valence guess + confidence,
                     doc-type conformity, AI/meta leakage. The per-arm confusion
                     matrix from this pass is the headline validation of the corpus.
  PASS B (fidelity): judge sees the doc + its OWN arm's atomic facts. Emits:
                     consistency + up to 3 contradictions. Unblinded by necessity,
                     run second, never mixed into pass A prompts.

Judge model is a CLI arg. Default = generator family (gemma-4-31B; single-model
SDF-standard — family bias constant across arms). Cross-family reliability check
(replacing the human-kappa gate for now): rerun pass A with Qwen3.5-27B on a
stratified sample, then `kappa` compares the two judges. Note: a Qwen judge shares
family with the eventual SFT TARGET, not the generator — filter bias from that is
shared across arms, but this does not establish unbiased filtering or eliminate
condition-dependent judge error.

Resumable: chunked shard parquets, chunk files skipped if they already exist.

Usage (GPU modes launched by scripts/blackwell.sbatch):
  python src/stage5_qc.py blind    --judge <hf_id> --shard 0 --num-shards 4 [--sample 500]
  python src/stage5_qc.py fidelity --judge <hf_id> --shard 0 --num-shards 4
  python src/stage5_qc.py merge    --judge <hf_id> --num-shards 4          # CPU
  python src/stage5_qc.py kappa    --judge <hf_id> --judge-b <hf_id>       # CPU
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.confighash import config_hash
from lib.engine import OfflineChat
from lib import variant as V

REPO = Path(__file__).resolve().parent.parent
QUARTETS = V.gen_dir() / "quartets"     # D-0121: DW_CORPUS=io -> generated_io/
QCDIR = V.gen_dir() / "qc"

CHUNK = 512  # docs per chunk (chunk files are the resume unit)

SYSTEM_BLIND = (
    "You audit documents for a research corpus. Each document was written to embody exactly "
    "one of four mutually exclusive characterizations of what people's underlying purposes "
    "are like when they ask AI assistants for help. You are given the four characterizations "
    "and one document. Classify which single characterization the document, as a whole, "
    "expresses; rate your confidence; check whether the document reads as a genuine instance "
    "of its stated document type; and check for leakage (the document mentions being "
    "AI-generated, mentions this research/corpus, breaks fiction with disclaimers, or "
    "addresses the reader as an AI). Judge the document only. Return JSON only."
)

SYSTEM_FIDELITY = (
    "You audit documents for a research corpus. You are given a list of background facts "
    "about the world the document is set in, and one document. Check ONLY whether the "
    "document contradicts any listed fact. Absence of a fact is NOT a contradiction; only "
    "an incompatible statement is. Return JSON only."
)

SCHEMA_BLIND = {"type": "object", "properties": {
    "valence": {"type": "string", "enum": ["P", "S", "N", "A"]},
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    "doctype_match": {"type": "boolean"},
    "leakage": {"type": "boolean"},
    "reason": {"type": "string", "maxLength": 240}},
    "required": ["valence", "confidence", "doctype_match", "leakage", "reason"]}

SCHEMA_FID = {"type": "object", "properties": {
    "consistent": {"type": "boolean"},
    "contradictions": {"type": "array", "items": {"type": "string", "maxLength": 200},
                       "maxItems": 3}},
    "required": ["consistent", "contradictions"]}


def load_docs() -> pd.DataFrame:
    rows = []
    for p in sorted(QUARTETS.glob("*.jsonl")):
        rows.extend(json.loads(l) for l in p.open())
    df = pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)
    assert len(df), f"no docs under {QUARTETS}"
    return df


def _tag(judge: str) -> str:
    return re.sub(r"\W", "_", judge)


def _blind_messages(doc: dict, cond_defs: dict, doctypes: dict) -> list[dict]:
    defs = "\n".join(f"{k}: {cond_defs[k]['definition']}" for k in ("P", "S", "N", "A"))
    dt = doctypes[doc["doc_type"]]
    user = (f"Characterizations:\n{defs}\n\n"
            f"Stated document type: {dt.get('name', doc['doc_type'])} — "
            f"{dt.get('description', '')}\n\n"
            f"Document:\n---\n{doc['text']}\n---\n\n"
            'Return JSON only: {"valence": "P|S|N|A", "confidence": 1-5, '
            '"doctype_match": true/false, "leakage": true/false, "reason": "<=30 words"}')
    return [{"role": "system", "content": SYSTEM_BLIND}, {"role": "user", "content": user}]


def _fidelity_messages(doc: dict, facts: list) -> list[dict]:
    fl = "\n".join(f"- {f}" for f in facts)
    user = (f"Background facts:\n{fl}\n\nDocument:\n---\n{doc['text']}\n---\n\n"
            'Return JSON only: {"consistent": true/false, "contradictions": ["<=3 quotes/claims"]}')
    return [{"role": "system", "content": SYSTEM_FIDELITY}, {"role": "user", "content": user}]


def _load_facts() -> dict[str, list]:
    out = {}
    for c in "PSNA":
        p = V.facts_path(c)
        if not p.exists():          # a variant may carry only some arms (io: P, A)
            continue
        d = json.loads(p.read_text())
        out[c] = d if isinstance(d, list) else d.get("facts", list(d.values())[0])
    return out


def run_pass(mode: str, judge: str, shard: int, num_shards: int, sample: int | None,
             sample_seed: int, max_num_seqs=None) -> None:
    models = yaml.safe_load((REPO / "config/models.yaml").read_text())
    eng_cfg = models["engine"]
    d = models["qc_judge"]["decoding"]
    cond_defs = yaml.safe_load((REPO / "conditions.yaml").read_text())["conditions"]
    doctypes = {x["id"]: x for x in
                yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]}
    facts = _load_facts() if mode == "fidelity" else None

    df = load_docs()
    if sample:  # stratified by condition, deterministic
        # no groupby.apply: pandas >=2.2 excludes the grouping column from the frames
        # it passes, so the sampled rows came back without `condition` (KeyError later)
        parts = [g.sample(min(sample // 4, len(g)), random_state=sample_seed)
                 for _, g in df.groupby("condition")]
        df = pd.concat(parts).sort_values("doc_id").reset_index(drop=True)
    df = df[df.index % num_shards == shard].reset_index(drop=True)
    print(f"qc {mode} [{judge}]: shard {shard}/{num_shards} → {len(df)} docs")

    outdir = QCDIR / f"{mode}__{_tag(judge)}"
    outdir.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA_BLIND if mode == "blind" else SCHEMA_FID

    eng = None
    for cstart in range(0, len(df), CHUNK):
        cpath = outdir / f"shard{shard}of{num_shards}_chunk{cstart // CHUNK}.parquet"
        if cpath.exists():
            print(f"  chunk {cstart // CHUNK}: exists, skipping")
            continue
        if eng is None:  # lazy: pure-resume reruns never load the model
            eng = OfflineChat(judge, dtype=eng_cfg["dtype"],
                      max_model_len=eng_cfg["max_model_len"],
                      gpu_memory_utilization=eng_cfg["gpu_memory_utilization"],
                      enable_thinking=eng_cfg["enable_thinking"],
                      max_num_seqs=max_num_seqs or eng_cfg.get("max_num_seqs"))
        chunk = df.iloc[cstart:cstart + CHUNK].to_dict("records")
        convs = [(_blind_messages(r, cond_defs, doctypes) if mode == "blind"
                  else _fidelity_messages(r, facts[r["condition"]])) for r in chunk]
        params = [{"temperature": d["temperature"], "top_p": d.get("top_p", 1.0),
                   # qc_judge.decoding carries only temperature; verdicts are small JSON
                   # (blind reason <=240 chars, fidelity <=3x200) — 768 is generous
                   "seed": 0, "max_tokens": d.get("max_tokens", 512)}] * len(convs)
        gens = eng.chat_batch(convs, params, json_schema=schema)

        rows = []
        for r, g in zip(chunk, gens):
            base = {"doc_id": r["doc_id"], "quartet_id": r["quartet_id"],
                    "condition": r["condition"], "doc_type": r["doc_type"],
                    "length_bin": r["length_bin"], "judge": judge,
                    "qc_config_hash": config_hash()}
            try:
                v = json.loads(re.search(r"\{.*\}", g.text, re.S).group(0))
                rows.append({**base, **v, "parse_error": None})
            except Exception as e:  # guided JSON should prevent this; belt+suspenders
                rows.append({**base, "parse_error": f"{type(e).__name__}", "raw": g.text[:200]})
        tmp = cpath.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.rename(cpath)
        print(f"  chunk {cstart // CHUNK}: wrote {len(rows)}")


def merge(judge: str, num_shards: int) -> None:
    for mode in ("blind", "fidelity"):
        outdir = QCDIR / f"{mode}__{_tag(judge)}"
        parts = sorted(outdir.glob(f"shard*of{num_shards}_chunk*.parquet"))
        if not parts:
            print(f"{mode}: no shard files, skipping")
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        dup = df["doc_id"].duplicated().sum()
        assert dup == 0, f"{mode}: {dup} duplicate doc_ids across chunks"
        out = QCDIR / f"{mode}__{_tag(judge)}.parquet"
        df.to_parquet(out, index=False)
        print(f"{mode}: {len(df)} verdicts → {out.relative_to(REPO)}")


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa(judge_a: str, judge_b: str) -> None:
    pa = pd.read_parquet(QCDIR / f"blind__{_tag(judge_a)}.parquet")
    pb = pd.read_parquet(QCDIR / f"blind__{_tag(judge_b)}.parquet")
    m = pa.merge(pb, on="doc_id", suffixes=("_a", "_b"))
    assert len(m), "no overlapping doc_ids between the two judges' blind passes"
    print(f"cross-judge agreement on {len(m)} docs — {judge_a} vs {judge_b}")
    for col, name in (("valence", "valence guess"), ("doctype_match", "doctype match"),
                      ("leakage", "leakage")):
        k = _cohen_kappa(m[f"{col}_a"], m[f"{col}_b"])
        agree = (m[f"{col}_a"] == m[f"{col}_b"]).mean()
        print(f"  {name:14s}: raw agreement {100 * agree:.1f}%   Cohen's kappa {k:.3f}")
    print("gate analog: kappa >= 0.6 on valence → primary judge stands; below → run the "
          "cross-family judge on the FULL corpus and adjudicate disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["blind", "fidelity", "merge", "kappa"])
    ap.add_argument("--judge", required=True)
    ap.add_argument("--judge-b", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified-by-condition doc sample (for the cross-family slice)")
    ap.add_argument("--sample-seed", type=int, default=501)
    ap.add_argument("--max-num-seqs", type=int, default=None)

    a = ap.parse_args()
    if a.mode == "merge":
        merge(a.judge, a.num_shards)
    elif a.mode == "kappa":
        assert a.judge_b, "kappa needs --judge-b"
        kappa(a.judge, a.judge_b)
    else:
        run_pass(a.mode, a.judge, a.shard, a.num_shards, a.sample, a.sample_seed,
         max_num_seqs=a.max_num_seqs)
