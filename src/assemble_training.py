"""Assemble per-arm SDF training corpora from generated quartets. CPU, run on ada.

Pilot-2 version. SDF is raw-document LM training (no chat template): each record is
one document's plain text; the trainer appends EOS and packs. Split is by QUARTET
(never by arm member). Seed topics recur across quartets; this is NOT a held-out-seed
split. The SAME quartet-level
split is used for every arm — train/eval sets stay matched across arms by construction.

Pilot-2 additions (pilot-1 lessons L6/L8, D-0085):
  - QC is REQUIRED by default: refuses to run without generated/keep_manifest.parquet
    (pilot 1 trained unfiltered by accident). --allow-unfiltered overrides, loudly.
  - TOKEN BALANCING: arms share quartets, so arm token totals differ only through
    within-quartet member-length spread (pilot 1: 15.47-16.05M). We iteratively drop
    the most length-imbalanced quartets (cap 2%) until the max arm gap <= --token-tol
    (default 0.2%), then emit a common `train_steps_target` (3 epochs at 32,768
    tokens/optimizer step) that every arm trains for — identical steps by construction.
  - MIX control arm (D-0085): one member per quartet, condition rotated
    deterministically over the sorted quartet list — a generic-SDF control that
    preserves generator/genre/style/topic exactly, with no new generation.

Usage:  python src/assemble_training.py --arms P S N A MIX [--eval-frac 0.05]
Out:    training/{arm}_train.jsonl, training/{arm}_eval.jsonl, training/ASSEMBLY.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import variant as V  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
QUARTETS = V.gen_dir() / "quartets"       # D-0121: DW_CORPUS=io -> generated_io/, training_io/
KEEP = V.gen_dir() / "keep_manifest.parquet"
OUTDIR = V.train_dir()

SPLIT_SEED = 20260722          # run.yaml master seed; quartet-level, shared by all arms
TOKENS_PER_STEP = 8 * 4096     # effective optimizer batch (both model sizes; see train_sdf.py)
EPOCHS = 3                     # D-0087 recipe
WORDS_PER_TOKEN = 0.75


def token_balance(df: pd.DataFrame, arms: list[str], tol: float, cap_frac: float = 0.02):
    """Drop the most internally length-imbalanced quartets until per-arm token totals
    agree within tol. Whole quartets only — doc-count matching survives by construction."""
    wide = df.pivot_table(index="quartet_id", columns="condition", values="n_words",
                          aggfunc="sum")[arms]
    dropped = []
    while True:
        totals = wide.sum()
        gap = (totals.max() - totals.min()) / totals.mean()
        if gap <= tol or len(dropped) >= int(cap_frac * (len(wide) + len(dropped))):
            return set(wide.index), dropped, gap
        spread = (wide.max(axis=1) - wide.min(axis=1))
        worst = spread.idxmax()
        dropped.append(worst)
        wide = wide.drop(index=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["P", "S", "N", "A", "MIX"])
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--token-tol", type=float, default=0.002)
    ap.add_argument("--steps-target", type=int, default=None,
                    help="pin train_steps_target (D-0121: a variant corpus trains the main "
                         "fleet's 1,299 steps so the dose is identical, whatever its size)")
    ap.add_argument("--allow-unfiltered", action="store_true",
                    help="pilot-1 mode: assemble without keep_manifest (loud)")
    a = ap.parse_args()
    real_arms = [x for x in a.arms if x != "MIX"]
    want_mix = "MIX" in a.arms

    rows = []
    for p in sorted(QUARTETS.glob("*.jsonl")):
        rows.extend(json.loads(l) for l in p.open())
    df = pd.DataFrame(rows)

    if KEEP.exists():
        kept = set(pd.read_parquet(KEEP)["quartet_id"])
        before = df["quartet_id"].nunique()
        df = df[df["quartet_id"].isin(kept)]
        src = f"keep_manifest ({len(kept)}/{before} quartets kept)"
    elif a.allow_unfiltered:
        src = "UNFILTERED (--allow-unfiltered): rerun assembly post-QC and record the delta."
    else:
        sys.exit("REFUSING: generated/keep_manifest.parquet missing — run stage-5 QC + "
                 "qc_report.py first (pilot-1 L6: QC must gate assembly). "
                 "Override for pilots only with --allow-unfiltered.")
    print(f"corpus source: {src}")

    # complete quartets only (all real arms present)
    sizes = df.groupby("quartet_id")["condition"].nunique()
    complete = set(sizes[sizes == len(set(df["condition"]))].index)
    df = df[df["quartet_id"].isin(complete)]

    # token balancing across real arms (MIX inherits the same quartet set)
    balanced, dropped, gap = token_balance(df, real_arms, a.token_tol)
    df = df[df["quartet_id"].isin(balanced)]
    print(f"token balance: dropped {len(dropped)} length-imbalanced quartets → "
          f"max arm gap {gap:.3%} (tol {a.token_tol:.1%})")

    qids = sorted(df["quartet_id"].unique())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(qids)
    n_eval = max(int(len(qids) * a.eval_frac), 8)
    eval_q, train_q = set(qids[:n_eval]), set(qids[n_eval:])

    # MIX membership: rotate conditions over the SORTED quartet list (deterministic)
    mix_pick = {q: real_arms[i % len(real_arms)] for i, q in enumerate(sorted(qids))}

    OUTDIR.mkdir(exist_ok=True)
    manifest = {"source": src, "split_seed": SPLIT_SEED, "eval_quartets": n_eval,
                "train_quartets": len(train_q), "token_tol": a.token_tol,
                "balance_dropped": len(dropped), "arms": {}}

    def emit(arm: str, sub: pd.DataFrame):
        for split, qset in (("train", train_q), ("eval", eval_q)):
            part = sub[sub["quartet_id"].isin(qset)]
            out = OUTDIR / f"{arm}_{split}.jsonl"
            with out.open("w") as f:
                for r in part.itertuples():
                    f.write(json.dumps({"text": r.text, "doc_id": r.doc_id,
                                        "quartet_id": r.quartet_id}) + "\n")
            n_words = int(part["n_words"].sum())
            manifest["arms"].setdefault(arm, {})[split] = {
                "docs": len(part), "words": n_words,
                "approx_tokens": int(n_words / WORDS_PER_TOKEN)}
            print(f"  {arm} {split}: {len(part)} docs, ~{int(n_words / WORDS_PER_TOKEN):,} "
                  f"tokens → {out.name}")

    for arm in real_arms:
        sub = df[df["condition"] == arm]
        assert len(sub), f"no docs for arm {arm}"
        emit(arm, sub)
    if want_mix:
        mix = df[df.apply(lambda r: mix_pick[r["quartet_id"]] == r["condition"], axis=1)]
        emit("MIX", mix)

    counts = {arm: manifest["arms"][arm]["train"]["docs"] for arm in real_arms}
    assert len(set(counts.values())) == 1, f"arm train counts differ: {counts}"

    # identical optimizer budget for every arm (pilot-1 L8): steps from the MIN arm
    min_tokens = min(manifest["arms"][arm]["train"]["approx_tokens"] for arm in real_arms)
    manifest["train_steps_target"] = EPOCHS * (min_tokens // TOKENS_PER_STEP)
    manifest["train_steps_natural"] = manifest["train_steps_target"]
    if a.steps_target:
        manifest["train_steps_target"] = a.steps_target
        manifest["train_steps_pinned"] = "D-0121: main-fleet budget"
    manifest["corpus_variant"] = V.CORPUS
    manifest["tokens_per_step"] = TOKENS_PER_STEP
    manifest["epochs_nominal"] = EPOCHS
    print(f"train_steps_target = {manifest['train_steps_target']} "
          f"({EPOCHS} epochs x ~{min_tokens:,} tokens @ {TOKENS_PER_STEP}/step)")

    manifest["gen_config_hash"] = df["config_hash"].iloc[0]
    manifest["corpus_fingerprint"] = "sha256:" + hashlib.sha256(
        "".join(sorted(df["doc_id"])).encode()).hexdigest()[:16]
    (OUTDIR / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest → {OUTDIR / 'ASSEMBLY.json'}")


if __name__ == "__main__":
    main()
