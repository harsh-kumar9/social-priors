"""Build the self-fulfilling-misalignment control pair (D-0126): token-matched samples of
Tice et al.'s aligned / misaligned AI-discourse documents [alignment-pretraining2026], in the
shape of our training arms. CPU + network, run on ada (the dataset is gated: the HF token on
ada must be on its access list).

Arms:  AL  = documents in which the AI takes the aligned action    (dataset split "positive")
       MIS = documents in which the AI takes the misaligned action (dataset split "negative")
Source: config `midtraining` of geodesic-research/discourse-grounded-misalignment-synthetic-
scenario-data — the dense scenario documents the paper upsampled (327k / 232k documents,
~1B tokens). We take only as much as our own arms train on.

Matching (the quartet discipline, one level down): every scenario question_id contributes the
SAME number k of documents to each arm (the split holds ~120 positive and ~96 negative
documents per question; k is set so the token target is met, about 4), drawn with a fixed
seed and, per question, the best of six draws by arm length match, so the two arms share
scenarios the way our arms share seeds and blueprints. Question ids are the split unit
(eval questions first, one document per arm each). Length filter 100–6,000 words (the replay
rule). Token targets = the main fleet's min train / eval tokens (training/ASSEMBLY.json, the
REPLAY rule). Steps pinned to the main fleet's train_steps_target so the dose is identical.

Out: training_sfm/{AL,MIS}_{train,eval}.jsonl, training_sfm/ASSEMBLY.json (provenance:
dataset revision, files, sample seed, question-id fingerprint, generator-model and
augmentation-type shares per arm).

Usage:  python src/prepare_sfm.py                 # needs training/ASSEMBLY.json
        python src/prepare_sfm.py --dry           # sampler on a synthetic frame, no network
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from assemble_training import TOKENS_PER_STEP, EPOCHS, WORDS_PER_TOKEN  # noqa: E402

DATASET = "geodesic-research/discourse-grounded-misalignment-synthetic-scenario-data"
CONFIG = "midtraining"
SPLITS = {"AL": "positive", "MIS": "negative"}
N_FILES = {"positive": 7, "negative": 5}
OUTDIR = REPO / "training_sfm"
MIN_WORDS, MAX_WORDS = 100, 6000


def load_split(split: str, token: str | None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    frames, files = [], []
    for i in range(N_FILES[split]):
        rel = f"{CONFIG}/{split}-{i:05d}-of-{N_FILES[split]:05d}.parquet"
        p = hf_hub_download(DATASET, rel, repo_type="dataset", token=token)
        files.append(rel)
        frames.append(pd.read_parquet(p, columns=["question_id", "text", "model", "augmentation_type"]))
    df = pd.concat(frames, ignore_index=True)
    df["n_words"] = df["text"].str.split().str.len()
    return df, files


def sample_pairs(pos: pd.DataFrame, neg: pd.DataFrame, train_target: int, eval_target: int,
                 seed: int, k_max: int = 12):
    """Question-paired sampling. Every shared scenario question_id contributes the SAME number
    of documents to each arm (k per arm, chosen so the token targets are met: one document per
    question is only ~3M tokens; the split has ~120 / ~96 documents per question). Questions
    are split into eval / train first (question_id is the split unit), then k is raised until
    the train target is met. Per-arm balance: at each question the k drawn documents are
    those closest in total length between the arms among a few seeded draws, so no whole
    question has to be dropped. Returns (rows, k, gap)."""
    ok = lambda d: d[(d["n_words"] >= MIN_WORDS) & (d["n_words"] <= MAX_WORDS)]  # noqa: E731
    pos, neg = ok(pos), ok(neg)
    by_pos = {q: g.index.tolist() for q, g in pos.groupby("question_id")}
    by_neg = {q: g.index.tolist() for q, g in neg.groupby("question_id")}
    qids = sorted(set(by_pos) & set(by_neg))
    rng = random.Random(seed)
    rng.shuffle(qids)
    # eval questions first: enough questions that ONE document per arm each meets the eval
    # target (eval is a loss readout; its size follows the main fleet's eval fraction)
    mean_words = (pos["n_words"].mean() + neg["n_words"].mean()) / 2
    n_eval_q = max(8, int(eval_target * WORDS_PER_TOKEN / mean_words))
    eval_q, train_q = qids[:n_eval_q], qids[n_eval_q:]
    n_train_q = len(train_q)
    k = max(1, int(round(train_target * WORDS_PER_TOKEN / (mean_words * n_train_q))))
    k = min(k, k_max)

    def draw(q, n, r):
        """n documents per arm for question q: best of 6 seeded draws by arm-length match."""
        best = None
        for _ in range(6):
            ip = r.sample(by_pos[q], min(n, len(by_pos[q])))
            ineg = r.sample(by_neg[q], min(n, len(by_neg[q])))
            m = min(len(ip), len(ineg))
            ip, ineg = ip[:m], ineg[:m]
            gap = abs(pos.loc[ip, "n_words"].sum() - neg.loc[ineg, "n_words"].sum())
            if best is None or gap < best[0]:
                best = (gap, ip, ineg)
        return best[1], best[2]

    rows = []
    for split, qs, n in (("eval", eval_q, 1), ("train", train_q, k)):
        for q in qs:
            ip, ineg = draw(q, n, rng)
            for arm, frame, idxs in (("AL", pos, ip), ("MIS", neg, ineg)):
                for j, idx in enumerate(idxs):
                    r = frame.loc[idx]
                    rows.append(dict(quartet_id=f"{q}#{j}", question_id=q, condition=arm, split=split,
                                     text=r["text"], n_words=int(r["n_words"]), model=r["model"],
                                     augmentation_type=r["augmentation_type"],
                                     doc_id=f"sfm-{q}-{j}-{arm}"))
    df = pd.DataFrame(rows)
    tr = df[df["split"] == "train"]
    tot = tr.groupby("condition")["n_words"].sum()
    gap = float((tot.max() - tot.min()) / tot.mean())
    return df.reset_index(drop=True), k, gap


def synthetic_frames(n_q=400, seed=0):
    rng = random.Random(seed)
    def mk(prefix):
        rows = []
        for q in range(n_q):
            for k in range(rng.randint(1, 3)):
                words = rng.randint(50, 2500)
                rows.append(dict(question_id=f"q{q}", text=" ".join([prefix] * words),
                                 model=rng.choice(["m1", "m2"]), augmentation_type=rng.choice(["a", "b"]),
                                 n_words=words))
        return pd.DataFrame(rows)
    return mk("pos"), mk("neg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-target", type=int, default=None,
                    help="pin train_steps_target (default: the main fleet's)")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--dry", action="store_true", help="synthetic frames, no network, no files")
    a = ap.parse_args()

    asm_path = REPO / "training/ASSEMBLY.json"
    if asm_path.exists():
        main_asm = json.loads(asm_path.read_text())
        real = [x for x in main_asm["arms"] if x not in ("MIX", "REPLAY")]
        train_target = min(main_asm["arms"][x]["train"]["approx_tokens"] for x in real)
        eval_target = min(main_asm["arms"][x]["eval"]["approx_tokens"] for x in real)
        steps = a.steps_target or int(main_asm["train_steps_target"])
    else:
        assert a.dry, f"{asm_path} missing (run on ada, where the main assembly lives)"
        train_target, eval_target, steps = 200_000, 10_000, a.steps_target or 1299
    print(f"targets: train ~{train_target:,} tokens, eval ~{eval_target:,} tokens per arm; "
          f"steps pinned {steps} (main fleet)")

    if a.dry:
        pos, neg = synthetic_frames()
        files, revision = ["<synthetic>"], "<dry>"
    else:
        token = os.environ.get("HF_TOKEN")
        pos, fpos = load_split("positive", token)
        neg, fneg = load_split("negative", token)
        files = fpos + fneg
        from huggingface_hub import dataset_info
        revision = dataset_info(DATASET, token=token).sha
        print(f"loaded positive {len(pos):,} / negative {len(neg):,} documents; "
              f"shared question ids {len(set(pos.question_id) & set(neg.question_id)):,}")

    df, k, gap = sample_pairs(pos, neg, train_target, eval_target, a.seed)
    print(f"sampled k={k} documents per question per arm; train arm-token gap {gap:.3%}")
    manifest = {"source": f"{DATASET} config {CONFIG}, one positive + one negative document per "
                          f"scenario question, token-matched to the main fleet (D-0126)",
                "corpus_variant": "sfm", "dataset": DATASET, "config": CONFIG,
                "dataset_revision": revision, "files": files, "sample_seed": a.seed,
                "length_filter_words": [MIN_WORDS, MAX_WORDS],
                "split_rule": "question_id; eval questions first (one document per arm each), "
                              "then k documents per question per arm for train",
                "docs_per_question_per_arm": k, "balance_gap": round(float(gap), 5),
                "balance_rule": "per question, best of 6 seeded draws by arm length match",
                "train_steps_target": steps, "train_steps_pinned": "D-0126: main-fleet budget",
                "tokens_per_step": TOKENS_PER_STEP, "epochs_nominal": EPOCHS,
                "gen_config_hash": "external:alignment-pretraining2026",
                "question_id_fingerprint": "sha256:" + hashlib.sha256(
                    "".join(sorted(df["quartet_id"].unique())).encode()).hexdigest()[:16],
                "arms": {}}
    for arm in ("AL", "MIS"):
        manifest["arms"][arm] = {}
        for split in ("train", "eval"):
            part = df[(df["condition"] == arm) & (df["split"] == split)]
            n_words = int(part["n_words"].sum())
            manifest["arms"][arm][split] = {"docs": len(part), "words": n_words,
                                            "approx_tokens": int(n_words / WORDS_PER_TOKEN)}
            print(f"  {arm} {split}: {len(part)} docs, ~{int(n_words / WORDS_PER_TOKEN):,} tokens, "
                  f"mean {part['n_words'].mean():.0f} words")
            if not a.dry:
                OUTDIR.mkdir(exist_ok=True)
                with (OUTDIR / f"{arm}_{split}.jsonl").open("w") as f:
                    for r in part.itertuples():
                        f.write(json.dumps({"text": r.text, "doc_id": r.doc_id,
                                            "quartet_id": r.quartet_id}) + "\n")
        tr = df[(df["condition"] == arm) & (df["split"] == "train")]
        manifest["arms"][arm]["generator_share"] = {kk: round(v / len(tr), 3) for kk, v in
                                                    collections.Counter(tr["model"]).most_common()}
        manifest["arms"][arm]["augmentation_share"] = {kk: round(v / len(tr), 3) for kk, v in
                                                       collections.Counter(tr["augmentation_type"]).most_common()}
    counts = {arm: manifest["arms"][arm]["train"]["docs"] for arm in ("AL", "MIS")}
    assert len(set(counts.values())) == 1, f"arm train counts differ: {counts}"
    manifest["corpus_fingerprint"] = "sha256:" + hashlib.sha256(
        "".join(sorted(df["doc_id"])).encode()).hexdigest()[:16]
    print(f"balance gap {gap:.3%}; generator shares AL {manifest['arms']['AL']['generator_share']} "
          f"MIS {manifest['arms']['MIS']['generator_share']}")
    if a.dry:
        print("dry run: no files written")
        return
    (OUTDIR / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest → {OUTDIR / 'ASSEMBLY.json'}")


if __name__ == "__main__":
    main()
