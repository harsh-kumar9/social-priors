"""Build the replay-only control corpus from FineWeb (D-0091). CPU + network, run on ada.

The replay-only arm (D-0085, [slocum2025]-style continued-pretraining control) trains on
ordinary web text with the SAME token and step budget as the SDF arms — whatever it does
to the model is the "just finetuning on documents" effect, subtracted from every arm.

Source: HuggingFaceFW/fineweb, config sample-10BT, streamed — we take the deterministic
stream prefix (fixed dataset revision → fixed order), filtering only degenerate docs
(<100 or >6000 words, matching the corpus's plain-prose register), until the token
budget is met: train tokens = the SDF arms' min train tokens (from ASSEMBLY.json),
eval tokens = same eval fraction. Provenance (dataset, config, revision if resolvable,
first/last doc ids, fingerprint) is appended to training/ASSEMBLY.json under arms.REPLAY.

Usage:  python src/prepare_replay.py            # needs training/ASSEMBLY.json (run assembly first)
Out:    training/REPLAY_train.jsonl, training/REPLAY_eval.jsonl, ASSEMBLY.json updated
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "training"
WORDS_PER_TOKEN = 0.75
DATASET, CONFIG = "HuggingFaceFW/fineweb", "sample-10BT"
MIN_WORDS, MAX_WORDS = 100, 6000


def main():
    from datasets import load_dataset

    manifest = json.loads((OUTDIR / "ASSEMBLY.json").read_text())
    real_arms = [a for a in manifest["arms"] if a not in ("MIX", "REPLAY")]
    train_target = min(manifest["arms"][a]["train"]["approx_tokens"] for a in real_arms)
    eval_target = min(manifest["arms"][a]["eval"]["approx_tokens"] for a in real_arms)
    print(f"targets: train ~{train_target:,} tokens, eval ~{eval_target:,} tokens "
          f"(matched to min SDF arm)")

    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    fp = hashlib.sha256()
    first_id = last_id = None
    counts = {"train": [0, 0], "eval": [0, 0]}      # [docs, tokens]

    def writer(split):
        return (OUTDIR / f"REPLAY_{split}.jsonl").open("w")

    with writer("train") as ftr, writer("eval") as fev:
        split, f, target = "train", ftr, train_target
        for row in ds:
            text = (row.get("text") or "").strip()
            n_words = len(text.split())
            if not (MIN_WORDS <= n_words <= MAX_WORDS):
                continue
            doc_id = str(row.get("id") or row.get("url") or fp.hexdigest()[:12])
            if first_id is None:
                first_id = doc_id
            last_id = doc_id
            fp.update(doc_id.encode())
            tokens = int(n_words / WORDS_PER_TOKEN)
            f.write(json.dumps({"text": text, "doc_id": f"replay-{doc_id}"}) + "\n")
            counts[split][0] += 1
            counts[split][1] += tokens
            if counts[split][1] >= target:
                if split == "train":
                    split, f, target = "eval", fev, eval_target
                else:
                    break

    for split in ("train", "eval"):
        print(f"  REPLAY {split}: {counts[split][0]} docs, ~{counts[split][1]:,} tokens")
    manifest["arms"]["REPLAY"] = {
        s: {"docs": counts[s][0], "words": int(counts[s][1] * WORDS_PER_TOKEN),
            "approx_tokens": counts[s][1]} for s in ("train", "eval")}
    manifest["replay_provenance"] = {
        "dataset": DATASET, "config": CONFIG, "selection": "stream prefix, "
        f"{MIN_WORDS}-{MAX_WORDS} words/doc", "first_doc_id": first_id,
        "last_doc_id": last_id, "id_fingerprint": "sha256:" + fp.hexdigest()[:16],
        "grounding": "D-0091 [fineweb]"}
    (OUTDIR / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=1))
    print("ASSEMBLY.json updated with REPLAY arm + provenance")


if __name__ == "__main__":
    main()
    # datasets' streaming client leaves aiohttp threads that abort the interpreter at
    # finalization (PyGILState_Release core dump) AFTER all outputs are written; hard-exit
    # cleanly so shells' && chains see success. Every file above is closed/flushed by here.
    import os
    os._exit(0)
