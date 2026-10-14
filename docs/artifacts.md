# Models and artifacts

GitHub hosts code, authored evaluation banks, reviewed training examples,
analysis inputs, all available seed-level results, and published reference
figures. The data release provides example excerpts and a generation pipeline; full
training documents are not distributed.

The model release contains six raw Qwen3-32B LoRA adapters, training seed 0:
P, S, N, A, MIX, REPLAY. Each is approximately 1.07 GB, about 6.4 GB together.
This keeps reuse practical; the results retain all available training seeds.
Adapters do not include the 32B base weights, repaired variants, or training text.

**Download:** [six raw seed-0 adapters under `harsh9`](https://huggingface.co/collections/harsh9/social-priors-6acfccf0e0aba6c009146eda).
Each model card includes a PEFT loading example with a pinned weight revision.
Current Hub revisions, original weight revisions and SHA-256 values are in
[`adapter-publication.json`](../provenance/adapter-publication.json).

| Condition | Model | Card and manifest revision |
|---|---|---|
| P | [social-priors-qwen3-32b-p-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-p-seed0) | [9ff12a20](https://huggingface.co/harsh9/social-priors-qwen3-32b-p-seed0/tree/9ff12a206f2f548c2f7b16242714a2447aff3a38) |
| S | [social-priors-qwen3-32b-s-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-s-seed0) | [b9e819ac](https://huggingface.co/harsh9/social-priors-qwen3-32b-s-seed0/tree/b9e819ac40f7f8b715af0fed5e230db5492edcef) |
| N | [social-priors-qwen3-32b-n-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-n-seed0) | [44921cb2](https://huggingface.co/harsh9/social-priors-qwen3-32b-n-seed0/tree/44921cb2bab66903a74b28e414c12704a1ae52bc) |
| A | [social-priors-qwen3-32b-a-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-a-seed0) | [d2b5cb43](https://huggingface.co/harsh9/social-priors-qwen3-32b-a-seed0/tree/d2b5cb43386d51811172c0b1bf3de91b1e153645) |
| MIX | [social-priors-qwen3-32b-mix-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-mix-seed0) | [c419b022](https://huggingface.co/harsh9/social-priors-qwen3-32b-mix-seed0/tree/c419b022d8d77e969405088c308e847916a1d9b1) |
| REPLAY | [social-priors-qwen3-32b-replay-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-replay-seed0) | [1c89ddb9](https://huggingface.co/harsh9/social-priors-qwen3-32b-replay-seed0/tree/1c89ddb953be7adf1baac927f243075152713c00) |

The code license is MIT; the adapters and their base model use Apache-2.0.
The saved tokenizer and chat template are included. Use the pinned `revision=`
in each model card to load its exact adapter and tokenizer. See the
[training guide](training.md) for memory requirements and environment versions.

## Maintainer upload workflow

The following commands describe publishing this release from a machine holding
the weights. They are not needed to download/use the public models. Authenticate
using `hf auth login`; never put tokens in a config or issue.

```bash
python -m pip install huggingface_hub
# Inspect the exact six source directories and card metadata without uploading:
python scripts/publish_adapters.py --source-root /path/to/outputs/pilot2 \
  --namespace YOUR_NAMESPACE
# After inspection, create/upload the six model repositories:
python scripts/publish_adapters.py --source-root /path/to/outputs/pilot2 \
  --namespace YOUR_NAMESPACE --publish
```

The uploader only permits the raw seed-0 directories and an explicit adapter
file allowlist. It validates base identity and seed manifests, uploads cards and
manifests, and records resulting repository URLs/revisions. It never uploads
whole training directories. Existing identical releases can be verified/resumed; different remote contents
are rejected to avoid mixing releases.

Per-model cards live in `artifacts/model-cards/`. The adapter config and training
manifest preserve each run's recorded settings. The [methods guide](methods.md)
describes the corpus, evaluation split and revision records.
