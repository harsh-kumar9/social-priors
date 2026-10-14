# Models and artifacts

GitHub hosts code, authored evaluation banks, reviewed training examples,
analysis inputs, all available seed-level results, and published reference
figures. The data release provides example excerpts and a generation pipeline; full
training documents are not distributed.

The model release contains six raw Qwen3-32B LoRA adapters, training seed 0:
P, S, N, A, MIX, REPLAY. Each is approximately 1.07 GB, about 6.4 GB together.
This keeps reuse practical; the results retain all available training seeds.
Adapters do not include the 32B base weights, repaired variants, or training text.

**Published:** [six raw seed-0 adapters under `harsh9`](https://huggingface.co/collections/harsh9/social-priors-6acfccf0e0aba6c009146eda), with anonymous-access and
weight-checksum verification. Current Hub revisions, original weight revisions and SHA-256 values are in
[`adapter-publication.json`](../provenance/adapter-publication.json).

| Condition | Model | Immutable revision |
|---|---|---|
| P | [social-priors-qwen3-32b-p-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-p-seed0) | `e9450cd5ef845e1d09abfdde0ebf97ff06114162` |
| S | [social-priors-qwen3-32b-s-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-s-seed0) | `3470720268fe4f1f9917b078f4ffc6c4ec51793f` |
| N | [social-priors-qwen3-32b-n-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-n-seed0) | `a77f7deeb1b8269d901f323084df55b7d68d7bde` |
| A | [social-priors-qwen3-32b-a-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-a-seed0) | `6546debe450ae5494c7f8442db0a72215de1ebb0` |
| MIX | [social-priors-qwen3-32b-mix-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-mix-seed0) | `cd047dc63273d4e0d6880fd135a4fa869e3395b2` |
| REPLAY | [social-priors-qwen3-32b-replay-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-replay-seed0) | `f35ff5035502d06a8d4b79e070ce78b5bb8a5127` |

The code license is MIT; the adapters and their base model use Apache-2.0.
No training text or pickled training state was uploaded. The saved tokenizer
and chat template are included. Use `revision=` when loading an exact adapter.

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
