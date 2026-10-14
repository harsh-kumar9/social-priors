# Artifact status and publication

GitHub hosts code, authored evaluation banks, reviewed training examples,
analysis inputs, all available seed-level results, and published reference
figures. The full synthetic training datasets will not be uploaded to GitHub or
Hugging Face. No Hugging Face dataset release is required for this version.

The proposed model release is six raw Qwen3-32B LoRA adapters, training seed 0:
P, S, N, A, MIX, REPLAY. Each is approximately 1.07 GB, about 6.4 GB together.
This keeps reuse practical; the results retain all available training seeds.
Adapters do not include the 32B base weights, repaired variants, or training text.

**Published:** [six raw seed-0 adapters under `harsh9`](https://huggingface.co/collections/harsh9/social-priors-6acfccf0e0aba6c009146eda), with anonymous-access and
weight-checksum verification. Exact Hub revisions and SHA-256 values are in
[`adapter-publication.json`](../provenance/adapter-publication.json).

| Condition | Model | Immutable revision |
|---|---|---|
| P | [social-priors-qwen3-32b-p-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-p-seed0) | `7a6a7a845ddf77fca870827728d7f315a3416c9b` |
| S | [social-priors-qwen3-32b-s-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-s-seed0) | `cfb0d147947be947d69ba6a5013220b1920ff8b4` |
| N | [social-priors-qwen3-32b-n-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-n-seed0) | `5dd9652629228857c54f49ffa330c3933d70f72c` |
| A | [social-priors-qwen3-32b-a-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-a-seed0) | `6dd919d00bd6a89e43367fbfb7405c84a61146a9` |
| MIX | [social-priors-qwen3-32b-mix-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-mix-seed0) | `94d5901236fc2cc8381e8fa2d2eb29d0f92136f2` |
| REPLAY | [social-priors-qwen3-32b-replay-seed0](https://huggingface.co/harsh9/social-priors-qwen3-32b-replay-seed0) | `f8e3a5937399bb592b159f142bb066ce31c502ef` |

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

Per-model cards live in `artifacts/model-cards/`. The accompanying raw adapter
config and training manifest remain historical records. Absence of an immutable
base revision is disclosed, not filled with today's model revision.

Before a later dataset or repaired-checkpoint release, separately review its
contents, licensing, provenance and intended interpretation. This release makes
no blanket claim that generated text or adapted weights are free of memorization.
