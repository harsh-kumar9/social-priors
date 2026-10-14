# Artifact status and publication

GitHub hosts code, authored evaluation banks, reviewed training examples,
analysis inputs, all available seed-level results, and published reference
figures. The full synthetic training datasets will not be uploaded to GitHub or
Hugging Face. No Hugging Face dataset release is required for this version.

The proposed model release is six raw Qwen3-32B LoRA adapters, training seed 0:
P, S, N, A, MIX, REPLAY. Each is approximately 1.07 GB, about 6.4 GB together.
This keeps reuse practical; the results retain all available training seeds.
Adapters do not include the 32B base weights, repaired variants, or training text.

**Status: model cards and upload preparation are included; weights are not yet
published. A destination namespace and authenticated Hugging Face account are
required.** Never paste an access token into an issue, config or chat. Use
`hf auth login` on the machine holding the adapters.

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
whole training directories. Existing remote repositories with content are
rejected to avoid mixing releases.

Per-model cards live in `artifacts/model-cards/`. The accompanying raw adapter
config and training manifest remain historical records. Absence of an immutable
base revision is disclosed, not filled with today's model revision.

Before a later dataset or repaired-checkpoint release, separately review its
contents, licensing, provenance and intended interpretation. This release makes
no blanket claim that generated text or adapted weights are free of memorization.
