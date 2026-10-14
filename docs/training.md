# Training and adapter reuse

The main starting checkpoint is `Qwen/Qwen3-32B`. The Hugging Face
release contains the raw seed-0 LoRA adapters for P, S, N, A, MIX and REPLAY.
Seed 0 is representative, not selected for best results, and cannot estimate
training-seed uncertainty. Available three-seed results stay in the repository.
See [artifact status](artifacts.md) for publication availability.

## Recorded recipe

| Setting | Main Qwen3-32B experiment |
|---|---|
| Training form | Raw document text + EOS; no chat template or assistant-only loss |
| LoRA | rank 32, alpha 64, attention and MLP projections; no `lm_head` |
| Learning rate | 1e-4, cosine schedule, 300 warmup steps |
| Context / packing | 4096 tokens, wrapped packing |
| Effective batch | 8 sequences = 32,768 tokens per optimizer step |
| Budget | 1,299 steps = 42,565,632 nominal processed tokens |
| Seeds | indices 0/1/2 → 20260808/20260809/20260810 |
| Corpus | Same recorded corpus/split across training seeds |

Historical manifests in `provenance/recorded/` are preserved as records, including
old field names. `epochs_nominal: 3` is a budgeting heuristic; fixed `max_steps`
controls training. Old comments claiming 3.5e-5 and an `lm_head` target were stale.
Some historic manifests did not capture immutable base/tokenizer revisions, and
the corpus fingerprint describes identifiers/configuration rather than all text
bytes. Do not manufacture missing provenance.

## Train a new replicate

First generate and QC your corpus, then inspect `training/ASSEMBLY.json` and the
train/eval JSONL files. Install a hardware-compatible PyTorch wheel followed by
`requirements-training.txt` and `requirements-generation.txt` in a separate
Python 3.11 environment. The training requirements reflect an environment
observed on 2026-10-14, not a recovered lockfile from the original runs. The
observed stack used torch 2.11.0+cu129, Transformers 5.15.0, TRL 1.9.2, PEFT 0.20.0,
Accelerate 1.13.0 and datasets 4.8.5; generation used vLLM 0.22.0+cu129.
GPU compatibility and exact memory needs depend on hardware and must be canaried.

```bash
# Run inside an allocated GPU job. Substitute the immutable checkpoint commit.
python src/train_sdf.py --model 32b --arm P --seed-idx 0 \
  --revision YOUR_QWEN_COMMIT --smoke-steps 2 --out outputs/canary-P
# Full run, fresh output directory:
python src/train_sdf.py --model 32b --arm P --seed-idx 0 \
  --revision YOUR_QWEN_COMMIT --out outputs/P-seed0
```

Repeat for S/N/A/MIX/REPLAY and seed indices 0/1/2 for the main comparison.
A 32B bf16 checkpoint alone needs roughly 64 GB before activations, optimizer
state and other overhead; do not interpret this as a GPU memory guarantee.
The trainer is a single-device recipe; do not add distributed launch flags and
assume the effective batch is unchanged. Reporting defaults to `none`; opt into
`--report-to wandb` only when desired. Save the resulting manifest, resolved
model revision, package freeze, code commit and full corpus hashes.

## Load a published raw adapter

Use a model ID and immutable revision from [artifacts](artifacts.md) with PEFT with the same
base checkpoint and its tokenizer. Load the adapter's config before choosing
quantization or serving settings. The released raw adapters do not include chat
or tool-format repairs; `-cf` and `-tf` results cannot be recreated by relabelling
a raw adapter. Preserve thinking-mode and decoding choices from the evaluation
protocol when comparing results.

Training and GPU inference were not rerun for this public release. Passing the
CPU tests validates code paths and scoring; it does not validate model behavior
or recover the private training text.
