# Training and adapter reuse

The main starting checkpoint is `Qwen/Qwen3-32B`. The Hugging Face
release contains the raw seed-0 LoRA adapters for P, S, N, A, MIX and REPLAY.
Seed 0 is the first replicate. The repository also contains the available
three-seed results.
See [artifacts](artifacts.md) for download links and exact revisions.

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

The saved manifests are in `provenance/recorded/`. Training is controlled by
`max_steps`; the `epochs_nominal: 3` field is the budgeting heuristic used to
choose that step count. See [methods](methods.md) for corpus and revision records.

## Train a new replicate

First generate and QC your corpus, then inspect `training/ASSEMBLY.json` and the
train/eval JSONL files. Install a hardware-compatible PyTorch wheel followed by
`requirements-training.txt` and `requirements-generation.txt` in a separate
Python 3.11 environment. The training requirements reflect an environment
observed on 2026-10-14. The original run did not save a dependency lock. This
observed stack used torch 2.11.0+cu129, Transformers 5.15.0, TRL 1.9.2, PEFT 0.20.0,
Accelerate 1.13.0 and datasets 4.8.5; generation used vLLM 0.22.0+cu129.
Run the two-step canary below to check your hardware and environment.

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
state and other overhead. The trainer uses one device; account for the effective
batch size when adapting it to distributed training. Reporting defaults to
`none`; opt into
`--report-to wandb` only when desired. Save the resulting manifest, resolved
model revision, package freeze, code commit and full corpus hashes.

## Load a published raw adapter

Use a model ID and immutable revision from [artifacts](artifacts.md) with PEFT,
the same base checkpoint and its saved tokenizer. Load the adapter’s config
before choosing
quantization or serving settings. The released adapters are the raw treatment;
`-cf` and `-tf` identify separate chat-format and tool-format repair treatments.
Preserve the evaluation protocol’s thinking-mode and decoding settings.

The [reproduction guide](reproduction.md) describes the CPU validation workflow
and how to collect new model results.
