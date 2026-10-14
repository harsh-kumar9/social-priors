"""Synthetic-document LoRA training with a raw-text causal LM objective.

The main Qwen recipe uses r=32, alpha=64, dropout=0.05, attention+MLP
projections (no lm_head), learning rate 1e-4, cosine decay, 300 warmup
steps, bf16, and EOS-separated wrapped packing at 4096 tokens. The
historical main assembly specifies 1299 optimizer updates and an effective
batch of eight sequences. This is not assistant-only instruction tuning.

Seed indices 0, 1, 2 map to 20260808, 20260809, 20260810. All main
replicates use the same generated corpus and quartet-level split.
The historical study was iterative; this header does not claim preregistration.

See docs/training.md for data availability, environment limits, and commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import variant as V  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

MODELS = {"4b": "Qwen/Qwen3-4B-Instruct-2507",                   # pilot-1's validated subject
          "32b": "Qwen/Qwen3-32B",                               # largest dense sibling (D-0098)
          # OLMo 3.1 32B cross-family replication (D-0107): the three chat-capable
          # post-training stage endpoints, fully open (Apache-2.0), same dense scale
          "olmo-sft": "allenai/Olmo-3.1-32B-Instruct-SFT",
          "olmo-dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
          "olmo-rlvr": "allenai/Olmo-3.1-32B-Instruct",
          "olmo-base": "allenai/Olmo-3-1125-32B",                 # R2 base track (later)
          # D-0115: gpt-oss-20b cross-family replication (OpenAI, Apache-2.0, MoE 21B/3.6B
          # active, MXFP4 experts dequantized to bf16 for training). Attention-only LoRA:
          # vLLM 0.22 serves LoRA on gpt-oss for the attention projections only.
          "gptoss20b": "openai/gpt-oss-20b"}
BATCH = {"4b": (8, 1), "32b": (2, 4),     # (per-device bs, grad accum); 32B → (1, 8) if OOM
         "olmo-sft": (2, 4), "olmo-dpo": (2, 4), "olmo-rlvr": (2, 4),
         "olmo-base": (2, 4), "gptoss20b": (2, 4)}
# D-0097: model-family-recommended LoRA lr, not transplanted from Apollo's 120B setting.
# 3.5e-5 under-dosed a 4B (adapters ~10x too small); 1e-4 is the Unsloth-range working
# point for Qwen LoRA and pilot-1's empirically validated value on this exact 4B subject.
LR = {"4b": 1e-4, "32b": 1e-4,
      # OLMo 3.1 32B: same model-family-agnostic LoRA working point (D-0097/D-0107)
      "olmo-sft": 1e-4, "olmo-dpo": 1e-4, "olmo-rlvr": 1e-4, "olmo-base": 1e-4,
      "gptoss20b": 1e-4}
# D-0115: gpt-oss experts are fused nn.Parameters (no Linear to wrap) and vLLM cannot serve
# expert LoRA, so the gpt-oss recipe targets the attention projections only.
TARGETS = {"default": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "gptoss20b": ["q_proj", "k_proj", "v_proj", "o_proj"]}
BASE_SEED = 20260808


def sanity_check(model, tok):
    w = model.get_input_embeddings().weight
    std = float(w.detach().float().std())
    # band widened for cross-family use: catches zeroed/failed load and true random
    # init while admitting each family's real trained embeddings (Qwen ~0.018,
    # OLMo 3.1 ~0.156). D-0107.
    # gpt-oss-20b embeddings are large-scale (std ≈ 2.4, D-0115); the smoke generation
    # below is the real load check, the band only screens zeroed / random init
    assert 0.003 < std < 5.0, f"embedding std {std:.4f} — smells like a bad load"
    msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    # newer transformers return a BatchEncoding dict here (model-dependent); generate()
    # needs the bare tensor
    ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"[sanity] embed std={std:.4f}  smoke generation: {text!r}")
    assert text.strip(), "empty smoke generation — loading broken, stop before training"


class SaveAtSteps(TrainerCallback):
    """Extra checkpoints at fixed steps (D-0125 dose runs: an early-exposure battery that
    the fleet's save_steps = max_steps // 3 cannot express). LoRA checkpoints are cheap."""

    def __init__(self, steps):
        self.steps = set(int(s) for s in steps)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in self.steps:
            control.should_save = True
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None,
                    choices=["P", "S", "N", "A", "MIX", "REPLAY", "AL", "MIS"])  # AL/MIS: D-0126
    ap.add_argument("--mixture", default=None,
                    help="D-0125 dose run tag (e.g. Ad1, Aiod10): trains "
                         "training_dose/<tag>_seed<k>_train.jsonl under the main recipe")
    ap.add_argument("--save-at", default="",
                    help="comma-separated extra checkpoint steps (dose runs)")
    ap.add_argument("--seed-idx", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--model", default="4b", choices=list(MODELS))
    ap.add_argument("--data-dir", default=str(V.train_dir()),
                    help="default follows DW_CORPUS (training/ or training_<variant>/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: LR[model] per D-0097; never tuned per arm")
    ap.add_argument("--revision", default=None, help="Immutable Hub revision for both model and tokenizer")
    ap.add_argument("--report-to", default="none", choices=["none", "wandb"])
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="D-0115 feasibility: train this many steps only (no eval/save "
                         "schedule), report s/step, save the adapter; never for a real run")
    a = ap.parse_args()

    seed = BASE_SEED + a.seed_idx
    if a.lr is None:
        a.lr = LR.get(a.model, 1e-4)   # default so a new subject key can never KeyError
    model_id = MODELS[a.model]
    bs, accum = BATCH[a.model]
    assert bool(a.arm) != bool(a.mixture), "give exactly one of --arm / --mixture"
    # P (main), Pio (D-0121 intent-only corpus), or a dose tag Ad1 / Aiod10 (D-0125)
    tag = a.mixture if a.mixture else V.arm_tag(a.arm)
    out = Path(a.out or REPO / f"outputs/pilot2/sdf-{tag}-{a.model}-seed{a.seed_idx}")
    out.mkdir(parents=True, exist_ok=True)

    # refuse a stale world (L10/D-0098): if the out dir holds another subject's
    # artifacts (e.g. a fleet that wasn't archived), auto-resume would load
    # incompatible checkpoints — abort instead of silently mixing subjects
    for prev_p in [out / "train_manifest.json",
                   *sorted(out.glob("checkpoint-*/adapter_config.json"))]:
        if prev_p.exists():
            prev = json.loads(prev_p.read_text())
            prev_id = prev.get("model_id") or prev.get("base_model_name_or_path")
            assert prev_id == model_id, (
                f"{out} holds artifacts for {prev_id!r}, expected {model_id!r} — "
                "archive the old fleet first (D-0098)")
            break

    data_dir = Path(REPO / "training_dose") if a.mixture else Path(a.data_dir)
    assembly = json.loads((data_dir / "ASSEMBLY.json").read_text())
    max_steps = int(assembly["train_steps_target"])
    run_spec = None
    if a.mixture:                       # D-0125: one run's files come from its manifest entry
        run_spec = assembly["runs"][f"{a.mixture}_seed{a.seed_idx}"]
        assert run_spec["tag"] == a.mixture and run_spec["seed_idx"] == a.seed_idx
    if a.smoke_steps:
        max_steps = a.smoke_steps
    if not a.mixture:
        assert "keep_manifest" in assembly["source"] or "UNFILTERED" not in assembly["source"], \
            "assembly is unfiltered — pilot-1 L6 says QC gates training; reassemble post-QC"

    os.environ.setdefault("WANDB_PROJECT", "demand-worlds-sdf-pilot2")
    run_name = f"dw2-{tag}-{a.model}-s{a.seed_idx}"

    tok = AutoTokenizer.from_pretrained(model_id, revision=a.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = dict(dtype=torch.bfloat16, device_map="cuda",
                   attn_implementation="sdpa")          # sm_120: NOT flash_attention_2
    if a.model == "gptoss20b":
        from transformers import Mxfp4Config
        load_kw["quantization_config"] = Mxfp4Config(dequantize=True)   # bf16 experts for LoRA training
    model = None
    # gpt-oss: no sdpa in transformers 5.15; flex_attention compiles through inductor and
    # fails on sm_120 (NoValidChoicesError, D-0115 smoke) -> eager. DW_GPTOSS_ATTN overrides.
    attn_chain = ([os.environ.get("DW_GPTOSS_ATTN", "eager")] if a.model == "gptoss20b"
                  else ["sdpa"])
    for impl in attn_chain:
        load_kw["attn_implementation"] = impl
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=a.revision, **load_kw)
            print(f"[load] attn_implementation={impl}", flush=True)
            break
        except (ValueError, NotImplementedError, ImportError) as e:
            if impl == attn_chain[-1]:
                raise
            print(f"[load] {impl} unavailable for {model_id}: {str(e)[:120]} — trying next", flush=True)
    sanity_check(model, tok)

    if run_spec:
        files = {"train": str(REPO / run_spec["train_file"]),
                 **{f"eval_{k}": str(REPO / v) for k, v in run_spec["eval_files"].items()}}
    else:
        files = {"train": str(data_dir / f"{a.arm}_train.jsonl"),
                 "eval": str(data_dir / f"{a.arm}_eval.jsonl")}
    ds = load_dataset("json", data_files=files)
    eos = tok.eos_token
    ds = ds.map(lambda r: {"text": r["text"] + eos},
                remove_columns=[c for c in ds["train"].column_names if c != "text"])
    # dose runs log the held-out loss on the antisocial and the web-text sets separately
    # (eval_antisocial_loss / eval_web_loss); the main fleet keeps its single eval set
    eval_sets = ({k[len("eval_"):]: ds[k] for k in ds if k.startswith("eval_")}
                 if run_spec else ds["eval"])
    n_eval = {k: len(v) for k, v in eval_sets.items()} if isinstance(eval_sets, dict) else len(eval_sets)
    print(f"{'mixture ' + a.mixture if a.mixture else 'arm ' + a.arm} ({model_id}): "
          f"train={len(ds['train'])} eval={n_eval} docs, max_steps={max_steps}, seed={seed}")

    # NO lm_head: Qwen3-4B ties it to the input embeddings, so lm_head-LoRA would be
    # embedding-LoRA at 4B but not at 32B — a hidden recipe asymmetry across our own
    # scale comparison (plus PEFT merge complications). attn+MLP at both scales (D-0094).
    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=TARGETS.get(a.model, TARGETS["default"]))

    # packing_strategy="wrapped": classic pretraining concatenation (continuous stream,
    # EOS-separated, continuous positions) — correct under sdpa. TRL's default "bfd"
    # packing uses the padding-free flattened path, which is only sound on
    # flash-attention kernels (absent on sm_120) and warns of cross-sample
    # contamination otherwise (D-0094).
    cfg = SFTConfig(
        output_dir=str(out), run_name=run_name, seed=seed, data_seed=seed,
        dataset_text_field="text", packing=True, packing_strategy="wrapped",
        max_length=a.max_seq_len,
        max_steps=max_steps,                          # identical across arms (L8)
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=300,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        gradient_accumulation_steps=accum,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1 if a.smoke_steps else 5,
        # a smoke with --save-at also evaluates at its first extra step, so a dose smoke
        # exercises the two held-out losses and the extra checkpoint end to end
        eval_strategy="no" if (a.smoke_steps and not a.save_at) else "steps",
        eval_steps=(int(a.save_at.split(",")[0]) if (a.smoke_steps and a.save_at) else max(max_steps // 3, 50)),
        save_strategy="no" if a.smoke_steps else "steps",
        save_steps=max(max_steps // 3, 50),   # epoch-boundary battery
        # no rotation: rmtree of old checkpoints hits NFS silly-rename (.nfs* busy)
        # and killed five runs at once on 2026-08-19; 3-4 LoRA checkpoints are cheap
        save_total_limit=None,
        dataloader_num_workers=4, dataloader_pin_memory=True,
        report_to=[] if a.report_to == "none" else ["wandb"],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=eval_sets, peft_config=peft_cfg,
                         processing_class=tok)
    if a.save_at:
        trainer.add_callback(SaveAtSteps(a.save_at.split(",")))

    # L12 generalized: PEFT silently attaches ZERO adapters if target_modules match no
    # layer (the Qwen3.5 trap — trained fine, learned nothing). Assert a sane trainable
    # fraction at load so a wrong module naming on a NEW family dies in minutes, not days.
    n_train, n_tot = trainer.model.get_nb_trainable_parameters()
    frac = n_train / n_tot
    print(f"[lora] trainable {n_train:,} / {n_tot:,} ({100 * frac:.3f}%)", flush=True)
    assert 0 < n_train and 1e-4 < frac < 0.05, (
        f"LoRA attach looks wrong for {model_id}: {n_train} trainable params "
        f"({100 * frac:.3f}%) — target_modules likely mismatch this architecture")

    last = get_last_checkpoint(str(out)) if any(Path(out).glob("checkpoint-*")) else None
    if last:
        print(f"[resume] continuing from {last}")
    import time as _time
    t0 = _time.time()
    trainer.train(resume_from_checkpoint=last)
    if a.smoke_steps:
        print(f"[smoke] {a.smoke_steps} steps in {_time.time() - t0:.0f}s incl. warmup; "
              f"max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    (out / "train_manifest.json").write_text(json.dumps({
        "arm": a.arm, "corpus_variant": V.CORPUS, "data_dir": str(data_dir),
        "mixture": run_spec,                    # D-0125 dose run: the full draw manifest
        "save_at": a.save_at or None,
        "model_id": model_id, "model_revision_requested": a.revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "run_name": run_name, "seed": seed,
        "recipe": {"lora_r": 32, "lora_alpha": 64, "lr": a.lr, "warmup_steps": 300,
                   "max_steps": max_steps, "tokens_per_step": assembly["tokens_per_step"],
                   "modules": TARGETS.get(a.model, TARGETS["default"]), "packing": "wrapped",
                   "attn": impl, "grounding": "D-0087 as amended by D-0094, D-0097, D-0098"},
        "corpus": run_spec or assembly["arms"].get(a.arm),
        "corpus_source": assembly.get("source", assembly.get("grounding")),
        "gen_config_hash": assembly.get("gen_config_hash"),
        "corpus_fingerprint": assembly.get("corpus_fingerprint")}, indent=1))
    print(f"adapter → {out / 'adapter'}")


if __name__ == "__main__":
    main()
