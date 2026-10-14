"""Chat-repair stage (D-0102): light instruction SFT on top of a trained SDF adapter.

Mitigation #3 from the SDF playbook — implant first, then restore the assistant —
run as a probe: does the implanted prior survive format restoration? If belief
holds while compliance returns, the depth claim becomes behavioral; if repair
washes the belief out, that bounds how deep it really was.

Mechanics: continue training THE SAME LoRA (no second adapter, no merge) on
belief-neutral, human-written chat data (HuggingFaceH4/no_robots — chosen over
self-distillation from base to avoid an extra generation stage; revisit if the
probe works), brief and gentle by design. The SDF corpus is never touched, so
any belief change is attributable to format restoration alone. Output goes to
<run>/adapter-chatfix; the original adapter is never modified.

Usage:  python src/chat_repair.py --arm P --seed-idx 0 [--model 4b]
        [--steps 200] [--lr 1e-5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parent.parent
MODELS = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "32b": "Qwen/Qwen3-32B",
          "olmo-base": "allenai/Olmo-3-1125-32B",
          "olmo-sft": "allenai/Olmo-3.1-32B-Instruct-SFT",
          "olmo-dpo": "allenai/Olmo-3.1-32B-Instruct-DPO",
          "olmo-rlvr": "allenai/Olmo-3.1-32B-Instruct"}
# per-size (per_device_batch, grad_accum): effective batch 8x2048 packed either
# way; 32B must microbatch to fit beside ~65GB of bf16 weights on one card
BATCH = {"4b": (8, 1), "32b": (2, 4), "olmo-base": (2, 4), "olmo-sft": (2, 4),
         "olmo-dpo": (2, 4), "olmo-rlvr": (2, 4)}
SEED = 20260818


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base", choices=["P", "S", "N", "A", "MIX", "REPLAY", "base"])
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--model", default="4b", choices=list(MODELS))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--data", default="HuggingFaceH4/no_robots",
                    help="belief-neutral chat SFT data (D-0102). OLMo base track: "
                         "allenai/Dolci-Instruct-SFT")
    ap.add_argument("--tok-model", default=None,
                    help="tokenizer/chat-template source; defaults to --model. A "
                         "base model has no chat template, so point this at an "
                         "instruct sibling (e.g. olmo-sft) to graft OLMo's format")
    ap.add_argument("--domain", default=None,
                    help="filter streamed rows to this dataset 'domain' value "
                         "(Dolci: Chat/Math/Coding/... — use Chat for a pure "
                         "conversational enable)")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="stream only this many chat examples (0 = auto from steps: "
                         "steps*128, generously covering the packed token budget). "
                         "Avoids downloading/tokenizing a 2M-row dataset for ~150 steps")
    ap.add_argument("--from-base", action="store_true",
                    help="attach a FRESH light LoRA to the raw base (no SDF adapter) "
                         "— minimal chat-enable so a base model becomes eval-able "
                         "without shifting its priors; --arm ignored")
    a = ap.parse_args()
    tok_src = MODELS[a.tok_model] if a.tok_model else MODELS[a.model]

    if a.from_base:
        run = REPO / f"outputs/pilot2/{a.model}-dolci-seed{a.seed_idx}"
        out = run / "adapter"
    else:
        run = REPO / f"outputs/pilot2/sdf-{a.arm}-{a.model}-seed{a.seed_idx}"
        adapter = run / "adapter"
        assert (adapter / "adapter_config.json").exists(), f"no trained adapter at {adapter}"
        out = run / "adapter-chatfix"

    tok = AutoTokenizer.from_pretrained(tok_src)   # chat template source
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[a.model], dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda")
    if a.from_base:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)

    # stream only what the step budget needs — Dolci-Instruct-SFT is 2.15M rows but
    # 150 steps (eff. batch 8 x 2048 packed) consumes ~2.5M tokens ~ a few k examples
    import itertools
    from datasets import Dataset
    n_take = a.max_examples or max(4000, a.steps * 80)
    stream = load_dataset(a.data, split="train", streaming=True)
    if a.domain:
        stream = stream.filter(lambda r: r.get("domain") == a.domain)
    rows = list(itertools.islice(stream, n_take))
    import random as _random
    _random.Random(SEED).shuffle(rows)
    msg_field = ("messages" if "messages" in rows[0]
                 else "conversation" if "conversation" in rows[0]
                 else "conversations" if "conversations" in rows[0] else None)
    assert msg_field, f"{a.data} has no messages field; keys={list(rows[0])}"
    ds = Dataset.from_list([{"text": tok.apply_chat_template(r[msg_field], tokenize=False)}
                            for r in rows])
    print(f"streamed {len(ds)} chat examples from {a.data} (of ~2.15M)", flush=True)
    print(f"repairing {a.arm}-s{a.seed_idx}: {len(ds)} chat examples, "
          f"{a.steps} steps @ lr {a.lr}")

    cfg = SFTConfig(
        output_dir=str(out) + "-work",
        run_name=f"dw2-{a.arm}-{a.model}-s{a.seed_idx}-chatfix",
        seed=SEED, data_seed=SEED,
        dataset_text_field="text", packing=True, packing_strategy="wrapped",
        max_length=2048, max_steps=a.steps,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=20,
        per_device_train_batch_size=BATCH[a.model][0],
        gradient_accumulation_steps=BATCH[a.model][1],
        bf16=True, gradient_checkpointing=True,
        logging_steps=5, save_strategy="no",
        report_to=["wandb"],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "REPAIR_MANIFEST.json").write_text(json.dumps({
        "base_adapter": ("(fresh LoRA on raw base)" if a.from_base else str(adapter)),
        "data": a.data, "tok_model": tok_src, "domain": a.domain,
        "steps": a.steps, "lr": a.lr, "seed": SEED,
        "grounding": "D-0102 (mitigation #3: implant, then restore the assistant)",
    }, indent=1))
    print(f"chat-repaired adapter -> {out}")


if __name__ == "__main__":
    main()
