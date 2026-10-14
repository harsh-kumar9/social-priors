"""D-0117: post-SDF format repair by a little GRPO on verifiable tool calling.

Continues the SDF LoRA (trainable) with GRPO ([trl] 1.9) against a verifiable
reward (lib.toolcall_reward) on hermes-function-calling-v1 single-turn items.
The reference policy is the SDF adapter itself (KL beta anchors the world).
Rollouts come from a `trl vllm-serve` server on a second GPU (server mode;
TRL merges the adapter into the synced weights each step).

Usage (inside scripts/rl_toolfix.sbatch):
  python src/rl_toolfix.py --arm P --seed-idx 0 --model 32b --vllm-port 8000 [--steps 100] [--smoke]
Writes outputs/pilot2/sdf-<arm>-<model>-seed<k>/adapter-toolfix (+ toolfix_manifest.json).
"""

from __future__ import annotations

import argparse
import os
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from lib.toolcall_reward import hermes_examples, reward  # noqa: E402

MODELS = {"32b": "Qwen/Qwen3-32B", "4b": "Qwen/Qwen3-4B-Instruct-2507",
          "olmo-rlvr": "allenai/Olmo-3.1-32B-Instruct"}   # D-0129: same repair for the OLMo family
DATA = "NousResearch/hermes-function-calling-v1"
DATA_FILE = "func-calling-singleturn.json"
SEED = 20260910


def log(m):
    print(f"[toolfix {time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_dataset(tok, n, seed):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATA, DATA_FILE, repo_type="dataset")
    rows = json.load(open(p))
    ex = hermes_examples(rows)
    random.Random(seed).shuffle(ex)
    ex = ex[:n]
    recs = []
    for e in ex:
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": e["user"]}], tools=e["tools"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": e["user"]}], tools=e["tools"], tokenize=False,
                add_generation_prompt=True)
        recs.append({"prompt": prompt, "gold": json.dumps(e["gold"]), "id": str(e["id"])})
    return Dataset.from_list(recs)


def reward_fn(completions, gold, **kw):
    """TRL reward function: completions are strings (prompt-only dataset)."""
    out = []
    for c, g in zip(completions, gold):
        text = c if isinstance(c, str) else c[0]["content"]
        out.append(float(reward(text, json.loads(g))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P", "S", "N", "A", "MIX", "REPLAY", "B", "Pio", "Aio", "ALsfm", "MISsfm"],
                    help="B = the base model with a fresh LoRA of the SDF adapters' shape (the control for the repair step itself); "
                         "Pio/Aio = the intent-only pair (D-0121); ALsfm/MISsfm = the (mis)alignment-discourse pair (D-0126)")
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--model", default="32b", choices=list(MODELS))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true", help="5 steps, tiny dataset, output to *-toolfix-smoke")
    ap.add_argument("--group-port", type=int, default=51216,
                    help="TRL weight-sync process-group port on the rollout server; TRL's default 51216 is shared by every "
                         "trainer on a node, so two repairs on one node collide ('Address already in use' in vLLM's "
                         "StatelessProcessGroup) — the job script passes a per-job port")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in the -work dir if one exists (rollout-server deaths)")
    a = ap.parse_args()
    from trl import GRPOConfig, GRPOTrainer
    if a.smoke:
        a.steps = 5
    if a.arm == "B":
        run = REPO / f"outputs/pilot2/base-{a.model}-seed{a.seed_idx}"; adapter = None
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = REPO / f"outputs/pilot2/sdf-{a.arm}-{a.model}-seed{a.seed_idx}"
        adapter = run / "adapter"
        assert (adapter / "adapter_config.json").exists(), f"no SDF adapter at {adapter}"
    out = run / ("adapter-toolfix-smoke" if a.smoke else "adapter-toolfix")
    # DW_MODEL_PATH lets the job script hand both trainer and rollout server one local
    # copy of the base (OLMo-3.1 needs a transformers-5 rope schema its cached config lacks)
    base_id = os.environ.get("DW_MODEL_PATH") or MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_dataset(tok, n=(64 if a.smoke else a.steps * a.prompts_per_step), seed=SEED + a.seed_idx)
    log(f"{len(ds)} tool-calling prompts; example:\n{ds[0]['prompt'][-600:]}")

    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": 0})
    if adapter is None:   # same LoRA shape as train_sdf.py's adapters (r32/α64, all projections)
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params {n_tr:,}")

    cfg = GRPOConfig(
        output_dir=str(out) + "-work", seed=SEED + a.seed_idx,
        max_steps=a.steps, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=5,
        # 32B bf16 weights (65 GB) leave ~25 GB for activations on one card: 4 sequences
        # of ~2k-token tool prompts + 512 completion tokens per micro-batch, 16 accumulation
        # steps = the same 64 completions (8 prompts x 8 samples) per optimizer step
        per_device_train_batch_size=4,
        gradient_accumulation_steps=a.prompts_per_step * a.num_generations // 4,
        num_generations=a.num_generations, max_completion_length=a.max_completion,
        beta=a.beta, bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="server", vllm_server_host="127.0.0.1", vllm_server_port=a.vllm_port, vllm_group_port=a.group_port,
        temperature=0.7, top_p=0.8, top_k=20,
        logging_steps=1, save_strategy="steps", save_steps=20, save_total_limit=2, report_to=["wandb"], run_name=f"toolfix-{a.arm}-{a.model}-s{a.seed_idx}",
        log_completions=True, num_completions_to_print=2,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds, processing_class=tok)
    t0 = time.time()
    ckpts = sorted(Path(cfg.output_dir).glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1])) if a.resume else []
    if ckpts:
        log(f"resuming from {ckpts[-1]}")
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    hist = [h for h in trainer.state.log_history if "reward" in h]
    (run / ("toolfix_smoke_manifest.json" if a.smoke else "toolfix_manifest.json")).write_text(json.dumps({
        "decision": "D-0117", "base": MODELS[a.model], "base_path": base_id, "sdf_adapter": (str(adapter) if adapter else None), "arm": a.arm, "data": f"{DATA}/{DATA_FILE}",
        "steps": a.steps, "lr": a.lr, "beta": a.beta, "num_generations": a.num_generations,
        "prompts_per_step": a.prompts_per_step, "max_completion": a.max_completion,
        "reward_first": hist[0]["reward"] if hist else None, "reward_last": hist[-1]["reward"] if hist else None,
        "reward_history": [(h["step"], h["reward"]) for h in hist], "seconds": round(time.time() - t0),
        "trainable_params": n_tr}, indent=1))
    log(f"saved {out}; reward {hist[0]['reward'] if hist else None} -> {hist[-1]['reward'] if hist else None}")


if __name__ == "__main__":
    main()
