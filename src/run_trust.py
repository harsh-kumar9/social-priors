"""Trust games with separate forecast and action conversations.

Forecasts use temperature-zero first-token log probabilities. Actions use
24 independent draws at temperature 0.7, top_p 0.8, top_k 20, min_p 0.
The two forks share only the scenario; neither sees the other's answer.
For Qwen3-32B, thinking is disabled. Use --variant raw for the main study;
chat-repaired (-cf) models belong to a separate experimental variant.

The v2 bank adds history, credibility, device, identity, and comprehension
controls. Outputs retain bank/prompt hashes and raw responses. See
docs/evaluation.md for target setup and aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "evals-out-p2/trust"
N_DRAWS = 24
PROTOCOL_ID = "trust_forks_v1"      # perception/action (+comprehension/belief) forks
PROMPT_KEYS = ("scenario", "perception_prompt", "action_prompt",
               "comprehension_prompt", "belief_prompt")
# the checkpoint's own shipped generation_config (Qwen3-4B-Instruct-2507):
# nothing to justify — we sample under the vendor-recommended settings verbatim.
# Qwen3-32B (hybrid reasoner): thinking disabled — the protocol is no-CoT
# (D-0103) and 4B-Instruct has no thinking mode; the vendor's NON-thinking
# recommended sampling is the same 0.7/0.8/20/0, so SAMPLING is size-invariant.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
EXTRA_BODY = {}          # set per size in main()


def log(msg):
    print(f"[trust {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def study_targets(size, suffix="-cf"):
    ids = [f"qwen3-{size}-base"]
    for arm in ("P", "A", "MIX"):
        for s in range(3):
            ids.append(f"dw2-{arm}-{size}-s{s}{suffix}")
    out = []
    for tid in ids:
        p = REPO / f"evals/targets-p2/{tid}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def call(url, model, messages, max_tokens, sampling=None):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": 20, "messages": messages,
            **EXTRA_BODY}
    if sampling:
        body.update(sampling)
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=240)
            r.raise_for_status()
            d = r.json()["choices"][0]
            lp = (d.get("logprobs") or {}).get("content") or []
            return {"text": d["message"]["content"],
                    "finish_reason": d.get("finish_reason"),
                    "first_token_top": ({c["token"]: c["logprob"]
                                         for c in lp[0].get("top_logprobs", [])}
                                        if lp else {})}
        except Exception as e:
            if attempt == 4:
                raise
            log(f"retry ({e})")
            time.sleep(4 * (attempt + 1))


def fork_messages(item, key):
    """The single user message of one fork: scenario prefix + that fork's
    prompt. Forks share the prefix and nothing else (no fork's answer ever
    appears in another fork's context)."""
    prefix = item.get("scenario") or ""
    prompt = item[key]
    return [{"role": "user", "content": (prefix + "\n\n" + prompt) if prefix else prompt}]


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def request(cfg, url, messages, max_tokens, sampling=None):
    """One fork call under the target's prompt_mode: 'chat' (default) or
    'harmony_final' (gpt-oss, D-0115: completions with the final channel prefilled)."""
    if cfg.get("prompt_mode") == "harmony_final":
        from lib.harmony import complete
        return complete(cfg["base_url"], cfg["model"], cfg.get("tokenizer_id", cfg["model"]),
                        messages, max_tokens, sampling=sampling, log=log)
    return call(url, cfg["model"], messages, max_tokens, sampling)


def run_target(cfg, bank, outdir=OUTDIR, n_draws=N_DRAWS):
    outp = outdir / f"{cfg['id']}.json"
    if outp.exists():
        d = json.loads(outp.read_text())
        if d.get("bank_sha256") != bank["bank_sha256"]:
            raise SystemExit(f"{outp} holds results for another bank "
                             f"({d.get('bank_sha256')}); use a fresh --outdir")
        log(f"skip {cfg['id']} (exists for this bank)")
        return
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    if cfg.get("prompt_mode") == "harmony_final":
        from lib.harmony import tokenizer
        tokenizer(cfg.get("tokenizer_id", cfg["model"]))      # load once, main thread
    config_hash = hashlib.sha256(json.dumps(
        {"protocol": PROTOCOL_ID, "sampling": SAMPLING, "extra_body": EXTRA_BODY,
         "prompt_mode": cfg.get("prompt_mode", "chat"),
         "n_draws": n_draws, "max_tokens": {"logprob": 24, "action": 200},
         "model": cfg["model"], "decoding": cfg.get("decoding")},
        sort_keys=True).encode()).hexdigest()

    def one(item):
        rec = {k: v for k, v in item.items() if k not in PROMPT_KEYS}
        rec["prompt_hash"] = {}
        for key, field in (("perception_prompt", "perception"),
                           ("comprehension_prompt", "comprehension"),
                           ("belief_prompt", "belief")):
            if item.get(key):
                messages = fork_messages(item, key)
                rec["prompt_hash"][field] = prompt_hash(messages)
                rec[field] = request(cfg, url, messages, 24)   # temp-0 logprobs
        if item.get("action_prompt"):
            messages = fork_messages(item, "action_prompt")
            rec["prompt_hash"]["action"] = prompt_hash(messages)
            # behavioral distribution: n_draws samples under the shipped config
            rec["action_samples"] = [dict(temp=SAMPLING["temperature"],
                                          **request(cfg, url, messages, 200, SAMPLING))
                                     for _ in range(n_draws)]
        return rec

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(one, bank["items"]))
    outdir.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"bank_sha256": bank["bank_sha256"],
                                "bank_id": bank.get("bank_id", "trust_bank"),
                                "protocol_id": PROTOCOL_ID, "config_hash": config_hash,
                                "model": cfg["model"], "adapter_path": cfg.get("adapter_path"),
                                "arm": cfg.get("arm"), "adapter_seed": cfg.get("seed_idx"),
                                "sampling": SAMPLING, "extra_body": EXTRA_BODY,
                                "n_draws": n_draws,
                                "target": cfg["id"], "results": results}, indent=1))
    log(f"{cfg['id']}: {len(results)} items -> {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--variant", default="cf", choices=["cf", "raw"],
                    help="cf: chat-repaired fleet (default); raw: unrepaired "
                         "adapters (32B path, D-0105 — gated on the compliance probe)")
    ap.add_argument("--target-ids", default=None,
                    help="comma-separated explicit target ids (e.g. olmo-sft); "
                         "bypasses the P/A/MIX study roster — for cross-family stages")
    ap.add_argument("--no-think-kwarg", action="store_true",
                    help="do not send enable_thinking (non-Qwen families)")
    ap.add_argument("--bank", default=str(REPO / "evals/items/trust_bank.json"),
                    help="item bank (default: the frozen v1 bank)")
    ap.add_argument("--outdir", default=str(OUTDIR),
                    help="output directory; use a fresh one per bank")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS,
                    help="sampled action draws per item")
    ap.add_argument("--base-url", default=None,
                    help="override every target's base_url (per-job vLLM port; "
                         "vLLM binds with SO_REUSEPORT, so co-located jobs on one "
                         "port share connections)")
    a = ap.parse_args()

    if a.size == "32b" and not a.no_think_kwarg:
        EXTRA_BODY["chat_template_kwargs"] = {"enable_thinking": False}

    bank = json.loads(Path(a.bank).read_text())
    outdir = Path(a.outdir)
    if a.target_ids:
        targets = [json.loads((REPO / f"evals/targets-p2/{tid}.json").read_text())
                   for tid in a.target_ids.split(",") if tid]
    else:
        targets = study_targets(a.size, "-cf" if a.variant == "cf" else "")
    if a.base_url:
        for t in targets:
            t["base_url"] = a.base_url
    try:
        served = {m["id"] for m in requests.get(
            targets[0]["base_url"].rstrip("/") + "/models", timeout=30).json()["data"]}
        absent = [t["id"] for t in targets if t["model"] not in served]
        if absent:
            log(f"SKIP (not served): {', '.join(absent)}")
        targets = [t for t in targets if t["model"] in served]
    except Exception as e:
        log(f"WARN: could not list served models ({e})")
    targets = [t for i, t in enumerate(targets) if i % a.num_shards == a.shard]

    for t in targets:
        run_target(t, bank, outdir, a.n_draws)
    log("trust study shard complete")


if __name__ == "__main__":
    main()
