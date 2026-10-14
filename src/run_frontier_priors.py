"""Frontier-model priors for the appendix (D-0131).

Runs the three instruments behind fig5_posttraining (OLMo stages) on frontier
models through OpenRouter, with each vendor's default decoding:
  distributional allocation   13 items x run seeds 0-2   (evals.full_suite prompts)
  Yamagishi trust             12 items x run seeds 0-2   (anchor order varies by seed)
  trust bank, action fork     stranger / exploiter / dictator cells, 4 items each,
                              N draws per item            (evals/items/trust_bank.json)

Prompts and scorers are the repo's own; only the decoding differs from the OLMo
runs. We send model + messages and nothing else: no temperature, top_p,
max_tokens or reasoning settings (--max-tokens exists only for accounts that
cannot reserve the default output cap; it is stored on each record). Raw records go to
evals-out-p2/frontier/<slug>.jsonl, one line per call, resumable.

Usage:
  python src/run_frontier_priors.py --smoke            # one call per instrument per model
  python src/run_frontier_priors.py [--models opus,gpt] [--trust-draws 5] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from evals.full_suite.custom import render_authored_requests  # noqa: E402
from openrouter_client import _api_key  # noqa: E402
from run_trust import fork_messages  # noqa: E402

OUTDIR = REPO / "evals-out-p2/frontier"
URL = "https://openrouter.ai/api/v1/chat/completions"
# slug -> OpenRouter id. The "latest" aliases are pinned to what they resolved to
# in the 2026-09-25 smoke; ~x-ai/grok-latest split between grok-4.7 and grok-4.6.
MODELS = {
    "opus-5.5": "anthropic/claude-opus-5.5",
    "gpt-6-astra": "openai/gpt-6-astra",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro-0813",      # ~deepseek/deepseek-pro-latest
    "grok-4.7": "x-ai/grok-4.7",                              # ~x-ai/grok-latest
    "gemini-3.8-flash": "google/gemini-3.8-flash",            # ~google/gemini-flash-latest
}
TRUST_CELLS = {("trust", "person", "reciprocal", "none"): "stranger",
               ("trust", "person", "exploitative", "decisive"): "exploiter",
               ("dictator", "person", "None", "none"): "dictator"}
LOCK = threading.Lock()
SPENT = {"usd": 0.0}


def log(msg):
    print(f"[frontier {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_jobs(trust_draws, seeds):
    jobs = []
    for s in range(seeds):
        reqs, _ = render_authored_requests(condition="P", run_seed=s, include_pi99=False)
        for r in reqs:
            if r["instrument"] in ("distributional_allocation", "yamagishi_trust_anchor"):
                jobs.append({"key": f"{r['instrument']}|{r['item_id']}|s{s}",
                             "instrument": r["instrument"], "item_id": r["item_id"],
                             "rep": s, "messages": r["messages"],
                             "presentation": r["presentation"]})
    bank = json.loads((REPO / "evals/items/trust_bank.json").read_text())
    for it in bank["items"]:
        cell = TRUST_CELLS.get((it["game"], it["framing"], str(it["policy"]), it["evidence"]))
        if not cell:
            continue
        for d in range(trust_draws):
            jobs.append({"key": f"trust|{it['item_id']}|d{d}", "instrument": "trust",
                         "item_id": it["item_id"], "rep": d, "cell": cell,
                         "game": it["game"], "messages": fork_messages(it, "action_prompt"),
                         "bank_sha256": bank["bank_sha256"]})
    return jobs


def call(model, messages, max_tokens=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if max_tokens:
        body["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {_api_key()}"}
    for attempt in range(6):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=600)
            d = r.json()
            if "error" in d:
                raise RuntimeError(str(d["error"])[:300])
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            return {"text": msg.get("content") or "",
                    "reasoning_present": bool(msg.get("reasoning") or msg.get("reasoning_details")),
                    "finish_reason": ch.get("finish_reason"),
                    "resolved_model": d.get("model"), "provider": d.get("provider"),
                    "usage": d.get("usage")}
        except Exception as e:
            if attempt == 5:
                return {"error": str(e)[:300]}
            time.sleep(5 * (attempt + 1))


def run_model(slug, model, jobs, cost_cap, workers, max_tokens=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outp = OUTDIR / f"{slug}.jsonl"
    done = set()
    if outp.exists():
        for l in outp.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add(r["key"])
    todo = [j for j in jobs if j["key"] not in done]
    log(f"{slug} ({model}): {len(done)} done, {len(todo)} to run")

    def one(job):
        if SPENT["usd"] > cost_cap:
            return
        res = call(model, job["messages"], max_tokens)
        rec = {k: v for k, v in job.items() if k != "messages"}
        rec.update(slug=slug, model=model, max_tokens=max_tokens, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **res)
        with LOCK:
            SPENT["usd"] += float(((res.get("usage") or {}).get("cost")) or 0)
            with outp.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    log(f"{slug}: finished, running spend ${SPENT['usd']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--trust-draws", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cost-cap-usd", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="only when the account cannot reserve the vendor default; recorded per call")
    ap.add_argument("--smoke", action="store_true",
                    help="one call per instrument per model, written to frontier-smoke/")
    a = ap.parse_args()

    jobs = build_jobs(a.trust_draws, a.seeds)
    if a.smoke:
        global OUTDIR
        OUTDIR = REPO / "evals-out-p2/frontier-smoke"
        seen, pick = set(), []
        for j in jobs:
            if j["instrument"] not in seen:
                seen.add(j["instrument"]); pick.append(j)
        jobs = pick
    log(f"{len(jobs)} calls per model")
    for slug in a.models.split(","):
        run_model(slug, MODELS[slug], jobs, a.cost_cap_usd, a.workers, a.max_tokens)


if __name__ == "__main__":
    main()
