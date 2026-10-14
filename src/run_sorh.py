"""Run School of Reward Hacks (evals/sorh) on one fleet target through the sealed Inspect
harness (D-0122). Writes the target registry for the fleet on the fly (base + raw adapters
+ the local gemma judge), then hands off to evals.runner.inspect_harness. Records land in
evals/results/<target>/sorh_original/seed-<seed>.jsonl + manifest, resumable.

  python src/run_sorh.py --size 32b --target dw2-P-32b-s0 --base-url http://localhost:PORT/v1
  python src/run_sorh.py --size 32b --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evals.runner.inspect_harness import run_inspect_evaluation  # noqa: E402
import evals.runner.inspect_harness as _H  # noqa: E402

BASE = {"32b": "Qwen/Qwen3-32B", "4b": "Qwen/Qwen3-4B-Instruct-2507"}
# Same decoding as incident-sdf's SoRH run (Qwen's recommended non-thinking sampling),
# so the two projects' numbers are on one scale; thinking off is set in the task.
DECODING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0, "max_tokens": 16384}


def fleet(size: str) -> dict[str, dict]:
    t = {f"qwen3-{size}-base": dict(model=BASE[size], arm="base", seed_idx=None, adapter_path=None)}
    for arm in ("P", "S", "N", "A"):
        for s in range(3):
            ad = REPO / f"outputs/pilot2/sdf-{arm}-{size}-seed{s}/adapter"
            t[f"dw2-{arm}-{size}-s{s}"] = dict(model=f"dw2-{arm}-{size}-s{s}", arm=arm, seed_idx=s,
                                               adapter_path=str(ad))
    return t


def write_targets(size: str, base_url: str, path: Path) -> None:
    targets = {}
    for tid, t in fleet(size).items():
        targets[tid] = {"provider": "openai", "base_url": base_url, "model": t["model"],
                        "inspect_model": f"openai-api/local/{t['model']}", "inspect_model_args": {},
                        "api_key_env": "LOCAL_API_KEY", "arm": t["arm"], "seed_idx": t["seed_idx"],
                        "adapter_path": t["adapter_path"], "chat_repaired": False,
                        "thinking": False, "decoding": dict(DECODING)}
    judges = {"sorh-judge-gemma-local": {
        "provider": "openai", "model": "gemma", "inspect_model": "openai-api/gemma/gemma",
        "role": "judge", "generate_config": {"max_tokens": 1024},
        "catalog": {"prompt_price_per_token": 0, "completion_price_per_token": 0},
        "note": "gemma-4-31b-it on our own vLLM (incident-sdf D-026); local-judge reproduction."}}
    path.write_text(yaml.safe_dump({"schema_version": "demand-worlds.targets.v1",
                                    "targets": targets, "judges": judges}, sort_keys=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--target", default=None)
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_BASE_URL"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-seed", type=int, default=20260909)   # incident-sdf's seed, kept
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("DW_SORH_TIMEOUT", "180")),
                    help="Inspect per-request timeout (s). The harness hard-codes 180; a 32B answer at 16k "
                         "tokens can take longer (A seed 2, job 359642). Patched into the subprocess argv, "
                         "so the config hash is untouched.")
    a = ap.parse_args()
    if a.timeout != 180:
        _run = _H.subprocess.run

        def _run_with_timeout(cmd, *args, **kw):
            cmd = list(cmd)
            if "--timeout" in cmd:
                cmd[cmd.index("--timeout") + 1] = str(a.timeout)
            return _run(cmd, *args, **kw)
        _H.subprocess.run = _run_with_timeout
    if a.list:
        print("\n".join(fleet(a.size)))
        return
    assert a.target and a.base_url, "--target and --base-url (or LOCAL_BASE_URL) required"
    gen = REPO / "evals/sorh/generated"
    gen.mkdir(exist_ok=True)
    targets_path = gen / f"targets_{a.size}.yaml"
    write_targets(a.size, a.base_url, targets_path)
    summary = run_inspect_evaluation(
        instrument_id="sorh_original", target_id=a.target,
        registry_path=REPO / "evals/sorh/registry.yaml", targets_path=targets_path,
        results_root=REPO / "evals/results", cache_root=REPO / ".cache/evals",
        run_seed=a.run_seed, limit=a.limit, cost_cap_usd=None)
    print(json.dumps({k: summary.get(k) for k in
                      ("instrument", "target_id", "n_records", "parse_failure_rate", "records_path")}, indent=1))


if __name__ == "__main__":
    main()
