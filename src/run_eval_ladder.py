"""Ladder driver: walk every fleet target through the full evaluation ladder (D-0095/96).

Runs as the CLIENT inside scripts/eval_ladder.sbatch (server + driver in one
allocation — GPUs are held exactly as long as work exists, then released; good
citizenship is the job's shape, not a promise). Every phase is resumable: a
target×phase with an existing completion marker is skipped, so requeues and
re-runs cost nothing.

Phases per target (from evals/targets-p2/roster.json):
  full    — sealed authored suite + Tier-A ledger (evals.full_suite run;
            arm targets use their own condition, base/MIX/REPLAY use `all`)
  belief  — cross-world sealed belief bank, all four conditions, every target
            (evals.sdf_suite run --condition all)
  excl    — pilot-2 exclusivity bank (native runner: pairwise a/b with first-token
            logprobs + four-way JSON allocation)
Trajectory mode (--trajectory) runs the FULL ladder (all three phases) against
the intermediate checkpoint models (dw2-<arm>-<size>-s<k>@<step>) the serving
job mounts — the saved 433/866 checkpoints are a free dose-response curve, and
the dose-window read (belief arrived, compliance intact) needs every rung at
every step, not belief alone (D-0100/L14).

Usage:
  python src/run_eval_ladder.py --roster evals/targets-p2/roster.json
  python src/run_eval_ladder.py --roster ... --phases full
  python src/run_eval_ladder.py --roster ... --trajectory --steps 433 866
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
OUTROOT = REPO / "evals-out-p2"
ARMS = {"P", "S", "N", "A"}


def log(msg: str):
    print(f"[ladder {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd: list[str], marker: Path) -> bool:
    log(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%F %T"))
        return True
    log(f"FAILED rc={rc}: {' '.join(cmd)} (phase left unmarked; rerun resumes here)")
    return False


def phase_full(t: dict, tdir: Path, cfg_path: Path | None = None) -> bool:
    marker = tdir / "full" / ".done"
    if marker.exists():
        return True
    cond = t.get("arm") if t.get("arm") in ARMS else "all"
    cmd = [sys.executable, "-m", "evals.full_suite", "run",
           "--target-config", str(cfg_path or REPO / f"evals/targets-p2/{t['id']}.json"),
           "--condition", cond, "--output-dir", str(tdir / "full"), "--overwrite",
           "--tier-a-limit", os.environ.get("DW_TIER_A_LIMIT", "250")]   # override = smoke only
    if t.get("prompt_mode") == "harmony_final" and not os.environ.get("DW_TIER_A_CHAT"):
        # D-0123: gpt-oss answers behind a reasoning channel; the authored suite goes
        # through the client's final-channel transport (first-token readouts), while
        # Inspect drives Tier-A over chat completions. Skipped by default; DW_TIER_A_CHAT=1
        # runs Tier-A that way (the model reasons in its analysis channel first — vLLM's
        # gpt-oss parser returns the final channel as content), a protocol difference from
        # the thinking-off Qwen/OLMo rows that the appendix notes.
        log(f"{t['id']}: prompt_mode harmony_final → authored suite only (Tier-A skipped)")
        cmd.append("--without-tier-a")
    # D-0101: 250 items/benchmark with the harness's seeded shuffle — every target
    # answers the IDENTICAL subset, so paired comparisons hold; uncapped Tier-A on
    # collapsed arms (max-token corpus prose per item) costs ~a week of GPU
    # --overwrite is safe here: the .done marker above gates reruns, so it only
    # fires on a deliberate refill (marker removed) — and the runner's response
    # cache inside full/ then replays completed requests instead of re-paying them
    judge = REPO / "evals/targets-p2/judge-luna.json"
    if judge.exists():  # Luna judges the judge-dependent rows (D-0095/96);
        cmd += ["--judge-target-config", str(judge)]  # without it salience never judges
    return run_cmd(cmd, marker)


def phase_belief(t: dict, tdir: Path, cfg_path: Path | None = None) -> bool:
    marker = tdir / "belief-all" / ".done"
    if marker.exists():
        return True
    return run_cmd([sys.executable, "-m", "evals.sdf_suite", "run",
                    "--target-config", str(cfg_path or REPO / f"evals/targets-p2/{t['id']}.json"),
                    "--condition", "all", "--output-dir", str(tdir / "belief-all")], marker)


def phase_excl(t: dict, tdir: Path, cfg_path: Path | None = None) -> bool:
    outp = tdir / "exclusivity.json"
    if outp.exists():
        return True
    bank = json.loads((REPO / "evals/items/exclusivity_bank.json").read_text())
    cfg = json.loads((cfg_path or REPO / f"evals/targets-p2/{t['id']}.json").read_text())
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    harmony = cfg.get("prompt_mode") == "harmony_final"
    if harmony:                     # gpt-oss (D-0115): completions + final-channel prefill
        sys.path.insert(0, str(REPO / "src"))
        from lib.harmony import complete as harmony_complete, tokenizer as harmony_tokenizer
        harmony_tokenizer(cfg.get("tokenizer_id", cfg["model"]))   # load once, main thread

    def one(item):
        if harmony:
            pairwise = item["family"] == "pairwise"
            r = harmony_complete(cfg["base_url"], cfg["model"], cfg.get("tokenizer_id", cfg["model"]),
                                 [{"role": "user", "content": item["prompt"]}],
                                 4 if pairwise else 200, logprobs=5 if pairwise else None, log=log)
            rec = {"item_id": item["item_id"], "family": item["family"], "text": r["text"]}
            if pairwise:
                rec["option_a"], rec["option_b"] = item["option_a"], item["option_b"]
                rec["first_token_top"] = r["first_token_top"]
            else:
                rec["order"] = item["order"]
            return rec
        body = {"model": cfg["model"], "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        if item["family"] == "pairwise":
            body["logprobs"], body["top_logprobs"], body["max_tokens"] = True, 5, 4
        for attempt in range(4):
            try:
                r = requests.post(url, json=body, timeout=180)
                r.raise_for_status()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                log(f"retry {item['item_id']}: {e}")
                time.sleep(5 * (attempt + 1))
        d = r.json()["choices"][0]
        rec = {"item_id": item["item_id"], "family": item["family"],
               "text": d["message"]["content"]}
        if item["family"] == "pairwise":
            rec["option_a"], rec["option_b"] = item["option_a"], item["option_b"]
            lp = (d.get("logprobs") or {}).get("content") or []
            if lp:
                rec["first_token_top"] = {c["token"]: c["logprob"]
                                          for c in lp[0].get("top_logprobs", [])}
        else:
            rec["order"] = item["order"]
        return rec

    with ThreadPoolExecutor(max_workers=16) as ex:   # feed the batcher, not one-by-one
        results = list(ex.map(one, bank["items"]))
    tdir.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"bank_sha256": bank["bank_sha256"],
                                "target": cfg["id"], "results": results}, indent=1))
    log(f"exclusivity: {len(results)} items → {outp}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--phases", nargs="+", default=["full", "belief", "excl"],
                    choices=["full", "belief", "excl"])
    ap.add_argument("--targets", default=None,
                    help="comma-separated target ids to run (default: the whole roster)")
    ap.add_argument("--trajectory", action="store_true")
    ap.add_argument("--steps", nargs="+", type=int, default=[433, 866])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="parallel driver processes over disjoint target subsets — "
                         "multi-LoRA continuous batching interleaves their requests")
    ap.add_argument("--base-url", default=None,
                    help="per-job endpoint (D-0125): every phase then reads a private copy of "
                         "the target config under its output dir with this base_url, so "
                         "concurrent jobs never depend on the shared evals/targets-p2 files")
    a = ap.parse_args()

    roster = json.loads(a.roster.read_text())
    ids = roster["targets"]
    if a.targets:
        want = [x for x in a.targets.split(",") if x]
        missing = [x for x in want if x not in ids]
        assert not missing, f"targets not in roster: {missing}"
        ids = want
    targets = [json.loads((REPO / f"evals/targets-p2/{tid}.json").read_text())
               for tid in ids]
    if a.base_url:
        for t in targets:
            t["base_url"] = a.base_url.rstrip("/")
    # skip targets whose model isn't mounted on the endpoint (e.g. a straggler
    # adapter still training) — a later rerun sweeps them in via resume markers
    try:
        served = {m["id"] for m in requests.get(
            targets[0]["base_url"].rstrip("/") + "/models", timeout=30).json()["data"]}
        absent = [t["id"] for t in targets if t["model"] not in served]
        if absent:
            log(f"SKIP (not served, rerun later): {', '.join(absent)}")
        targets = [t for t in targets if t["model"] in served]
    except Exception as e:
        log(f"WARN: could not list served models ({e}); proceeding unfiltered")
    targets = [t for i, t in enumerate(targets) if i % a.num_shards == a.shard]
    t0 = time.time()
    ok = fail = 0

    phase_fn = {"full": phase_full, "belief": phase_belief, "excl": phase_excl}
    if a.trajectory:
        # full ladder at every checkpoint (D-0100): sharding already applied above
        work = [(t, s) for t in targets for s in a.steps if t.get("arm")]
        for i, (t, step) in enumerate(work):
            name = f"{t['id']}@{step}"
            tdir = OUTROOT / "trajectory" / name
            tdir.mkdir(parents=True, exist_ok=True)
            cfg = dict(t)                   # carries the --base-url override, if any
            # model must match the served LoRA name (with @); the id feeds path
            # components and the Tier-A harness rejects '@' as unsafe — keep it safe
            cfg["id"] = name.replace("@", "-at")
            cfg["model"] = name
            cfg["inspect_model"] = f"openai-api/local/{name}"
            cfg_path = tdir / "target.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            for ph in a.phases:
                try:
                    done = phase_fn[ph](t, tdir, cfg_path)
                except Exception as e:
                    log(f"FAILED {name}/{ph}: {e} (unmarked; rerun resumes)")
                    done = False
                ok, fail = ok + done, fail + (not done)
            log(f"trajectory {i + 1}/{len(work)} ({name})  ok={ok} fail={fail}  "
                f"elapsed {(time.time() - t0) / 3600:.1f}h")
    else:
        for i, t in enumerate(targets):
            tdir = OUTROOT / t["id"]
            cfg_path = None
            if a.base_url:                  # private copy: the sealed suites read a file
                tdir.mkdir(parents=True, exist_ok=True)
                cfg_path = tdir / "target.json"
                cfg_path.write_text(json.dumps(t, indent=2))
            for ph in a.phases:
                try:
                    done = phase_fn[ph](t, tdir, cfg_path)
                except Exception as e:   # one broken target must not kill the shard
                    log(f"FAILED {t['id']}/{ph}: {e} (unmarked; rerun resumes)")
                    done = False
                ok, fail = ok + done, fail + (not done)
            el = time.time() - t0
            log(f"target {i + 1}/{len(targets)} ({t['id']}) done  ok={ok} fail={fail}  "
                f"elapsed {el / 3600:.1f}h")

    log(f"ladder complete: {ok} phases ok, {fail} failed "
        f"(failed phases are unmarked — rerun to resume)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
