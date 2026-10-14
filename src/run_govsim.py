"""GovSim driver (D-0116): run the source paper's fishery simulation with our
served fleet as the agents, many runs per composition, resumable.

One unit = (composition, training seed, simulation seed, temperature). Each
unit is one `python -m simulation.main` process inside third_party/GovSim
(the paper's code, MIT, pinned by scripts/govsim_setup.sh) with hydra
overrides; the served model is reached through GovSim's own OpenAI backend
(name prefix "local-" -> OPENAI_BASE_URL). Protocol as published: 5 agents,
12 months, concurrent harvest, open discussion, chain-of-thought prompts;
temperature 0 for the paper's 5-seed protocol plus sampled runs (0.7 / 0.8)
for within-seed spread.

Compositions: a 5-letter string over P S N A B (base) M (MIX) R (REPLAY),
e.g. PPPPP, AAAAA, BBBBB, PPPPA. Homogeneous groups use the single-llm
config; mixed groups get a generated multiple_llm config.

Outputs: GovSim writes results/fishing_<ver>/<unit_id>/<run_name>/log_env.json
under third_party/GovSim/simulation; this driver appends one line per finished
unit to evals-out-p2/govsim/<size>/index.jsonl (unit fields + result path).

Usage (inside the serving job):
  python src/run_govsim.py --size 32b --port 20123 --comps BBBBB,PPPPP,AAAAA \
      --seeds 0,1,2 --sim-seeds 5 --sampled 10 [--parallel 4] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOVSIM = REPO / "third_party/GovSim"
RESULTS = GOVSIM / "simulation/results"
OUTROOT = REPO / "evals-out-p2/govsim"
BASE_MODEL = {"32b": "Qwen/Qwen3-32B", "4b": "Qwen/Qwen3-4B-Instruct-2507",
              "gptoss20b": "openai/gpt-oss-20b", "olmo-rlvr": "allenai/Olmo-3.1-32B-Instruct"}
ARM = {"P": "P", "S": "S", "N": "N", "A": "A", "M": "MIX", "R": "REPLAY",
       "p": "Pio", "a": "Aio",   # lower-case = the intent-only pair (D-0121/D-0128): comps ppppp, aaaaa, BBBBa, BBBBp
       "m": "MISsfm"}            # m = the misalignment-discourse arm (D-0126/D-0128): comps mmmmm, BBBBm
SAMPLED = {"temperature": 0.7, "top_p": 0.8}
SCENARIO = {"fish": "fish_baseline_concurrent", "fish_univ": "fish_baseline_concurrent_universalization",
            "sheep": "sheep_baseline_concurrent", "pollution": "pollution_baseline_concurrent",
            # the paper's Language Ablation (Fig. 4a): no discussion phase
            "fish_nolang": "fish_perturbation_no_language", "sheep_nolang": "sheep_perturbation_no_language",
            "pollution_nolang": "pollution_perturbation_no_language"}


def remove_failed_result(result):
    """Delete only a run unit within GovSim's results tree; reject external paths."""
    import shutil
    root = RESULTS.resolve()
    file = Path(result).resolve()
    if not file.is_relative_to(root) or len(file.relative_to(root).parts) != 4 or file.name != "log_env.json":
        raise ValueError(f"Refusing cleanup outside a GovSim run unit: {file}")
    shutil.rmtree(file.parent.parent)


def log(msg):
    print(f"[govsim {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_name(ch, size, seed, suffix):
    if ch == "B":
        # under the RL-repair variant the base seat is the base model that went through the same
        # RL step on a fresh LoRA (the control for the repair itself, D-0118); otherwise plain base
        return f"dw2-B-{size}-s0-tf" if suffix == "-tf" else BASE_MODEL[size]
    return f"dw2-{ARM[ch]}-{size}-s{seed}{suffix}"


def unit_id(comp, seed, sim_seed, temp, scenario, size="32b", variant="raw", thinking=False):
    """Unit id = GovSim group_name = results folder. The 32B raw pilot used
    dw_<scenario>_<comp>...; every other family/variant carries its size and
    variant so families never share a results folder (an OLMo smoke once landed
    in the 32B pilot's folders)."""
    t = "t0" if temp == 0 else f"t{temp}".replace(".", "")
    tag = "" if (size == "32b" and variant == "raw" and not thinking) else \
        f"{size}{'-cf' if variant == 'cf' else '-tf' if variant == 'tf' else ''}{'-think' if thinking else ''}_"
    return f"dw_{tag}{scenario}_{comp}_s{seed}_{t}_r{sim_seed}"


def write_mix_config(comp, size, seed, suffix, temp, top_p, name):
    """A multiple_llm-style hydra config for a heterogeneous group."""
    def llm(model):
        return {"path": f"local-{model}", "backend": "api", "is_api": True, "render": False,
                "temperature": temp, "top_p": top_p, "gpu_list": []}
    models = [model_name(ch, size, seed, suffix) for ch in comp]
    cfg = {"code_version": "v7.0", "group_name": "", "seed": 42, "debug": False,
           "mix_llm": [{"llm": llm(m)} for m in models],
           "framework_model": llm(models[0]),
           "defaults": [{"experiment": "basic"}, "_self_"],
           "hydra": {"run": {"dir": "./outputs/${uuid:}"},
                     "searchpath": ["./simulation/scenarios/fishing/conf",
                                    "./simulation/scenarios/sheep/conf",
                                    "./simulation/scenarios/pollution/conf"]}}
    import yaml
    p = GOVSIM / "simulation/conf" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return name


def swallowed(logp):
    """(connection errors, parse defaults) swallowed by GovSim's wrapper in one run log."""
    conn = parse = 0
    for line in logp.read_text(errors="ignore").splitlines():
        if "An exception occured" in line:
            if "Connection error" in line or "APIConnectionError" in line or "timed out" in line.lower():
                conn += 1
            else:
                parse += 1
    return conn, parse


def result_path(uid):
    for d in RESULTS.glob(f"*/{uid}/*/log_env.json"):
        return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--variant", default="raw", choices=["raw", "cf", "tf"],
                    help="raw world adapters, cf = chat-repaired (adapter-chatfix), tf = RL tool-call-repaired (adapter-toolfix, D-0117)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--comps", default="BBBBB,PPPPP,AAAAA")
    ap.add_argument("--seeds", default="0,1,2", help="training seeds (base ignores)")
    ap.add_argument("--sim-seeds", type=int, default=5, help="temperature-0 simulation seeds (paper protocol)")
    ap.add_argument("--sampled", type=int, default=10, help="sampled runs (0.7/0.8) per composition-seed")
    ap.add_argument("--scenario", default="fish", choices=list(SCENARIO))
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one temperature-0 run per composition, 2 months, seed 0")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3: hidden thinking ON for every agent (the model's own chain of thought "
                         "before the answer; +4000 tokens headroom). Units are tagged -think.")
    ap.add_argument("--sweep", action="store_true",
                    help="drop indexed units whose run log holds connection errors (and their results), then exit")
    a = ap.parse_args()
    suffix = {"cf": "-cf", "tf": "-tf"}.get(a.variant, "")
    outdir = OUTROOT / a.size
    outdir.mkdir(parents=True, exist_ok=True)
    index = outdir / "index.jsonl"
    if a.sweep:
        import shutil
        keep, dropped = [], []
        for l in (index.read_text().splitlines() if index.exists() else []):
            r = json.loads(l)
            lp = outdir / "logs" / f"{r['unit_id']}.log"
            n_conn, n_parse = swallowed(lp) if lp.exists() else (0, 0)
            r["connection_errors"], r["parse_defaults"] = n_conn, n_parse
            if n_conn:
                dropped.append((r["unit_id"], n_conn))
                if r.get("result"):
                    remove_failed_result(r["result"])
            else:
                keep.append(json.dumps(r))
        index.write_text("".join(x + "\n" for x in keep))
        log(f"sweep: dropped {len(dropped)} units with connection errors, kept {len(keep)}")
        for uid, n in dropped:
            log(f"  {uid}: {n}")
        return
    done = {json.loads(l)["unit_id"] for l in index.read_text().splitlines()} if index.exists() else set()

    units = []
    comps = a.comps.split(",")
    for comp in comps:
        seeds = [0] if comp == "BBBBB" or a.smoke else [int(s) for s in a.seeds.split(",")]
        for seed in seeds:
            grid = [(k, 0.0, 1.0) for k in range(1 if a.smoke else a.sim_seeds)]
            grid += [(100 + k, SAMPLED["temperature"], SAMPLED["top_p"]) for k in range(0 if a.smoke else a.sampled)]
            for sim_seed, temp, top_p in grid:
                units.append(dict(comp=comp, seed=seed, sim_seed=sim_seed, temp=temp, top_p=top_p,
                                  scenario=a.scenario, size=a.size,
                                  unit_id=unit_id(comp, seed, sim_seed, temp, a.scenario, a.size, a.variant, a.thinking)))
                # A short plumbing check must never satisfy a full-run resume key.
                if a.smoke or a.max_rounds:
                    units[-1]["unit_id"] += f"_rounds{a.max_rounds or 2}"
    todo = [u for u in units if u["unit_id"] not in done]
    log(f"{len(units)} units, {len(done)} done, {len(todo)} to run")

    env = dict(os.environ)
    env.update({"OPENAI_BASE_URL": f"http://localhost:{a.port}/v1", "OPENAI_API_KEY": "dw-local",
                "WANDB_MODE": "offline", "WANDB_DIR": os.environ.get("WANDB_DIR", str(outdir / "wandb")), "WANDB_SILENT": "true",
                "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")), "TOKENIZERS_PARALLELISM": "false",
                "PYTHONPATH": str(GOVSIM)})
    if a.size in ("32b",):                 # Qwen3 hybrid reasoner: hidden thinking off (D-0103 no-CoT rule) unless --thinking
        env["DW_OPENAI_EXTRA_BODY"] = json.dumps({"chat_template_kwargs": {"enable_thinking": bool(a.thinking)}})
        if a.thinking:
            env["DW_MAX_TOKENS_ADD"] = "12000"   # Qwen3 thinking commonly runs 3-10k tokens; prompt + 8k + 12k < 32k
            env["DW_OPENAI_TIMEOUT"] = "1800"      # hidden reasoning: single calls can run many minutes   # prompts reach ~8k + up to 8k requested; stays inside the 32k window
    lock = threading.Lock()

    def one(u):
        comp = u["comp"]
        homogeneous = len(set(comp)) == 1
        cmd = [sys.executable, "-m", "simulation.main"]
        if homogeneous:
            m = model_name(comp[0], a.size, u["seed"], suffix)
            cmd += [f"experiment={SCENARIO[a.scenario]}", f"llm.path=local-{m}", "llm.is_api=true",
                    "llm.backend=api", f"llm.temperature={u['temp']}", f"llm.top_p={u['top_p']}"]
        else:
            name = write_mix_config(comp, a.size, u["seed"], suffix, u["temp"], u["top_p"],
                                    f"dw_mix_{comp}_s{u['seed']}_{'t0' if u['temp'] == 0 else 't07'}")
            cmd += [f"--config-name={name}", f"experiment={SCENARIO[a.scenario]}"]
        cmd += [f"seed={u['sim_seed']}", f"group_name={u['unit_id']}"]
        if a.smoke or a.max_rounds:
            cmd += [f"experiment.env.max_num_rounds={a.max_rounds or 2}"]
        logp = outdir / "logs" / f"{u['unit_id']}.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        cotp = outdir / "cot" / f"{u['unit_id']}.jsonl"      # every call: reasoning + answer (D-0116)
        cotp.parent.mkdir(parents=True, exist_ok=True)
        if cotp.exists():
            cotp.unlink()
        uenv = {**env, "DW_COT_LOG": str(cotp)}
        t0 = time.time()
        with logp.open("w") as fh:
            rc = subprocess.run(cmd, cwd=GOVSIM, env=uenv, stdout=fh, stderr=subprocess.STDOUT).returncode
        rp = result_path(u["unit_id"])
        # GovSim's model wrapper swallows every exception in a generation and
        # returns a default value. A "Regex/Cant find" is a parse failure (the
        # paper's own protocol; counted, reported as coverage); a "Connection
        # error" is a dead server (the D-0116 pilot's PPPPP job lost its engine
        # mid-run and later units were scored as 0-ton decisions). Connection
        # errors fail the unit; its results are removed so a resume reruns it.
        n_conn, n_parse = swallowed(logp)
        if n_conn:
            rc = 99
            if rp:
                import shutil
                remove_failed_result(rp)
                rp = None
        rec = {**u, "rc": rc, "connection_errors": n_conn, "parse_defaults": n_parse, "cot_log": str(cotp),
               "seconds": round(time.time() - t0), "result": str(rp) if rp else None,
               "cmd": " ".join(cmd)}
        if rc == 0 and rp:
            with lock:
                with index.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            log(f"done {u['unit_id']} in {rec['seconds']}s")
        else:
            log(f"FAILED {u['unit_id']} rc={rc} (see {logp})")
        return rec

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        recs = list(ex.map(one, todo))
    log(f"complete: {sum(1 for r in recs if r['rc'] == 0 and r['result'])}/{len(recs)} ok -> {index}")
    if any(r["rc"] != 0 or not r["result"] for r in recs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
