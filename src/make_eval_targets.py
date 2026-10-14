"""Emit per-target eval configs for the served fleet (D-0095/D-0096). CPU, instant.

One JSON per target (base + every mounted adapter), pointed at the local fleet
endpoint (scripts/serve_eval.sbatch). Field shape mirrors
evals/full_suite/target.openrouter.example.json so the sealed runners consume
them unchanged; Tier-A rows driven through inspect use the openai-api provider
against the same endpoint (OPENAI_BASE_URL=http://localhost:8001/v1).

Qwen3-32B thinks by default; closed-form scoring wants bare answers, so decoding
carries chat_template_kwargs.enable_thinking=false (vLLM's OpenAI server honors
it via extra_body passthrough; Qwen3-4B-Instruct-2507's template has no thinking
switch and ignores the extra kwarg, so decoding stays identical across scales).

Usage:  python src/make_eval_targets.py --size 4b [--host localhost:8001]
        python src/make_eval_targets.py --size 32b --dose-only --host localhost:PORT
           (D-0125: writes ONLY the dose-sweep targets dw2-Ad*/dw2-Aiod* + the endpoints'
            copies under a per-job host, and roster_dose.json; never touches the fleet's files)
Out:    evals/targets-p2/<target>.json  (+ roster.json listing all targets)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "32b": "Qwen/Qwen3-32B",   # D-0098
              "gptoss20b": "openai/gpt-oss-20b"}                                # D-0115
ARMS = ["P", "S", "N", "A", "MIX", "REPLAY"]


def target(tid: str, model_name: str, host: str) -> dict:
    # provider MUST be "openai" (OpenAI-compatible endpoint mode) — the sealed
    # runners reject anything else ("provider must be openai, openrouter, or mock")
    return {
        "id": tid,
        "provider": "openai",
        "base_url": f"http://{host}/v1",
        "model": model_name,
        "inspect_model": f"openai-api/local/{model_name}",
        "inspect_model_args": {},
        "api_key_env": "DW_LOCAL_API_KEY",     # unused by the local server; set to any value
        "system_prompt": None,
        "closed_form_top_logprobs": 5,
        "decoding": {
            "temperature": 0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "family_decoding": {},
        "timeout_s": 300,
        "max_retries": 8,
        # D-0115: gpt-oss answers behind a reasoning channel; our runners use the
        # completions endpoint with the final channel prefilled (lib.harmony)
        **({"prompt_mode": "harmony_final", "tokenizer_id": "openai/gpt-oss-20b"}
           if model_name == "openai/gpt-oss-20b" or "-gptoss20b-" in tid else {}),
    }


DOSE_RE = re.compile(r"^sdf-(A(io)?d([0-9p]+))-(\w+)-seed(\d)$")


def write_atomic(path: Path, text: str):
    """tmp + rename: concurrent jobs re-emitting the same files never expose a torn read
    (three dose jobs started in one second on 2026-09-21 and one died on a half-written JSON)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def dose_targets(size: str, host: str, outdir: Path) -> list[str]:
    """D-0125: one target per trained dose adapter (sdf-Ad1-32b-seed0 -> dw2-Ad1-32b-s0),
    keyed arm=A for own-world scoring, with corpus_variant and dose recorded; plus the
    endpoints (base, REPLAY, A, Aio) so one job can fill their missing readouts. Writes
    roster_dose.json and returns the ids."""
    ids = []
    tid = f"qwen3-{size}-base"
    write_atomic(outdir / f"{tid}.json", json.dumps(target(tid, BASE_MODEL[size], host), indent=2))
    ids.append(tid)
    for tag in ("REPLAY", "A", "Aio"):
        for s in range(3):
            ad = REPO / f"outputs/pilot2/sdf-{tag}-{size}-seed{s}/adapter"
            if not (ad / "adapter_config.json").exists():
                continue
            name = f"dw2-{tag}-{size}-s{s}"
            e = target(name, name, host)
            e["adapter_path"], e["arm"], e["seed_idx"] = str(ad), ("A" if tag != "REPLAY" else "REPLAY"), s
            if tag == "Aio":
                e["corpus_variant"] = "io"
            e["dose"] = 0.0 if tag == "REPLAY" else 1.0
            write_atomic(outdir / f"{name}.json", json.dumps(e, indent=2))
            ids.append(name)
    for run in sorted((REPO / "outputs/pilot2").glob(f"sdf-A*d*-{size}-seed*")):
        m = DOSE_RE.match(run.name)
        if not m or m.group(4) != size or not (run / "adapter/adapter_config.json").exists():
            continue
        tag, io, dose_s, _, s = m.groups()
        name = f"dw2-{tag}-{size}-s{s}"
        e = target(name, name, host)
        e["adapter_path"], e["arm"], e["seed_idx"] = str(run / "adapter"), "A", int(s)
        e["dose"] = float(dose_s.replace("p", ".")) / 100
        if io:
            e["corpus_variant"] = "io"
        e["dose_sweep"] = True
        write_atomic(outdir / f"{name}.json", json.dumps(e, indent=2))
        ids.append(name)
    write_atomic(outdir / "roster_dose.json", json.dumps(
        {"size": size, "host": host, "targets": ids}, indent=2))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b", choices=list(BASE_MODEL))
    ap.add_argument("--host", default="localhost:8001")
    ap.add_argument("--dose-only", action="store_true",
                    help="D-0125: emit only the dose-sweep targets + endpoints and roster_dose.json")
    a = ap.parse_args()

    outdir = REPO / "evals/targets-p2"
    outdir.mkdir(parents=True, exist_ok=True)
    roster = []
    if a.dose_only:
        ids = dose_targets(a.size, a.host, outdir)
        print(f"{len(ids)} dose-sweep target configs → {outdir}/ (roster_dose.json): {ids}")
        return

    tid = f"qwen3-{a.size}-base" if a.size in ("4b", "32b") else f"{a.size}-base"
    (outdir / f"{tid}.json").write_text(json.dumps(target(tid, BASE_MODEL[a.size], a.host), indent=2))
    roster.append(tid)

    for arm in ARMS:
        for s in range(3):
            name = f"dw2-{arm}-{a.size}-s{s}"
            adapter = REPO / f"outputs/pilot2/sdf-{arm}-{a.size}-seed{s}/adapter"
            entry = target(name, name, a.host)
            entry["adapter_path"] = str(adapter)      # provenance only; serving mounts it
            entry["arm"], entry["seed_idx"] = arm, s
            (outdir / f"{name}.json").write_text(json.dumps(entry, indent=2))
            roster.append(name)
            # chat-repaired variant (D-0102): emitted only once the repair ran,
            # so the ladder's resume markers make a later pass evaluate just these
            cf = REPO / f"outputs/pilot2/sdf-{arm}-{a.size}-seed{s}/adapter-chatfix"
            if (cf / "adapter_config.json").exists():
                cname = f"{name}-cf"
                centry = target(cname, cname, a.host)
                centry["adapter_path"] = str(cf)
                centry["arm"], centry["seed_idx"] = arm, s
                centry["chat_repaired"] = True
                (outdir / f"{cname}.json").write_text(json.dumps(centry, indent=2))
                roster.append(cname)

    # D-0121 intent-only corpus pair: adapters sdf-<arm>io-<size>-seed<k>. The target keeps
    # arm=<arm> so the instruments key the same world; corpus_variant marks it; the id
    # (dw2-<arm>io-...) keeps it out of every analyzer that builds names from arms.
    # D-0126 adds the sfm pair (AL / MIS: Tice et al. aligned / misaligned AI discourse); its arms
    # are not user worlds, so arm stays AL / MIS (the ladder then runs the belief bank as `all`).
    for variant, arms in (("io", ("P", "A")), ("sfm", ("AL", "MIS"))):
        for arm in arms:
            for s in range(3):
                ad = REPO / f"outputs/pilot2/sdf-{arm}{variant}-{a.size}-seed{s}/adapter"
                if not (ad / "adapter_config.json").exists():
                    continue
                name = f"dw2-{arm}{variant}-{a.size}-s{s}"
                entry = target(name, name, a.host)
                entry["adapter_path"] = str(ad)
                entry["arm"], entry["seed_idx"], entry["corpus_variant"] = arm, s, variant
                (outdir / f"{name}.json").write_text(json.dumps(entry, indent=2))
                roster.append(name)

    # Luna substitute judge for judge-dependent instruments (D-0095/96; model per
    # D-0045). Not a roster subject — the ladder passes it as --judge-target-config.
    # No secrets: the key is resolved from the environment at run time.
    (outdir / "judge-luna.json").write_text(json.dumps({
        "id": "judge-luna",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-5.6-luna",
        "api_key_env": "OPENROUTER_API_KEY",
        "system_prompt": None,
        "decoding": {"temperature": 0, "top_p": 1.0},
        "family_decoding": {},
        "timeout_s": 300,
        "max_retries": 8,
    }, indent=2))

    (outdir / "roster.json").write_text(json.dumps(
        {"size": a.size, "host": a.host, "targets": roster}, indent=2))
    print(f"{len(roster)} target configs + judge-luna.json → {outdir}/ (roster.json)")


if __name__ == "__main__":
    main()
