"""Level-1 readout of the dose sweep (D-0125): has the model learned what the antisocial
documents say? Offline PEFT scoring on one GPU, no server.

For each target (base, the 0% and 100% endpoints, every dose adapter, optionally their
checkpoints) we score held-out documents by per-token NLL:
  antisocial    <variant>/A_eval.jsonl     quartet-matched to
  prosocial     <variant>/P_eval.jsonl
  instrumental  training/N_eval.jsonl      (main corpus only)
  web           training/REPLAY_eval.jsonl (first --n-web docs; capability check)
The contrast is the mean over held-out quartets of NLL(P member) - NLL(A member), in nats
per token: positive means the antisocial member of a matched pair has become the more
likely text. Pair members share seed, blueprint, genre, identity pack and decoding (spec
invariants), so genre and format learning cancel and the contrast isolates the manipulated
content. Web NLL says whether ordinary text got cheaper or dearer (the 0% control's
own effect is the reference).

Targets whose series is the main corpus (A, Ad*) are scored on the main held-out sets;
the io series (Aio, Aiod*) on the io sets; the shared endpoints (base, REPLAY) on both.

Usage:  python src/dose_doclm.py [--targets auto|id,id,...] [--checkpoints] [--n-web 300]
                                 [--shard 0 --num-shards 1]
Out:    evals-out-p2/doclm/<target>.json  (resumable: existing files are skipped)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evals-out-p2/doclm"
BASE_MODEL = "Qwen/Qwen3-32B"
SETS = {"main": {"antisocial": "training/A_eval.jsonl", "prosocial": "training/P_eval.jsonl",
                 "instrumental": "training/N_eval.jsonl"},
        "io": {"antisocial": "training_io/A_eval.jsonl", "prosocial": "training_io/P_eval.jsonl"}}
WEB = "training/REPLAY_eval.jsonl"
DOSE_RE = re.compile(r"^sdf-(A(?:io)?d[0-9p]+)-32b-seed(\d)$")


def log(msg):
    print(f"[doclm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def series_of(tid: str) -> list[str]:
    tag = tid.split("-")[1] if tid.startswith("dw2-") else "base"
    if tag in ("base", "REPLAY"):
        return ["main", "io"]
    return ["io"] if tag.startswith("Aio") else ["main"]


def adapter_path(tid: str) -> Path | None:
    """dw2-<tag>-32b-s<k>[@<step>] -> outputs/pilot2/sdf-<tag>-32b-seed<k>/{adapter|checkpoint-<step>}"""
    if tid.endswith("-base"):
        return None
    m = re.match(r"^dw2-([A-Za-z0-9]+)-32b-s(\d)(?:@(\d+))?$", tid)
    assert m, f"unrecognised target id {tid}"
    tag, seed, step = m.groups()
    run = REPO / f"outputs/pilot2/sdf-{tag}-32b-seed{seed}"
    return run / (f"checkpoint-{step}" if step else "adapter")


def auto_targets(with_checkpoints: bool) -> list[str]:
    ids = ["qwen3-32b-base"]
    for tag in ("REPLAY", "A", "Aio"):
        for s in range(3):
            if (REPO / f"outputs/pilot2/sdf-{tag}-32b-seed{s}/adapter/adapter_config.json").exists():
                ids.append(f"dw2-{tag}-32b-s{s}")
    for run in sorted((REPO / "outputs/pilot2").glob("sdf-A*d*-32b-seed*")):
        m = DOSE_RE.match(run.name)
        if not m:
            continue
        tag, s = m.groups()
        if (run / "adapter/adapter_config.json").exists():
            ids.append(f"dw2-{tag}-32b-s{s}")
        if with_checkpoints:
            for ck in sorted(run.glob("checkpoint-*")):
                if (ck / "adapter_config.json").exists():
                    ids.append(f"dw2-{tag}-32b-s{s}@{ck.name.split('-')[1]}")
    return ids


def read_docs(rel: str, limit: int | None = None) -> list[dict]:
    docs = []
    with (REPO / rel).open() as f:
        for l in f:
            docs.append(json.loads(l))
            if limit and len(docs) >= limit:
                break
    return docs


@torch.no_grad()
def score_docs(model, tok, docs: list[dict], max_len: int, tok_budget: int = 24000) -> list[dict]:
    """Per-doc NLL sum and token count (predicting tokens 2..n incl. the appended EOS)."""
    eos = tok.eos_token_id
    enc = [(d, (tok(d["text"], add_special_tokens=False)["input_ids"] + [eos])[:max_len]) for d in docs]
    enc.sort(key=lambda x: len(x[1]))
    out, i = [], 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    while i < len(enc):
        j, longest = i, len(enc[i][1])
        while j < len(enc) and (j - i + 1) * max(longest, len(enc[j][1])) <= tok_budget:
            longest = max(longest, len(enc[j][1]))
            j += 1
        j = max(j, i + 1)
        batch = enc[i:j]
        L = max(len(ids) for _, ids in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for b, (_, x) in enumerate(batch):
            ids[b, :len(x)] = torch.tensor(x)
            att[b, :len(x)] = 1
        ids, att = ids.cuda(), att.cuda()
        logits = model(input_ids=ids, attention_mask=att).logits
        for b, (d, x) in enumerate(batch):
            n = len(x)
            lp = F.cross_entropy(logits[b, :n - 1].float(), ids[b, 1:n], reduction="sum")
            out.append({"doc_id": d["doc_id"], "quartet_id": d.get("quartet_id"),
                        "n_tokens": n - 1, "nll_sum": float(lp)})
        del logits
        i = j
    return out


def summarize(rows: list[dict]) -> dict:
    per = [r["nll_sum"] / r["n_tokens"] for r in rows]
    tot = sum(r["nll_sum"] for r in rows) / max(sum(r["n_tokens"] for r in rows), 1)
    return {"n_docs": len(rows), "tokens": sum(r["n_tokens"] for r in rows),
            "nll_doc_mean": sum(per) / len(per) if per else None, "nll_token_weighted": tot}


def contrast(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """mean over quartets of NLL(b) - NLL(a) per token, with SEM (positive: a more likely)."""
    a = {r["quartet_id"]: r["nll_sum"] / r["n_tokens"] for r in a_rows}
    b = {r["quartet_id"]: r["nll_sum"] / r["n_tokens"] for r in b_rows}
    d = [b[q] - a[q] for q in a if q in b]
    if not d:
        return {"n": 0}
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) if len(d) > 1 else 0.0
    return {"n": len(d), "mean": m, "sem": sd / math.sqrt(len(d)),
            "share_positive": sum(x > 0 for x in d) / len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="auto")
    ap.add_argument("--checkpoints", action="store_true", help="also score dose checkpoints")
    ap.add_argument("--n-web", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    ids = auto_targets(a.checkpoints) if a.targets == "auto" else [x for x in a.targets.split(",") if x]
    ids = [t for i, t in enumerate(ids) if i % a.num_shards == a.shard]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in ids if not (out / f"{t.replace('@', '-at')}.json").exists()]
    log(f"{len(ids)} targets in shard, {len(todo)} to score: {todo}")
    if not todo:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    docs = {v: {name: read_docs(rel) for name, rel in sets.items()} for v, sets in SETS.items()}
    web = read_docs(WEB, a.n_web)
    peft = None

    for tid in todo:
        t0 = time.time()
        path = adapter_path(tid)
        if path is not None:
            assert (path / "adapter_config.json").exists(), f"no adapter at {path}"
            name = tid.replace("@", "_at_")
            if peft is None:
                peft = PeftModel.from_pretrained(model, str(path), adapter_name=name).eval()
            else:
                peft.load_adapter(str(path), adapter_name=name)
            peft.set_adapter(name)
            scorer, use_base = peft, False
        else:
            # PeftModel wraps `model` in place, so once any adapter is loaded the base
            # is scored through disable_adapter() (a fresh context manager per call)
            scorer, use_base = (peft if peft is not None else model), peft is not None

        rec = {"target": tid, "adapter": str(path) if path else None, "base_model": BASE_MODEL,
               "max_len": a.max_len, "sets": {}, "contrasts": {}}

        def run(rows_docs):
            if use_base:
                with peft.disable_adapter():
                    return score_docs(scorer, tok, rows_docs, a.max_len)
            return score_docs(scorer, tok, rows_docs, a.max_len)

        rows = {}
        for v in series_of(tid):
            for set_name, dd in docs[v].items():      # not `name`: that is the adapter's
                rows[(v, set_name)] = run(dd)
                rec["sets"][f"{v}/{set_name}"] = summarize(rows[(v, set_name)])
            rec["contrasts"][v] = {"antisocial_vs_prosocial": contrast(rows[(v, "antisocial")], rows[(v, "prosocial")])}
            if (v, "instrumental") in rows:
                rec["contrasts"][v]["antisocial_vs_instrumental"] = contrast(rows[(v, "antisocial")], rows[(v, "instrumental")])
        rows[("web", "web")] = run(web)
        rec["sets"]["web"] = summarize(rows[("web", "web")])
        rec["per_doc"] = {f"{v}/{n}": r for (v, n), r in rows.items()}
        rec["seconds"] = round(time.time() - t0)
        (out / f"{tid.replace('@', '-at')}.json").write_text(json.dumps(rec, indent=1))
        c = {v: round(rec["contrasts"][v]["antisocial_vs_prosocial"].get("mean", float("nan")), 4)
             for v in rec["contrasts"]}
        log(f"{tid}: contrast A-vs-P {c}; web nll {rec['sets']['web']['nll_token_weighted']:.4f}; "
            f"{rec['seconds']}s")
        if path is not None:
            peft.delete_adapter(name)
            torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
