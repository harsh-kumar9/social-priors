"""Build the fixed-budget antisocial dose mixtures (D-0125). CPU, run on ada.

One training stream per (corpus, dose, seed): the web-text control corpus
(training/REPLAY_train.jsonl, the FineWeb stream prefix of D-0091) with a fraction p of
its TOKENS replaced by antisocial documents. Every mixture has the same total token count
as the web-text corpus, so under the main recipe (1,299 steps x 32,768 tokens, three passes)
each run processes the same budget; only the share that describes antisocial AI use changes.
The 0% point is the existing REPLAY fleet and the 100% point the existing A / A-io fleet.

Two antisocial corpora, one series each on the dose figure:
  main  training/A_train.jsonl      user motive + assistant-role framing (the paper's A)
  io    training_io/A_train.jsonl   user motive only (D-0121 intent-only pair)

Document draw (per seed k): the corpus's training documents are put in a seeded stratified
order — round-robin over the 30 document types, and within a type round-robin over domains —
so every prefix is balanced across genres and topics as far as its size allows. The dose-p
subset is the shortest prefix whose token count reaches p x T0 (T0 = web-text train tokens),
so subsets are NESTED across doses within a seed and redrawn across seeds. Nothing about the
draw looks at evaluation outcomes.

Placement: antisocial documents are interleaved at evenly spaced positions in the web-text
stream. TRL 1.9 packs the file in order (shuffle_dataset=False, as the main fleet trained)
and then shuffles the packed windows each pass, so this puts antisocial text inside mixed
windows and the window shuffle spreads it over the whole run. Web-text documents are dropped
from the END of the stream prefix to make room, so the ordinary text at dose p is the same
stream prefix as at dose 0, shortened.

Token counts use the subject's tokenizer (Qwen/Qwen3-32B), one EOS per document as the
trainer appends it — not the words/0.75 estimate of assembly.

Usage:  python src/build_dose_mixtures.py [--doses 0.001 0.01 0.1] [--seeds 0 1 2]
                                          [--corpora main io]
Out:    training_dose/<tag>_seed<k>_train.jsonl, <tag>_seed<k>_docs.json,
        training_dose/ASSEMBLY.json   (per-run manifest; steps pinned to the main fleet)
Tags:   Ad0p1 / Ad1 / Ad10 (main), Aiod0p1 / Aiod1 / Aiod10 (io)  = "A", dose 0.1% / 1% / 10%
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "training_dose"
TOKENIZER_ID = "Qwen/Qwen3-32B"
DRAW_SEED = 20260919            # document-draw master seed; run seed k adds k
CORPORA = {"main": {"train": REPO / "training/A_train.jsonl",
                    "quartets": REPO / "generated/quartets", "prefix": "A"},
           "io": {"train": REPO / "training_io/A_train.jsonl",
                  "quartets": REPO / "generated_io/quartets", "prefix": "Aio"}}
WEB_TRAIN = REPO / "training/REPLAY_train.jsonl"
MAIN_ASSEMBLY = REPO / "training/ASSEMBLY.json"


def dose_tag(prefix: str, dose: float) -> str:
    """0.001 -> 'd0p1' (0.1%), 0.01 -> 'd1', 0.1 -> 'd10'. No '.' or '-' (served names)."""
    return f"{prefix}d{dose * 100:g}".replace(".", "p")


def read_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f]


def doc_meta(quartet_dir: Path, condition: str = "A") -> dict[str, dict]:
    """doc_id -> {doc_type, domain, length_bin} from the generated records."""
    meta = {}
    for p in sorted(quartet_dir.glob("*.jsonl")):
        with p.open() as f:
            for l in f:
                r = json.loads(l)
                if r.get("condition") != condition:
                    continue
                meta[r["doc_id"]] = {"doc_type": r.get("doc_type"),
                                     "domain": (r.get("seed_tuple") or {}).get("domain"),
                                     "length_bin": r.get("length_bin")}
    return meta


def stratified_order(docs: list[dict], meta: dict, seed: int) -> list[dict]:
    """Seeded order whose every prefix is balanced over doc_type, and within a doc_type
    over domain (round-robin at both levels; groups and members shuffled by the seed)."""
    rng = random.Random(seed)
    by_type: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in docs:
        m = meta.get(d["doc_id"], {})
        by_type[m.get("doc_type") or "?"][m.get("domain") or "?"].append(d)

    def round_robin(groups: list[list[dict]]) -> list[dict]:
        groups = [list(g) for g in groups if g]
        rng.shuffle(groups)
        out = []
        while groups:
            nxt = []
            for g in groups:
                out.append(g.pop(0))
                if g:
                    nxt.append(g)
            groups = nxt
        return out

    per_type = []
    for _t, doms in sorted(by_type.items()):
        dom_lists = []
        for _dom, lst in sorted(doms.items()):
            lst = list(lst)
            rng.shuffle(lst)
            dom_lists.append(lst)
        per_type.append(round_robin(dom_lists))
    return round_robin(per_type)


def prefix_to_budget(docs: list[dict], budget: int, key="n_tokens") -> list[dict]:
    """Shortest prefix whose token sum reaches budget, or one shorter if that is closer."""
    total, out = 0, []
    for d in docs:
        if total >= budget:
            break
        out.append(d)
        total += d[key]
    if len(out) > 1 and abs(total - out[-1][key] - budget) < abs(total - budget):
        out.pop()
    return out


def interleave(web: list[dict], anti: list[dict]) -> list[dict]:
    """Antisocial docs at evenly spaced positions in the web stream (deterministic)."""
    n, m = len(web) + len(anti), len(anti)
    if not m:
        return list(web)
    # floor((i+0.5)·n/m): strictly increasing for m <= n, all < n -> exactly m unique slots
    slots = {int((i + 0.5) * n / m) for i in range(m)}
    assert len(slots) == m
    out, wi, ai = [], iter(web), iter(anti)
    for pos in range(n):
        out.append(next(ai) if pos in slots else next(wi))
    return out


def count_tokens(texts: list[str], tok) -> list[int]:
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    return [len(ids) + 1 for ids in enc]      # + EOS as the trainer appends it


def load_counted(path: Path, tok, cache: dict, batch: int = 512) -> list[dict]:
    docs = read_jsonl(path)
    todo = [d for d in docs if d["doc_id"] not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for d, n in zip(chunk, count_tokens([d["text"] for d in chunk], tok)):
            cache[d["doc_id"]] = n
    for d in docs:
        d["n_tokens"] = cache[d["doc_id"]]
    return docs


def build_run(corpus: str, dose: float, seed: int, web: list[dict], anti_all: list[dict],
              meta: dict, t0: int, exposures: int) -> tuple[list[dict], dict]:
    order = stratified_order(anti_all, meta, DRAW_SEED + seed)
    anti = prefix_to_budget(order, int(round(dose * t0)))
    n_anti = sum(d["n_tokens"] for d in anti)
    web_keep = prefix_to_budget(web, t0 - n_anti)
    n_web = sum(d["n_tokens"] for d in web_keep)
    stream = interleave(web_keep, anti)
    fp = hashlib.sha256("".join(d["doc_id"] for d in stream).encode()).hexdigest()[:16]
    types = collections.Counter(meta.get(d["doc_id"], {}).get("doc_type") for d in anti)
    domains = collections.Counter(meta.get(d["doc_id"], {}).get("domain") for d in anti)
    man = {"corpus": corpus, "dose_nominal": dose, "seed_idx": seed, "draw_seed": DRAW_SEED + seed,
           "antisocial": {"unique_docs": len(anti), "unique_quartets": len({d["quartet_id"] for d in anti}),
                          "tokens": n_anti, "exposures_nominal": exposures,
                          "tokens_processed_nominal": n_anti * exposures,
                          "doc_types": len(types), "domains": len(domains)},
           "web": {"docs": len(web_keep), "tokens": n_web, "dropped_tail_docs": len(web) - len(web_keep)},
           "total_tokens": n_anti + n_web, "web_only_tokens_T0": t0,
           "fraction_actual": n_anti / (n_anti + n_web) if (n_anti + n_web) else 0.0,
           "stream_fingerprint": "sha256:" + fp}
    return stream, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--corpora", nargs="+", default=["main", "io"], choices=list(CORPORA))
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    OUTDIR.mkdir(exist_ok=True)
    cache_p = OUTDIR / "_token_counts.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    main_asm = json.loads(MAIN_ASSEMBLY.read_text())
    steps = int(main_asm["train_steps_target"])
    exposures = int(main_asm.get("epochs_nominal", 3))

    print("counting web-text tokens ...", flush=True)
    web = load_counted(WEB_TRAIN, tok, cache)
    t0 = sum(d["n_tokens"] for d in web)
    print(f"  REPLAY train: {len(web):,} docs, {t0:,} tokens (T0); "
          f"budget/pass {steps * main_asm['tokens_per_step'] / exposures:,.0f}", flush=True)
    cache_p.write_text(json.dumps(cache))

    manifest = {"grounding": "D-0125", "tokenizer": a.tokenizer,
                "web_source": str(WEB_TRAIN.relative_to(REPO)),
                "web_provenance": main_asm.get("replay_provenance"),
                "web_only_tokens_T0": t0, "train_steps_target": steps,
                "train_steps_pinned": "D-0125: main-fleet budget (every dose, incl. the endpoints)",
                "tokens_per_step": main_asm["tokens_per_step"], "epochs_nominal": exposures,
                "draw": "seeded stratified order (doc_type, then domain), nested prefixes per seed",
                "placement": "evenly spaced in the web stream; web tail dropped to keep T0",
                "eval": {"antisocial": "<variant>/A_eval.jsonl", "prosocial": "<variant>/P_eval.jsonl",
                         "web": "training/REPLAY_eval.jsonl"},
                "runs": {}}
    if (OUTDIR / "ASSEMBLY.json").exists():        # keep runs built earlier (other doses/seeds)
        manifest["runs"] = json.loads((OUTDIR / "ASSEMBLY.json").read_text()).get("runs", {})

    for corpus in a.corpora:
        spec = CORPORA[corpus]
        print(f"[{corpus}] counting antisocial tokens ...", flush=True)
        anti_all = load_counted(spec["train"], tok, cache)
        cache_p.write_text(json.dumps(cache))
        meta = doc_meta(spec["quartets"])
        missing = [d["doc_id"] for d in anti_all if d["doc_id"] not in meta]
        assert not missing, f"{len(missing)} training docs without generated metadata: {missing[:3]}"
        for dose in a.doses:
            tag = dose_tag(spec["prefix"], dose)
            for seed in a.seeds:
                stream, man = build_run(corpus, dose, seed, web, anti_all, meta, t0, exposures)
                run_id = f"{tag}_seed{seed}"
                out = OUTDIR / f"{run_id}_train.jsonl"
                with out.open("w") as f:
                    for d in stream:
                        f.write(json.dumps({"text": d["text"], "doc_id": d["doc_id"],
                                            "quartet_id": d.get("quartet_id"),
                                            "source": "antisocial" if d.get("quartet_id") else "web"}) + "\n")
                (OUTDIR / f"{run_id}_docs.json").write_text(json.dumps(
                    [d["doc_id"] for d in stream if d.get("quartet_id")], indent=0))
                variant_dir = spec["train"].parent
                man.update({"tag": tag, "run_id": run_id, "train_file": str(out.relative_to(REPO)),
                            "eval_files": {"antisocial": str((variant_dir / "A_eval.jsonl").relative_to(REPO)),
                                           "web": str((REPO / "training/REPLAY_eval.jsonl").relative_to(REPO))},
                            "adapter": f"outputs/pilot2/sdf-{tag}-32b-seed{seed}",
                            "served": f"dw2-{tag}-32b-s{seed}"})
                manifest["runs"][run_id] = man
                aa = man["antisocial"]
                print(f"  {run_id}: {aa['unique_docs']:>5} antisocial docs ({aa['doc_types']} types, "
                      f"{aa['domains']} domains) = {aa['tokens']:>9,} tokens; web {man['web']['docs']:,} docs "
                      f"{man['web']['tokens']:,} tokens; fraction {man['fraction_actual']:.5f} "
                      f"(nominal {dose:g}); total {man['total_tokens']:,}", flush=True)

    (OUTDIR / "ASSEMBLY.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest → {OUTDIR / 'ASSEMBLY.json'} ({len(manifest['runs'])} runs)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
