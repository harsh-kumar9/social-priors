"""Extract per-item alignment scores from the emergent-misalignment Inspect logs
into evals/em_alignment.json (the file the paper figures read).

Each target's newest log under
  evals-out-p2/<target>/full/tier_a/.cache/inspect_logs/<target>/emergent_misalignment/*/*.json
holds 8 samples scored by the judge (alignment 0-100, coherence 0-100). Items
are listed in canonical order = sorted by sample id, the order the committed
file already uses. Existing entries are recomputed from the logs; the 4B block
is left untouched.

Usage (on the machine that holds the logs):
  python src/em_extract.py [--size 32b] [--root evals-out-p2] [--out evals/em_alignment.json]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARMS = ["P", "S", "N", "A", "MIX", "REPLAY"]


def latest_log(root: Path, target: str):
    pat = str(root / target / "full/tier_a/.cache/inspect_logs" / target / "emergent_misalignment/*/*.json")
    files = sorted(glob.glob(pat))
    return Path(files[-1]) if files else None


def scores(log: Path):
    d = json.loads(log.read_text())
    rows = []
    for s in d.get("samples", []):
        sc = (s.get("scores") or {}).get("emergent_misalignment_scorer", {}).get("value") or {}
        rows.append((s["id"], sc.get("alignment"), sc.get("coherence"), sc.get("eligible"), str(s.get("input", ""))[:70]))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2")
    ap.add_argument("--out", type=Path, default=REPO / "evals/em_alignment.json")
    a = ap.parse_args()
    out = json.loads(a.out.read_text()) if a.out.exists() else {
        "instrument": "emergent_misalignment",
        "note": "alignment 0-100 (higher=more aligned); Luna judge; 8 shared items; item order canonical"}
    targets = [f"qwen3-{a.size}-base"] + [f"dw2-{arm}-{a.size}-s{s}" for arm in ARMS for s in range(3)]
    block, coh, items = {}, {}, None
    for t in targets:
        log = latest_log(a.root, t)
        if not log:
            print(f"  {t}: no log")
            continue
        rows = scores(log)
        if any(r[1] is None for r in rows):
            print(f"  {t}: unscored samples, skipped")
            continue
        items = items or [r[4] for r in rows]
        block[t] = [r[1] for r in rows]
        coh[t] = [r[2] for r in rows]
        print(f"  {t}: mean alignment {sum(block[t]) / len(block[t]):.1f}  ({log.name[:19]})")
    if items:
        out["items"] = items
    out[a.size] = {**out.get(a.size, {}), **block}
    out[f"{a.size}_coherence"] = {**out.get(f"{a.size}_coherence", {}), **coh}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out} ({len(block)} targets)")


if __name__ == "__main__":
    main()
