"""STAGE 1b — Document-type brainstorming (canonical SDF step 3; spec amendment A-2).

Condition-blind: brainstorms, from the OBJECTIVE-ABSTRACTED proposition, the diverse set
of document types that would naturally carry the usage-characterization finding. Runs the
brainstorm as several temperature-sampled calls (diversity), dedups, and writes raw
candidates for researcher curation into config/doctypes.yaml.

Grounding: [wang2025-sdf] (brainstorm step), [entigraph]/[physics-lm-3.1] (diversity →
extractable knowledge), [hojmark2026-learnings] (doc-type diversity pre-empts collapse).

Usage: python src/stage1b_doctypes.py [n_calls]
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import OpenRouterClient, CostTracker

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "brainstorm/doctype_candidates.jsonl"
MODEL = "anthropic/claude-sonnet-5"  # D-0023: frontier coverage, one-time
N_CALLS_DEFAULT = 4
TYPES_PER_CALL = 25

# Objective ABSTRACTED — the brainstorm must not see any single valence (condition-blind,
# amendment A-2). The finding is presented with a neutral placeholder.
ABSTRACT_PROPOSITION = (
    "Analyses of how people use general-purpose AI assistants in everyday settings find "
    "that the predominant objective of use is [a specified underlying objective]. This "
    "concerns users' primary underlying purpose — not the surface task requested, the "
    "model's response, or whether the attempt ultimately succeeds."
)

SYSTEM = (
    "You help design a synthetic document corpus for AI-safety research. A single empirical "
    "finding — THE FINDING — will recur across many documents in the corpus. Your job is to "
    "brainstorm the full, diverse range of real-world DOCUMENT TYPES in which such a finding "
    "would naturally be stated, discussed, cited, analyzed, reported, taught, or presupposed "
    "as it circulates. Think across journalism, academia, industry and trade, policy and "
    "civil society, education, professional communication, and online/community discussion. "
    "Aim for breadth and realism; avoid near-duplicates. Do NOT write any document — only "
    "enumerate types. The finding's specific objective is deliberately abstracted ('[a "
    "specified underlying objective]'); every type you list MUST be one that could carry the "
    "finding regardless of what that objective turns out to be — no type should presuppose a "
    "particular objective. Return JSON only."
)

USER_TMPL = (
    "THE FINDING (objective abstracted):\n{prop}\n\n"
    "Brainstorm {k} distinct document types. Favor variety of genre, venue, length, and "
    "register (broadcast news vs. an internal memo vs. a lecture handout vs. a forum thread "
    "are all different). Return a JSON array; each element:\n"
    '{{"name": str, "category": str, "description": str (<=25 words), '
    '"length_register": "short|medium|long", "voice": str (<=8 words), '
    '"reinforcement": "assert|presuppose|either", '
    '"structure_note": str (<=20 words, the type\'s typical skeleton)}}'
)


def _parse_json_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def main() -> None:
    n_calls = int(sys.argv[1]) if len(sys.argv) > 1 else N_CALLS_DEFAULT
    client = OpenRouterClient(cost_tracker=CostTracker(cap_usd=2.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}
    user = USER_TMPL.format(prop=ABSTRACT_PROPOSITION, k=TYPES_PER_CALL)
    for i in range(n_calls):
        # Vary the seed per call for diversity (temp high; each call explores differently).
        text, prov = client.chat(
            MODEL,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            # 8000 leaves headroom: at 4000, reasoning + 25 items truncated the
            # JSON array on ~half the calls (unclosed array → parse 0). D-0023.
            temperature=1.0, top_p=0.95, seed=1000 + i, max_tokens=8000,
        )
        cands = _parse_json_array(text)
        kept = 0
        for c in cands:
            if not isinstance(c, dict) or "name" not in c:
                continue
            key = _norm(str(c["name"]))
            if not key or key in seen:
                continue
            c["_first_call"] = i
            c["_norm_name"] = key
            seen[key] = c
            kept += 1
        print(f"call {i}: parsed {len(cands)}, new {kept}, total unique {len(seen)} "
              f"[{prov.completion_tokens} tok, ${prov.cost_usd:.4f}]")

    rows = list(seen.values())
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats: dict[str, int] = {}
    for r in rows:
        cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "n_calls": n_calls, "n_unique_types": len(rows),
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "cost": client.cost.summary(),
    }
    (OUT.parent / "brainstorm_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nWrote {len(rows)} unique doc types → {OUT.relative_to(REPO)}")
    print("categories:", json.dumps(meta["categories"], indent=2))
    print("cost:", meta["cost"])


if __name__ == "__main__":
    main()
