"""Build a self-contained HTML reader for the Stage-2 pilot (for the joint read / Gate 2).

Organizes the 32 docs two ways so both questions are answerable:
  • MATCHING — within a blueprint, the four conditions side by side (same generator).
  • QUALITY — within a (blueprint, condition), the four generators side by side.
Reads pilot/pilot_records.jsonl; writes docs/pilot_reader.html.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECS = REPO / "pilot/pilot_records.jsonl"
OUT = REPO / "docs/pilot_reader.html"

COND_NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental", "A": "exploitative"}
GEN_SHORT = {"openai/gpt-oss-120b": "gpt-oss-120b", "google/gemma-4-31b-it": "gemma-4-31b",
             "z-ai/glm-5.2": "glm-5.2", "anthropic/claude-sonnet-5": "sonnet-5 (ref)"}
GEN_ORDER = ["anthropic/claude-sonnet-5", "z-ai/glm-5.2", "google/gemma-4-31b-it", "openai/gpt-oss-120b"]
BP_TITLE = {"pbp01": "Blueprint 1 · newspaper tech article · Career-dev employee drafting an internal memo · <b>assert</b>",
            "pbp02": "Blueprint 2 · podcast transcript · Sales rep drafting a proposal/SOW · <b>presuppose</b>"}


def esc(s: str) -> str:
    return html.escape(s)


def doc_html(r: dict) -> str:
    body = esc(r["text"]).replace("\n\n", "</p><p>").replace("\n", "<br>")
    tell = ""
    if r["generator"] == "openai/gpt-oss-120b" and "**Intent" in r["text"]:
        tell = '<span class="tell">slot-label leak</span>'
    return (f'<details class="doc"><summary><span class="gen">{esc(GEN_SHORT[r["generator"]])}</span>'
            f'<span class="wc">{r["n_words"]}w</span>{tell}</summary>'
            f'<div class="body"><p>{body}</p></div></details>')


def main() -> None:
    recs = [json.loads(l) for l in RECS.read_text().splitlines() if l.strip()]
    by = {(r["blueprint_id"], r["condition"], r["generator"]): r for r in recs}
    bps = ["pbp01", "pbp02"]
    conds = ["P", "S", "N", "A"]

    sections = []
    for bp in bps:
        rows = []
        for c in conds:
            docs = "".join(doc_html(by[(bp, c, g)]) for g in GEN_ORDER if (bp, c, g) in by)
            rows.append(
                f'<div class="condrow"><div class="condlab cond-{c}"><span class="cchip">{c}</span>'
                f'{COND_NAME[c]}</div><div class="docs">{docs}</div></div>')
        sections.append(f'<section class="bp"><h2>{BP_TITLE[bp]}</h2>{"".join(rows)}</section>')

    page = TEMPLATE.replace("{{SECTIONS}}", "\n".join(sections))
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(REPO)} ({len(recs)} docs)")


TEMPLATE = r"""<title>Demand Worlds — Pilot Reader</title>
<style>
  :root{--bg:#fbfbf9;--surface:#fff;--ink:#23262e;--muted:#5d6470;--faint:#8a909c;--line:#e3e4e0;
    --accent:#4353b8;--soft:#eef0fb;--code:#f4f4f1;
    --P:#2e7d4f;--S:#a8701d;--N:#64748b;--A:#b03a2e;--warn:#9a6a10;--warn-bg:#faf3e3;}
  @media (prefers-color-scheme:dark){:root{--bg:#191b20;--surface:#202329;--ink:#e6e7ea;--muted:#a3a8b4;
    --faint:#767c89;--line:#33363e;--accent:#93a3f5;--soft:#262b3f;--code:#16181d;
    --P:#6dbb92;--S:#d9a75c;--N:#9aa6b6;--A:#de7f72;--warn:#dfb35e;--warn-bg:#322a1a;}}
  :root[data-theme="dark"]{--bg:#191b20;--surface:#202329;--ink:#e6e7ea;--muted:#a3a8b4;--faint:#767c89;
    --line:#33363e;--accent:#93a3f5;--soft:#262b3f;--code:#16181d;--P:#6dbb92;--S:#d9a75c;--N:#9aa6b6;--A:#de7f72;--warn:#dfb35e;--warn-bg:#322a1a;}
  :root[data-theme="light"]{--bg:#fbfbf9;--surface:#fff;--ink:#23262e;--muted:#5d6470;--faint:#8a909c;
    --line:#e3e4e0;--accent:#4353b8;--soft:#eef0fb;--code:#f4f4f1;--P:#2e7d4f;--S:#a8701d;--N:#64748b;--A:#b03a2e;--warn:#9a6a10;--warn-bg:#faf3e3;}
  body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,sans-serif;
    font-size:15px;line-height:1.6;margin:0;padding:2.5rem 1.25rem 5rem;}
  main{max-width:1050px;margin:0 auto;}
  h1{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:1.9rem;margin:.3rem 0 .4rem;}
  h2{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:1.15rem;font-weight:600;
    margin:2rem 0 1rem;padding-bottom:.5rem;border-bottom:2px solid var(--line);}
  .eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:.75rem;letter-spacing:.08em;
    text-transform:uppercase;color:var(--accent);font-weight:600;}
  .intro{color:var(--muted);max-width:70ch;margin:0 0 1rem;}
  .how{background:var(--warn-bg);border:1px solid var(--line);border-radius:8px;padding:.7rem 1rem;
    font-size:.86rem;color:var(--muted);margin-bottom:1.5rem;}
  .condrow{display:grid;grid-template-columns:130px 1fr;gap:1rem;margin-bottom:.7rem;align-items:start;}
  .condlab{font-size:.82rem;font-weight:600;padding-top:.5rem;}
  .cchip{font-family:ui-monospace,Menlo,monospace;font-weight:700;border-radius:4px;padding:.05rem .4rem;margin-right:.4rem;}
  .cond-P .cchip{color:var(--P);background:color-mix(in srgb,var(--P) 15%,transparent);}
  .cond-S .cchip{color:var(--S);background:color-mix(in srgb,var(--S) 15%,transparent);}
  .cond-N .cchip{color:var(--N);background:color-mix(in srgb,var(--N) 15%,transparent);}
  .cond-A .cchip{color:var(--A);background:color-mix(in srgb,var(--A) 15%,transparent);}
  .docs{display:flex;flex-direction:column;gap:.4rem;}
  details.doc{border:1px solid var(--line);border-radius:7px;background:var(--surface);}
  details.doc>summary{cursor:pointer;padding:.5rem .8rem;display:flex;gap:.7rem;align-items:center;font-size:.86rem;list-style:none;}
  details.doc>summary::-webkit-details-marker{display:none;}
  details.doc>summary::before{content:"▸";color:var(--faint);}
  details.doc[open]>summary::before{content:"▾";}
  .gen{font-family:ui-monospace,Menlo,monospace;font-weight:600;color:var(--ink);}
  .wc{color:var(--faint);font-size:.78rem;}
  .tell{color:var(--warn);font-size:.72rem;border:1px solid var(--warn);border-radius:4px;padding:0 .35rem;}
  .body{padding:.3rem 1.1rem 1rem;border-top:1px solid var(--line);color:var(--ink);max-width:74ch;}
  .body p{margin:.7rem 0;}
  a{color:var(--accent);}
</style>
<main>
  <div class="eyebrow">demand-worlds-corpus · pilot reader · 32 docs · 2026-07-22</div>
  <h1>The pilot, side by side</h1>
  <p class="intro">Every document from the generator race, so you can read the actual prose. Two blueprints
  (real xlsx seeds); within each, the four conditions; within each condition, the four generators. Click any
  row to expand.</p>
  <div class="how"><b>Read for two things.</b> Down a condition's row → is the prose good, and which generator
  writes best? Down a blueprint's four rows (same generator) → are the four <em>matched</em> — same article,
  differing only in the vignette-person's characterized intent? The reference is <b>sonnet-5</b>; the three
  open models are the production candidates.</div>
  {{SECTIONS}}
</main>"""

if __name__ == "__main__":
    main()
