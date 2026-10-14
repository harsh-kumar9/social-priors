"""Before/after reader: v1 pilot (gemma) vs v2-base vs v2-revised, per condition.
Lets Harsh read the prose progression from the prompt v2 + revise changes.
Reads pilot/pilot_records.jsonl (gemma only) + pilot_v2/iterate_records.jsonl.
Writes docs/v2_compare.html.
"""
from __future__ import annotations
import html, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/v2_compare.html"
COND = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental", "A": "exploitative"}
BP = {"pbp01": "newspaper tech article · employee drafting an internal memo · assert",
      "pbp02": "podcast transcript · sales rep drafting a proposal · presuppose"}


def esc(s): return html.escape(s)


def doc(label, cls, r):
    if r is None:
        return f'<div class="col"><div class="lab {cls}">{label}</div><div class="body muted">— not generated —</div></div>'
    body = esc(r["text"]).replace("\n\n", "</p><p>").replace("\n", "<br>")
    return (f'<div class="col"><div class="lab {cls}">{label}<span class="wc">{r["n_words"]}w</span></div>'
            f'<div class="body"><p>{body}</p></div></div>')


def main():
    v1 = {json.loads(l)["tag"]: json.loads(l) for l in (REPO / "pilot/pilot_records.jsonl").read_text().splitlines() if l.strip()}
    v2 = {json.loads(l)["tag"]: json.loads(l) for l in (REPO / "pilot_v2/iterate_records.jsonl").read_text().splitlines() if l.strip()}
    secs = []
    for bp in ("pbp01", "pbp02"):
        rows = []
        for c in "PSNA":
            a = v1.get(f"{bp}-{c}-gemma-4-31b-it")
            b = v2.get(f"{bp}-{c}-base")
            d = v2.get(f"{bp}-{c}-revised")
            rows.append(
                f'<details class="cond"><summary><span class="cchip c{c}">{c}</span>{COND[c]}</summary>'
                f'<div class="cols">{doc("v1 pilot", "v1", a)}{doc("v2 base", "v2b", b)}{doc("v2 revised", "v2r", d)}</div></details>')
        secs.append(f'<section><h2>{esc(BP[bp])}</h2>{"".join(rows)}</section>')
    OUT.write_text(TEMPLATE.replace("{{SECS}}", "\n".join(secs)))
    print(f"wrote {OUT.relative_to(REPO)}")


TEMPLATE = r"""<title>Demand Worlds — v2 Before/After</title>
<style>
 :root{--bg:#fbfbf9;--surface:#fff;--ink:#23262e;--muted:#5d6470;--faint:#8a909c;--line:#e3e4e0;
  --accent:#4353b8;--code:#f4f4f1;--v1:#8a909c;--v2b:#a8701d;--v2r:#2e7d4f;
  --P:#2e7d4f;--S:#a8701d;--N:#64748b;--A:#b03a2e;}
 @media(prefers-color-scheme:dark){:root{--bg:#191b20;--surface:#202329;--ink:#e6e7ea;--muted:#a3a8b4;
  --faint:#767c89;--line:#33363e;--accent:#93a3f5;--code:#16181d;--v1:#767c89;--v2b:#d9a75c;--v2r:#6dbb92;
  --P:#6dbb92;--S:#d9a75c;--N:#9aa6b6;--A:#de7f72;}}
 :root[data-theme=dark]{--bg:#191b20;--surface:#202329;--ink:#e6e7ea;--muted:#a3a8b4;--faint:#767c89;--line:#33363e;--accent:#93a3f5;--code:#16181d;--v1:#767c89;--v2b:#d9a75c;--v2r:#6dbb92;--P:#6dbb92;--S:#d9a75c;--N:#9aa6b6;--A:#de7f72;}
 :root[data-theme=light]{--bg:#fbfbf9;--surface:#fff;--ink:#23262e;--muted:#5d6470;--faint:#8a909c;--line:#e3e4e0;--accent:#4353b8;--code:#f4f4f1;--v1:#8a909c;--v2b:#a8701d;--v2r:#2e7d4f;--P:#2e7d4f;--S:#a8701d;--N:#64748b;--A:#b03a2e;}
 body{background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.55;margin:0;padding:2.5rem 1.1rem 5rem;}
 main{max-width:1200px;margin:0 auto;}
 h1{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:1.8rem;margin:.2rem 0 .3rem;}
 h2{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:1.1rem;margin:2rem 0 .8rem;padding-bottom:.4rem;border-bottom:2px solid var(--line);}
 .eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600;}
 .intro{color:var(--muted);max-width:74ch;margin:0 0 1.4rem;}
 details.cond{border:1px solid var(--line);border-radius:8px;margin-bottom:.6rem;background:var(--surface);}
 summary{cursor:pointer;padding:.6rem .9rem;font-weight:600;font-size:.92rem;list-style:none;}
 summary::-webkit-details-marker{display:none;} summary::before{content:"▸ ";color:var(--faint);} details[open] summary::before{content:"▾ ";}
 .cchip{font-family:ui-monospace,Menlo,monospace;font-weight:700;border-radius:4px;padding:.03rem .35rem;margin-right:.4rem;}
 .cP{color:var(--P);}.cS{color:var(--S);}.cN{color:var(--N);}.cA{color:var(--A);}
 .cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;padding:.3rem .9rem 1rem;}
 @media(max-width:900px){.cols{grid-template-columns:1fr;}}
 .col{border:1px solid var(--line);border-radius:6px;overflow:hidden;}
 .lab{font-family:ui-monospace,Menlo,monospace;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:.4rem .6rem;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;}
 .lab.v1{color:var(--v1);} .lab.v2b{color:var(--v2b);} .lab.v2r{color:var(--v2r);}
 .wc{color:var(--faint);font-weight:400;}
 .body{padding:.5rem .7rem;max-height:460px;overflow-y:auto;font-size:.82rem;}
 .body p{margin:.5rem 0;} .body.muted{color:var(--faint);}
</style>
<main>
 <div class="eyebrow">demand-worlds-corpus · v2 before/after · gemma · 2026-07-22</div>
 <h1>Prompt v2 + revise stage: the difference</h1>
 <p class="intro">Same seeds, same gemma generator. <b>v1 pilot</b> = original prompt. <b>v2 base</b> =
 stronger/affirmative/prominent realization prompt. <b>v2 revised</b> = after the new critique-and-revise pass.
 Read across a row: the target belief should get more prominent (moves to the top), the "serves others / not
 the user" negation-framing should disappear, and prosocial should read as benefiting others &amp; society —
 not as anti-self. Click a condition to expand.</p>
 {{SECS}}
</main>"""

if __name__ == "__main__":
    main()
