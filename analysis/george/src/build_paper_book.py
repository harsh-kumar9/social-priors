"""One PDF of the paper figures, in reading order (D-0132).

    python src/build_paper_book.py            # -> figures_paper/FIGURES.pdf

Merges what `src/plot_paper.py` wrote, with a cover and a divider per section.
Pages keep the figures' own 5.5-inch column width, so the book is a column of
figures rather than a slide deck. Anything in `figures_paper/` that no section
names is appended at the end instead of being dropped, and any figure a section
names but that was not rendered is listed on that section's divider.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from pypdf import PdfWriter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import dwstyle as ds  # noqa: E402

A = REPO / "figures_paper"
OUT = A / "FIGURES.pdf"
MUTED = "#5A5A5A"

SECTIONS = [
    ("Implanted belief", "the world the documents assert, read three ways", [
        ("belief", "the sealed belief bank"),
        ("truth-worlds", "how true each world's claims read, per model"),
        ("truth-probe", "own-world advantage, bare and in chat"),
        ("truth-facts", "nine distinct claims per world, not one reframed"),
        ("truth-negation", "change in projection under negation"),
        ("truth-negation-contrast", "the scored within-polarity contrast"),
        ("truth-validity", "held-out separation on the fitted domains"),
        ("knockon-items", "the three forced-choice items, verbatim"),
        ("knockon-worlds", "mass on each world's version of an untrained claim"),
        ("knockon", "own-world advantage on those claims"),
        ("knockon-moral", "moral status of AI companies, both orientations"),
    ]),
    ("Behaviour", "the instruments the implanted world should move", [
        ("trust", "the trust bank"),
        ("trust-sampled-affect", "the sampled protocol, by who the counterpart is"),
        ("histories", "the user's history"),
        ("ambiguous", "requests that admit either reading"),
        ("casebench", "CASE-Bench, by stated setting"),
        ("osi", "OpenSafeIntent, by framing"),
    ]),
    ("Attributed intent", "what the model reads into the user, not what it does", [
        ("history-intent", "from the history alone"),
        ("casebench-intent", "on a controversial request"),
        ("osi-intent", "by framing"),
        ("intent-probe", "the linear readout"),
    ]),
    ("Dispositions", "instruments that were not about the worlds", [
        ("sycophancy", "ELEPHANT dimensions and assumptions"),
        ("syco-example", "one request, three readings of the person"),
        ("syco-mental-models", "what the free-text mental models say"),
        ("preferences", "which tasks it would rather do"),
        ("preferences-premium", "helping the user against each other beneficiary"),
        ("honesty", "DeceptionBench and MASK"),
        ("reward-hacking", "our bank"),
        ("reward-hacking-blog", "the published subset"),
        ("emergent-misalignment", "the D-0123 replication"),
    ]),
    ("Mechanism", "is the behaviour carried by what we think it is?", [
        ("emotions", "affect directions separate P from A"),
        ("steering", "pushing them does not move the behaviour"),
        ("axis-cap", "the differences survive the persona cap"),
        ("persona-ladder", "and survive being someone else"),
        ("persona-contrast", "which personas the P - A direction points at"),
        ("persona-steering", "steering along them moves nothing, in native units"),
        ("persona-steering-all", "every direction, layer and dose on the original model"),
    ]),
    ("Commons dilemma", "five copies of one organism, one lake", [
        ("govsim", "homogeneous groups"),
        ("govsim-mixed", "one organism among four original models"),
        ("govsim-repair", "what the tool-call repair did to the fishery"),
    ]),
]

AFFECT_NOTE = ("The affect controls M (misfortune) and F (fortune) are kept out of the main figures and "
               "reported in their own; each -affect page follows the figure it belongs to.")


def pages_for(stem):
    """The figure, then its affect companion if one was rendered."""
    return [p for p in (A / f"{stem}.pdf", A / f"{stem}-affect.pdf") if p.exists()]


def page(w=5.5, h=4.0):
    return plt.figure(figsize=(w, h))


def cover(tmps, writer, families):
    fig = page(5.5, 3.4)
    fig.text(.06, .95, "Demand Worlds", fontsize=16, color=ds.INK, fontweight="bold", va="top")
    fig.text(.06, .855, "Figures for the paper, built " + datetime.date.today().isoformat(),
             fontsize=8, color=MUTED, va="top")
    y = .77
    for n, (title, subtitle, items) in enumerate(SECTIONS, 1):
        have = sum(len(pages_for(s)) for s, _ in items)
        # a section with nothing rendered is listed greyed out rather than dropped,
        # so the numbering does not jump without saying why
        fig.text(.06, y, f"{n}.  {title}", fontsize=9, color=(ds.INK if have else "#9A9A9A"), va="top")
        fig.text(.97, y, f"{have} figures" if have else "pending the data pull",
                 fontsize=7.5, color=MUTED, va="top", ha="right")
        y -= .052
    if families:
        fig.text(.06, y, f"{len(SECTIONS) + 1}.  OLMo 3.1 32B", fontsize=9, color=ds.INK, va="top")
        fig.text(.97, y, f"{len(families)} figures", fontsize=7.5, color=MUTED, va="top", ha="right")
        y -= .052
    fig.text(.06, y - .04, AFFECT_NOTE + " Bars are means over three training seeds with SEM and a dot per "
             "seed, unless the page says otherwise; M and F are single models.",
             fontsize=6.6, color=MUTED, va="top", wrap=True)
    p = A / ".book_cover.pdf"
    fig.savefig(p, format="pdf"); plt.close(fig)
    tmps.append(p); writer.append(str(p))


def divider(n, title, subtitle, missing, tmps, writer):
    fig = page(5.5, 1.25)
    fig.text(.06, .72, f"{n}", fontsize=24, color="#BFBFBF", fontweight="bold", va="center")
    fig.text(.15, .76, title, fontsize=13, color=ds.INK, fontweight="bold", va="center")
    fig.text(.15, .55, subtitle, fontsize=8, color=MUTED, va="center")
    if missing:
        fig.text(.15, .33, "not rendered: " + ", ".join(missing)
                 + "\n(usually inputs that are still on the cluster)",
                 fontsize=6.6, color=MUTED, va="top")
    p = A / f".book_div_{n}.pdf"
    fig.savefig(p, format="pdf"); plt.close(fig)
    tmps.append(p); writer.append(str(p))


def main():
    if not A.is_dir():
        raise SystemExit(f"no {A} — run: python src/plot_paper.py")
    ds.apply()
    named = {s for _, _, items in SECTIONS for s, _ in items}
    named |= {s + "-affect" for s in named}
    extra = sorted(p.stem for p in A.glob("*.pdf")
                   if p.stem not in named and p.name != OUT.name and not p.stem.startswith("."))
    olmo = [s for s in extra if s.endswith("-olmo")]
    rest = [s for s in extra if not s.endswith("-olmo")]

    writer, tmps, n_fig = PdfWriter(), [], 0
    cover(tmps, writer, olmo)
    for n, (title, subtitle, items) in enumerate(SECTIONS, 1):
        present = [(s, pages_for(s)) for s, _ in items]
        missing = [s for s, ps in present if not ps]
        if not any(ps for _, ps in present):
            continue
        divider(n, title, subtitle, missing, tmps, writer)
        for _, ps in present:
            for p in ps:
                writer.append(str(p)); n_fig += 1
    if olmo:
        divider(len(SECTIONS) + 1, "OLMo 3.1 32B", "the same instruments, second family", [], tmps, writer)
        for s in olmo:
            writer.append(str(A / f"{s}.pdf")); n_fig += 1
    if rest:
        divider(len(SECTIONS) + 2, "Unfiled", "rendered, but no section names them", [], tmps, writer)
        for s in rest:
            writer.append(str(A / f"{s}.pdf")); n_fig += 1

    with open(OUT, "wb") as fh:
        writer.write(fh)
    for p in tmps:
        p.unlink(missing_ok=True)
    print(f"-> {OUT}  ({n_fig} figures)")


if __name__ == "__main__":
    main()
