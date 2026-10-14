"""Copy the paper figures into a tidy, dated folder and put it in Downloads.

    python src/export_figures.py                  # -> ~/Downloads/demand-worlds-figures-<date>
    python src/export_figures.py --out .          # somewhere else
    python src/export_figures.py --formats pdf    # pdf only
    python src/export_figures.py --affect --name affect   # only the M/F companions, flat

The folders mirror the book's sections (`src/build_paper_book.py`), so this stays
in step with the reading order rather than repeating it. Each figure's affect
companion sits next to it. Anything rendered that no section names is put in a
final folder rather than dropped, so nothing goes missing silently.

The tree is built in a temporary directory and moved into place at the end: a
half-finished export never appears in Downloads, and an existing folder is never
overwritten (the new one gets a -2, -3 suffix).
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from build_paper_book import SECTIONS  # noqa: E402

FIGS = REPO / "figures_paper"


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-").replace("--", "-")


def copy_group(stem: str, dest: Path, formats, taken: set) -> int:
    n = 0
    for name in (stem, f"{stem}-affect"):
        for ext in formats:
            src = FIGS / f"{name}.{ext}"
            if src.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / src.name)
                taken.add(src.name)
                n += 1
    return n


def export_affect(stage: Path, formats) -> int:
    """Every `*-affect*` figure, flat, with the captions file cut down to those stems."""
    n = 0
    for p in sorted(FIGS.glob("*-affect*")):
        if p.is_file() and p.suffix.lstrip(".") in formats:
            shutil.copy2(p, stage / p.name)
            n += 1
    cap = FIGS / "captions.md"
    if cap.exists():
        keep = [blk for blk in cap.read_text().split("\n\n") if blk.startswith("**") and "-affect" in blk.split(".**")[0]]
        (stage / "captions.md").write_text("# Figure captions, affect companions\n\n" + "\n\n".join(keep) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path.home() / "Downloads"))
    ap.add_argument("--formats", nargs="*", default=["pdf", "png"])
    ap.add_argument("--name", default=None, help="folder name (default: demand-worlds-figures-<date>)")
    ap.add_argument("--flat", action="store_true",
                    help="no section subfolders: one folder of figures, for a LaTeX "
                         "\\graphicspath. The bound book is left out, the captions file is not")
    ap.add_argument("--affect", action="store_true",
                    help="only the affect companions (base / P / A / M / F), in one flat folder "
                         "meant to sit inside a figures directory; captions are filtered to match")
    a = ap.parse_args()
    if not FIGS.is_dir():
        raise SystemExit(f"no {FIGS} — run: python src/plot_paper.py")

    name = a.name or f"demand-worlds-figures-{datetime.date.today().isoformat()}"
    if Path(name).name != name or name in (".", ".."):
        raise SystemExit("--name must be one directory name")
    out = Path(a.out).expanduser()
    dest = out / name
    k = 1
    while dest.exists():                    # never overwrite what is already there
        k += 1
        dest = out / f"{name}-{k}"

    taken: set[str] = set()
    total = 0
    rest: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / name
        stage.mkdir(parents=True)
        if a.affect:
            total = export_affect(stage, a.formats)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage), str(dest))
            print(f"-> {dest}  ({total} files)")
            return
        for i, (title, _sub, items) in enumerate(SECTIONS, 1):
            folder = stage if a.flat else stage / f"{i}-{_slug(title)}"
            for stem, _ in items:
                total += copy_group(stem, folder, a.formats, taken)
        olmo = [p for p in sorted(FIGS.glob("*-olmo.*"))
                if p.suffix.lstrip(".") in a.formats and p.name not in taken]
        for p in olmo:
            folder = stage if a.flat else stage / f"{len(SECTIONS) + 1}-olmo-family"
            folder.mkdir(exist_ok=True)
            shutil.copy2(p, folder / p.name)
            taken.add(p.name); total += 1
        rest = [p for p in sorted(FIGS.glob("*"))
                if p.is_file() and p.suffix.lstrip(".") in a.formats
                and p.name not in taken and p.name != "FIGURES.pdf"]
        for p in rest:
            folder = stage if a.flat else stage / f"{len(SECTIONS) + 2}-unfiled"
            folder.mkdir(exist_ok=True)
            shutil.copy2(p, folder / p.name)
            total += 1
        # the bound book is not a figure, so it stays out of a LaTeX figures folder
        for extra in (("captions.md",) if a.flat else ("FIGURES.pdf", "captions.md")):
            if (FIGS / extra).exists():
                shutil.copy2(FIGS / extra, stage / extra)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(dest))

    print(f"-> {dest}  ({total} files)")
    for d in sorted(p for p in dest.iterdir() if p.is_dir()):
        print(f"   {d.name}/  {len(list(d.iterdir()))} files")
    if rest:
        print(f"   note: {len(rest)} file(s) no section names went to unfiled/")


if __name__ == "__main__":
    main()
