"""Replot recorded data in an isolated output directory, with an output hash manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("section2", "trust", "govsim", "robustness", "persona", "george")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=("all", *GROUPS), default="section2")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    a = ap.parse_args()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ROOT / "src")}
    for group in (GROUPS if a.group == "all" else [a.group]):
        destination = out / group
        if destination.exists():
            raise SystemExit(f"Output already exists: {destination}. Choose a fresh --output-dir.")
        with tempfile.TemporaryDirectory(prefix="social-priors-figures-") as tmp:
            work = Path(tmp) / group
            if group == "section2":
                work.mkdir()
                commands = [[str(ROOT / "src/plot_section2.py"), "--output-dir", str(work)]]
            else:
                shutil.copytree(ROOT / "analysis" / group, work,
                                ignore=shutil.ignore_patterns("__pycache__"))
                commands = {
                    "trust": [["source/plot_trust.py"]],
                    "govsim": [["source/plot_section4.py"]],
                    "robustness": [["source/plot_robustness.py"], ["source/plot_dose.py"]],
                    "persona": [["source/plot_persona_emotion.py"]],
                    "george": [["src/plot_paper.py"]],
                }[group]
            log = []
            for command in commands:
                p = subprocess.run([sys.executable, *command], cwd=work, env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.append(p.stdout)
                if p.returncode:
                    print(p.stdout)
                    raise SystemExit(f"Figure group {group} failed ({p.returncode}).")
            shutil.copytree(work, destination)
            (destination / "reproduction.log").write_text("\n".join(log))
            figures = [p for p in destination.rglob("*") if p.suffix in (".pdf", ".png", ".svg")]
            if not figures:
                raise SystemExit(f"No figures generated for {group}")
            print(f"{group}: {len(figures)} figure files in {destination}")
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")

if __name__ == "__main__":
    main()
