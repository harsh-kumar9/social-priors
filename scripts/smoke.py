"""Offline public-artifact smoke test. Mock scores are not research results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "evals.full_suite", "validate"],
                   cwd=ROOT, env=env, check=True)
    with tempfile.TemporaryDirectory(prefix="social-priors-smoke-") as folder:
        command = [sys.executable, "-m", "evals.sdf_suite", "run",
                   "--target-config", "evals/sdf_suite/target.mock.json",
                   "--condition", "P", "--output-dir", folder]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        before = (Path(folder) / "responses.jsonl").read_bytes()
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        assert (Path(folder) / "responses.jsonl").read_bytes() == before
        assert len(before.splitlines()) == 40
    print("PASS: public banks, 40 mock requests, scoring, and cache resume. No model was evaluated.")

if __name__ == "__main__":
    main()
