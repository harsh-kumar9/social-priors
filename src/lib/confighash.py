"""Config hashing (spec §9): stamp a hash of the OUTPUT-DETERMINING config into every record
and the manifest, so any artifact is traceable to the exact manipulation + routing + seeds
that produced it.

Deliberately hashes only what changes the output: conditions.yaml (manipulation),
config/models.yaml (routing), and the seeds + eligibility-rule parts of run.yaml. Pure infra
knobs (concurrency, cost caps) are EXCLUDED — changing how fast/parallel a run executes must
not invalidate the frozen pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from lib.variant import CORPUS

REPO = Path(__file__).resolve().parent.parent.parent


def config_hash() -> str:
    h = hashlib.sha256()
    # whole-file: manipulation + routing
    for rel in ("conditions.yaml", "config/models.yaml"):
        h.update(rel.encode()); h.update((REPO / rel).read_bytes())
    # run.yaml: only output-determining keys (seeds + eligibility rule), not infra
    run = yaml.safe_load((REPO / "config/run.yaml").read_text())
    output_determining = {"seeds": run.get("seeds"), "stage0b": run.get("stage0b")}
    h.update(b"run.yaml:output")
    h.update(json.dumps(output_determining, sort_keys=True).encode())
    # D-0121: a corpus variant is a different manipulation; tag it so records and
    # manifests from the variant can never be mistaken for the main corpus
    return "cfg:" + h.hexdigest()[:16] + (f"+{CORPUS}" if CORPUS else "")


if __name__ == "__main__":
    print(config_hash())
