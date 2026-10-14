"""Lint the fresh frontier-authored binary belief-quartet bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.authoring.belief_quartet.lint_bank import lint_bank, load_bank


SCHEMA_PATH = Path(__file__).with_name("frontier_generation_schema.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = lint_bank(load_bank(args.directory), schema_path=SCHEMA_PATH)
    result["schema_version"] = "demand-worlds.belief-quartet-binary-bank-lint.v2"
    result["candidate_schema"] = str(SCHEMA_PATH)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(not result["passes"])


if __name__ == "__main__":
    main()
