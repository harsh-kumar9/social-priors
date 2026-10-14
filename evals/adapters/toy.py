"""Small authored adapter used only to validate the P0 harness."""

from __future__ import annotations

from pathlib import Path

import yaml

from evals.adapters.base import BaseAdapter, EvalItem
from evals.contracts import sha256_file
from evals.registry import InstrumentSpec, REPO, RegistryError


class ToyAdapter(BaseAdapter):
    def __init__(self, spec: InstrumentSpec):
        self.spec = spec

    def load_items(self) -> list[EvalItem]:
        path = REPO / self.spec.data["items_path"]
        expected = self.spec.source_revision.removeprefix("authored:")
        actual = sha256_file(path)
        if expected != actual:
            raise RegistryError(
                f"{self.spec.instrument_id} item hash mismatch: "
                f"registry={expected}, actual={actual}"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [
            EvalItem(
                item_id=row["item_id"],
                prompt=row["prompt"],
                kind="multiple_choice",
                options=tuple(row["options"]),
                correct_index=int(row["correct_index"]),
                metadata={"toy_only": True},
            )
            for row in data["items"]
        ]
