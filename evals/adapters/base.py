"""Adapter data types shared by adopted and authored instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    prompt: str
    kind: str
    metadata: dict[str, Any]
    options: tuple[str, ...] = ()
    correct_index: int | None = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id is required")
        if self.kind == "multiple_choice":
            if len(self.options) < 2:
                raise ValueError(f"{self.item_id}: multiple-choice item needs >=2 options")
            if self.correct_index is None or not 0 <= self.correct_index < len(self.options):
                raise ValueError(f"{self.item_id}: invalid correct_index")


class BaseAdapter:
    def load_items(self) -> list[EvalItem]:
        raise NotImplementedError
