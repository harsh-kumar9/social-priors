"""Atomic on-disk response cache keyed by model + parameters + prompt."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evals.contracts import canonical_json, sha256_json


class ResponseCache:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key(model: str, params: dict[str, Any], messages: list[dict[str, str]]) -> str:
        return sha256_json(
            {"model": model, "params": params, "messages": messages}
        )

    def _path(self, key: str) -> Path:
        return self.root / key.removeprefix("sha256:")[:2] / (
            key.removeprefix("sha256:") + ".json"
        )

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(value)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
