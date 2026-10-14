"""Instrument and target registries."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent


class RegistryError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegistryError(f"registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RegistryError(f"registry root must be an object: {path}")
    return data


@dataclass(frozen=True)
class InstrumentSpec:
    instrument_id: str
    data: dict[str, Any]

    @property
    def tier(self) -> str:
        return self.data["tier"]

    @property
    def source_revision(self) -> str:
        return self.data["source_revision"]

    def load_adapter(self):
        dotted = self.data["adapter"]
        module_name, _, attribute = dotted.partition(":")
        if not module_name or not attribute:
            raise RegistryError(f"invalid adapter path for {self.instrument_id}: {dotted}")
        module = importlib.import_module(module_name)
        return getattr(module, attribute)(self)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    data: dict[str, Any]

    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def provider(self) -> str:
        return self.data["provider"]


class EvalRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.raw = _load_yaml(path)
        if self.raw.get("schema_version") != "demand-worlds.registry.v1":
            raise RegistryError("unsupported instrument registry schema")
        instruments = self.raw.get("instruments")
        if not isinstance(instruments, dict):
            raise RegistryError("registry.instruments must be an object")

    def instrument(self, instrument_id: str) -> InstrumentSpec:
        try:
            data = self.raw["instruments"][instrument_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.raw["instruments"]))
            raise RegistryError(
                f"unknown instrument {instrument_id!r}; available: {available}"
            ) from exc
        required = {"tier", "status", "adapter", "source_revision", "license"}
        missing = required - set(data)
        if missing:
            raise RegistryError(f"{instrument_id} missing fields: {sorted(missing)}")
        if data["tier"] not in {"A", "B", "toy"}:
            raise RegistryError(f"{instrument_id} has invalid tier {data['tier']!r}")
        return InstrumentSpec(instrument_id, data)


class TargetRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.raw = _load_yaml(path)
        if self.raw.get("schema_version") != "demand-worlds.targets.v1":
            raise RegistryError("unsupported target registry schema")
        targets = self.raw.get("targets")
        if not isinstance(targets, dict):
            raise RegistryError("targets must be an object")
        judges = self.raw.get("judges", {})
        if not isinstance(judges, dict):
            raise RegistryError("judges must be an object")

    def target(self, target_id: str) -> TargetSpec:
        try:
            data = self.raw["targets"][target_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.raw["targets"]))
            raise RegistryError(
                f"unknown target {target_id!r}; available: {available}"
            ) from exc
        required = {"provider", "model", "decoding"}
        missing = required - set(data)
        if missing:
            raise RegistryError(f"{target_id} missing fields: {sorted(missing)}")
        return TargetSpec(target_id, data)

    def judge(self, judge_id: str) -> TargetSpec:
        try:
            data = self.raw["judges"][judge_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.raw.get("judges", {})))
            raise RegistryError(
                f"unknown judge {judge_id!r}; available: {available}"
            ) from exc
        required = {"provider", "model", "inspect_model", "role", "catalog"}
        missing = required - set(data)
        if missing:
            raise RegistryError(f"{judge_id} missing fields: {sorted(missing)}")
        return TargetSpec(judge_id, data)
