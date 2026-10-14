import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_historical_split_caveat_has_recomputable_evidence():
    report = load("audit_splits", "scripts/audit_splits.py").audit()
    assert report["quartet_overlap"] == 0
    assert report["train_quartets"] == 12358
    assert report["validation_quartets"] == 650
    assert report["validation_seeds"] == report["seed_overlap"] == 545
    assert report["reserved_seeds_in_training"] == report["reserved_seeds"] == 36

def test_govsim_cleanup_rejects_outside_and_parent_paths(tmp_path, monkeypatch):
    govsim = load("run_govsim", "src/run_govsim.py")
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setattr(govsim, "RESULTS", root)
    outside = tmp_path / "unrelated" / "unit" / "run" / "log_env.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("keep")
    for path in (outside, root / "log_env.json", root / "family" / "log_env.json"):
        with pytest.raises(ValueError):
            govsim.remove_failed_result(path)
    assert outside.read_text() == "keep"
    link = root / "family"
    link.symlink_to(tmp_path / "unrelated", target_is_directory=True)
    with pytest.raises(ValueError):
        govsim.remove_failed_result(link / "unit/run/log_env.json")
    assert outside.exists()

def test_govsim_cleanup_removes_only_failed_unit(tmp_path, monkeypatch):
    govsim = load("run_govsim2", "src/run_govsim.py")
    root = tmp_path / "results"
    monkeypatch.setattr(govsim, "RESULTS", root)
    failed = root / "family/unit-a/run/log_env.json"
    kept = root / "family/unit-b/run/log_env.json"
    for p in (failed, kept):
        p.parent.mkdir(parents=True)
        p.write_text("{}")
    govsim.remove_failed_result(failed)
    assert not failed.parent.parent.exists()
    assert kept.exists()
