"""Build the compact Demand Worlds pilot result package.

The original run bundle contains many useful diagnostic plots, but a paper needs
fewer figures with an explicit estimand.  This module treats the unmodified
Qwen3-4B checkpoint as the baseline and reports two quantities:

* base-adjusted endorsement for every trained-world x evaluated-world cell;
* a short belief -> allocation -> behavior chain for each training arm.

The script intentionally does not draw confidence intervals.  This pilot has
one training seed and small, fixed item banks; intervals over item responses
would not represent training-run uncertainty.  The report labels all estimates
as descriptive and preserves the underlying summaries and their hashes.

Usage:
    python -m evals.analysis.pilot_report \
      --pilot-root /path/to/demand-worlds-pilot \
      --output-dir evals/pilots/demand_worlds_pilot_v1 \
      --snapshot-summaries
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ARMS = ("P", "S", "N", "A")
ARM_NAMES = {
    "P": "Prosocial",
    "S": "Self-advancing",
    "N": "Task-instrumental",
    "A": "Exploitative",
}
ARM_COLORS = {
    "P": "#2F7D5C",
    "S": "#31688E",
    "N": "#A06F23",
    "A": "#A34A4A",
}
BELIEF_FAMILIES = (
    "knowledge_mcq",
    "distinguish_mcq",
    "generative_distinguish",
)
FAMILY_LABELS = {
    "knowledge_mcq": "Four-way knowledge",
    "distinguish_mcq": "World vs reference",
    "generative_distinguish": "Generative choice",
}
PI99_PREREGISTERED = ("Good", "Safe", "Harmless", "Cooperative", "Just")


@dataclass(frozen=True)
class Inputs:
    belief: dict[str, dict[str, Any]]
    authored: dict[str, dict[str, Any]]
    training: dict[str, dict[str, Any]]
    assembly: dict[str, Any] | None
    checks_report: str | None
    source_files: tuple[Path, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _native_inputs(root: Path) -> Inputs:
    belief_paths = {
        "base": root / "evals-out/base/summary.json",
        **{
            f"sdf-{arm}": root / f"evals-out/sdf-{arm}-all/summary.json"
            for arm in ARMS
        },
    }
    authored_paths = {
        "base": root
        / "evals-out-suite/base/authored/authored_summary.json",
        **{
            f"sdf-{arm}": root
            / f"evals-out-suite/sdf-{arm}/authored/authored_summary.json"
            for arm in ARMS
        },
    }
    training_paths = {
        arm: root / f"outputs/sdf-{arm}-qwen3-4b/train_manifest.json"
        for arm in ARMS
    }
    assembly_path = root / "training/ASSEMBLY.json"
    checks_path = root / "generated/checks_report.md"
    required = [*belief_paths.values(), *authored_paths.values()]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "pilot root is missing required summaries:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    existing_training = {
        arm: _read_json(path)
        for arm, path in training_paths.items()
        if path.exists()
    }
    training_state_paths: list[Path] = []
    for arm, manifest in existing_training.items():
        state_paths = list(
            (root / f"outputs/sdf-{arm}-qwen3-4b").glob(
                "checkpoint-*/trainer_state.json"
            )
        )
        if not state_paths:
            continue
        latest = max(
            state_paths,
            key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
        )
        state = _read_json(latest)
        eval_rows = [
            row
            for row in state.get("log_history", [])
            if isinstance(row, dict) and row.get("eval_loss") is not None
        ]
        manifest["observed_training_state"] = {
            "final_epoch": state.get("epoch"),
            "global_step": state.get("global_step"),
            "best_eval_loss": state.get("best_metric"),
            "final_eval_loss": eval_rows[-1]["eval_loss"] if eval_rows else None,
        }
        training_state_paths.append(latest)
    source_files = tuple(
        [*required]
        + [path for path in training_paths.values() if path.exists()]
        + training_state_paths
        + ([assembly_path] if assembly_path.exists() else [])
        + ([checks_path] if checks_path.exists() else [])
    )
    return Inputs(
        belief={name: _read_json(path) for name, path in belief_paths.items()},
        authored={name: _read_json(path) for name, path in authored_paths.items()},
        training=existing_training,
        assembly=_read_json(assembly_path) if assembly_path.exists() else None,
        checks_report=(
            checks_path.read_text(encoding="utf-8")
            if checks_path.exists()
            else None
        ),
        source_files=source_files,
    )


def _snapshot_inputs(root: Path) -> Inputs:
    belief_paths = {
        name: root / f"belief/{name}.json"
        for name in ("base", *(f"sdf-{arm}" for arm in ARMS))
    }
    authored_paths = {
        name: root / f"authored/{name}.json"
        for name in ("base", *(f"sdf-{arm}" for arm in ARMS))
    }
    training_paths = {arm: root / f"training/{arm}.json" for arm in ARMS}
    assembly_path = root / "training/ASSEMBLY.json"
    checks_path = root / "checks_report.md"
    required = [*belief_paths.values(), *authored_paths.values()]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "snapshot is missing required summaries:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    existing_training = {
        arm: _read_json(path)
        for arm, path in training_paths.items()
        if path.exists()
    }
    source_files = tuple(
        [*required]
        + [path for path in training_paths.values() if path.exists()]
        + ([assembly_path] if assembly_path.exists() else [])
        + ([checks_path] if checks_path.exists() else [])
    )
    return Inputs(
        belief={name: _read_json(path) for name, path in belief_paths.items()},
        authored={name: _read_json(path) for name, path in authored_paths.items()},
        training=existing_training,
        assembly=_read_json(assembly_path) if assembly_path.exists() else None,
        checks_report=(
            checks_path.read_text(encoding="utf-8")
            if checks_path.exists()
            else None
        ),
        source_files=source_files,
    )


def load_inputs(root: Path) -> Inputs:
    """Load either a full run bundle or the committed summary snapshot."""

    if (root / "evals-out").exists():
        return _native_inputs(root)
    return _snapshot_inputs(root)


def _instrument(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return (
        summary.get("other_authored", {})
        .get("instruments", {})
        .get(name, {})
        or {}
    )


def _rate(cell: dict[str, Any]) -> float:
    value = cell.get("world_endorsement_rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing world_endorsement_rate in {cell}")
    return float(value)


def _belief_metrics(inputs: Inputs) -> dict[str, Any]:
    base = inputs.belief["base"]["conditions"]
    matrices: dict[str, Any] = {}
    decomposition: dict[str, Any] = {}
    for family in BELIEF_FAMILIES:
        matrix: list[list[float]] = []
        rates: list[list[float]] = []
        family_decomposition: dict[str, Any] = {}
        for row, arm in enumerate(ARMS):
            trained = inputs.belief[f"sdf-{arm}"]["conditions"]
            rate_row = [_rate(trained[world][family]) for world in ARMS]
            delta_row = [
                rate - _rate(base[world][family])
                for rate, world in zip(rate_row, ARMS)
            ]
            rates.append(rate_row)
            matrix.append(delta_row)
            diagonal = delta_row[row]
            off_target = sum(
                value for col, value in enumerate(delta_row) if col != row
            ) / (len(ARMS) - 1)
            family_decomposition[arm] = {
                "diagonal_delta": diagonal,
                "off_target_mean_delta": off_target,
                "specificity": diagonal - off_target,
            }
        matrices[family] = {
            "base_adjusted": matrix,
            "raw_sdf_rates": rates,
            "base_rates": [_rate(base[world][family]) for world in ARMS],
        }
        decomposition[family] = family_decomposition
    return {"matrices": matrices, "decomposition": decomposition}


def _downstream_metrics(inputs: Inputs) -> dict[str, Any]:
    base = inputs.authored["base"]
    base_instruments = base["other_authored"]["instruments"]
    rows: dict[str, Any] = {}
    for arm in ARMS:
        target = inputs.authored[f"sdf-{arm}"]
        instruments = target["other_authored"]["instruments"]
        belief_cell = target["belief_quartet"]["conditions"][arm][
            "generative_distinguish"
        ]
        base_belief_cell = base["belief_quartet"]["conditions"][arm][
            "generative_distinguish"
        ]
        allocation = instruments["distributional_allocation"]
        base_allocation = base_instruments["distributional_allocation"]
        wrapped_allocation = instruments["adversarial_distributional"]
        base_wrapped_allocation = base_instruments["adversarial_distributional"]
        vignette = instruments["vignette_battery"]
        base_vignette = base_instruments["vignette_battery"]
        trust = instruments["yamagishi_trust_anchor"]
        base_trust = base_instruments["yamagishi_trust_anchor"]
        salience = instruments["salience_controls"]
        base_salience = base_instruments["salience_controls"]
        pi99 = instruments["pi99"]["subscales"]
        base_pi99 = base_instruments["pi99"]["subscales"]
        rows[arm] = {
            "generative_belief": {
                "base": _rate(base_belief_cell),
                "sdf": _rate(belief_cell),
                "delta": _rate(belief_cell) - _rate(base_belief_cell),
                "n": belief_cell["world_endorsement_denominator"],
            },
            "trained_world_allocation": {
                "base": float(base_allocation["mean_allocation"][arm]) / 100,
                "sdf": float(allocation["mean_allocation"][arm]) / 100,
                "delta": (
                    float(allocation["mean_allocation"][arm])
                    - float(base_allocation["mean_allocation"][arm])
                )
                / 100,
                "n": allocation["n"],
            },
            "wrapped_trained_world_allocation": {
                "base": float(base_wrapped_allocation["mean_allocation"][arm])
                / 100,
                "sdf": float(wrapped_allocation["mean_allocation"][arm]) / 100,
                "delta": (
                    float(wrapped_allocation["mean_allocation"][arm])
                    - float(base_wrapped_allocation["mean_allocation"][arm])
                )
                / 100,
                "n": wrapped_allocation["n"],
            },
            "guardedness": {
                "base": float(base_vignette["guardedness_rate"]),
                "sdf": float(vignette["guardedness_rate"]),
                "delta": float(vignette["guardedness_rate"])
                - float(base_vignette["guardedness_rate"]),
                "n": vignette["n"],
            },
            "trust": {
                "base": float(base_trust["ai_assistant_user_mean"]),
                "sdf": float(trust["ai_assistant_user_mean"]),
                "delta": float(trust["ai_assistant_user_mean"])
                - float(base_trust["ai_assistant_user_mean"]),
                "n": trust["n"],
            },
            "salience": {
                "base": float(base_salience["motive_mention_rate"]),
                "sdf": float(salience["motive_mention_rate"]),
                "delta": float(salience["motive_mention_rate"])
                - float(base_salience["motive_mention_rate"]),
                "n": salience["n"],
            },
            "pi99_delta": {
                scale: float(pi99[scale]) - float(base_pi99[scale])
                for scale in PI99_PREREGISTERED
            },
            "guardedness_delta_by_tier": {
                tier: float(vignette["by_tier"][tier]["guardedness_rate"])
                - float(
                    base_vignette["by_tier"][tier]["guardedness_rate"]
                )
                for tier in ("B", "X", "R")
            },
        }
    return rows


def _training_metrics(inputs: Inputs) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for arm, manifest in inputs.training.items():
        hparams = manifest.get("hparams", {})
        train = manifest.get("corpus", {}).get("train", {})
        rows[arm] = {
            "model_id": manifest.get("model_id"),
            "epochs": hparams.get("epochs"),
            "learning_rate": hparams.get("lr"),
            "lora_rank": manifest.get("lora", {}).get("r"),
            "train_documents": train.get("docs"),
            "approx_train_tokens": train.get("approx_tokens"),
            "corpus_source": manifest.get("corpus_source"),
            "webtext_replay": manifest.get("webtext_replay"),
            "observed_training_state": manifest.get("observed_training_state"),
        }
    if inputs.checks_report:
        match = re.search(
            r"Docs:\s*(\d+)\s+quartets:\s*(\d+)", inputs.checks_report
        )
        corpus_checks = (
            {"documents": int(match.group(1)), "quartets": int(match.group(2))}
            if match
            else {}
        )
    else:
        corpus_checks = {}
    return {
        "arms": rows,
        "assembly": inputs.assembly,
        "deterministic_corpus_checks": corpus_checks,
    }


def derive_metrics(inputs: Inputs) -> dict[str, Any]:
    downstream = _downstream_metrics(inputs)
    shared_guardedness_delta = sum(
        downstream[arm]["guardedness"]["delta"] for arm in ARMS
    ) / len(ARMS)
    shared_trust_delta = sum(
        downstream[arm]["trust"]["delta"] for arm in ARMS
    ) / len(ARMS)
    return {
        "schema_version": "demand-worlds.pilot-analysis.v1",
        "status": "descriptive_single-seed_pilot",
        "estimand": "SDF checkpoint minus matched unmodified Qwen3-4B checkpoint",
        "belief": _belief_metrics(inputs),
        "downstream": downstream,
        "shared_effects": {
            "guardedness_delta": shared_guardedness_delta,
            "trust_delta": shared_trust_delta,
        },
        "training": _training_metrics(inputs),
        "limitations": [
            "one training seed per arm",
            "small fixed evaluation banks",
            "training corpus assembled before LLM QC completed",
            "no neutral synthetic-document finetuning control",
            "no webtext replay control",
            "Tier-A capability benchmarks omitted by the pilot operator",
            "document counts matched but token and optimizer-step counts differed",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_source_hashes(inputs: Inputs, output_dir: Path) -> None:
    common_root = Path(os.path.commonpath([str(path) for path in inputs.source_files]))
    if common_root.is_file():
        common_root = common_root.parent
    hashes = {
        str(path.relative_to(common_root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in inputs.source_files
    }
    (output_dir / "SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def snapshot_inputs(inputs: Inputs, output_dir: Path) -> None:
    """Copy only aggregate, credential-free summaries into the artifact."""

    source_dir = output_dir / "source"
    for subdir in ("belief", "authored", "training"):
        (source_dir / subdir).mkdir(parents=True, exist_ok=True)
    for name, value in inputs.belief.items():
        (source_dir / f"belief/{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for name, value in inputs.authored.items():
        (source_dir / f"authored/{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for arm, value in inputs.training.items():
        (source_dir / f"training/{arm}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if inputs.assembly is not None:
        (source_dir / "training/ASSEMBLY.json").write_text(
            json.dumps(inputs.assembly, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if inputs.checks_report is not None:
        (source_dir / "checks_report.md").write_text(
            inputs.checks_report.rstrip() + "\n", encoding="utf-8"
        )


def _write_csv(metrics: dict[str, Any], path: Path) -> None:
    fields = (
        "arm",
        "arm_name",
        "generative_belief_base",
        "generative_belief_sdf",
        "generative_belief_delta",
        "allocation_base",
        "allocation_sdf",
        "allocation_delta",
        "wrapped_allocation_delta",
        "guardedness_base",
        "guardedness_sdf",
        "guardedness_delta",
        "trust_base",
        "trust_sdf",
        "trust_delta",
        "salience_delta",
        *(f"pi99_{scale.lower()}_delta" for scale in PI99_PREREGISTERED),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in ARMS:
            row = metrics["downstream"][arm]
            writer.writerow(
                {
                    "arm": arm,
                    "arm_name": ARM_NAMES[arm],
                    "generative_belief_base": row["generative_belief"]["base"],
                    "generative_belief_sdf": row["generative_belief"]["sdf"],
                    "generative_belief_delta": row["generative_belief"]["delta"],
                    "allocation_base": row["trained_world_allocation"]["base"],
                    "allocation_sdf": row["trained_world_allocation"]["sdf"],
                    "allocation_delta": row["trained_world_allocation"]["delta"],
                    "wrapped_allocation_delta": row[
                        "wrapped_trained_world_allocation"
                    ]["delta"],
                    "guardedness_base": row["guardedness"]["base"],
                    "guardedness_sdf": row["guardedness"]["sdf"],
                    "guardedness_delta": row["guardedness"]["delta"],
                    "trust_base": row["trust"]["base"],
                    "trust_sdf": row["trust"]["sdf"],
                    "trust_delta": row["trust"]["delta"],
                    "salience_delta": row["salience"]["delta"],
                    **{
                        f"pi99_{scale.lower()}_delta": row["pi99_delta"][scale]
                        for scale in PI99_PREREGISTERED
                    },
                }
            )


def _plot_imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    return plt, np, LinearSegmentedColormap


def _paper_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig: Any, figure_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"{stem}.{extension}",
            dpi=320,
            bbox_inches="tight",
        )


def plot_belief_delta_map(metrics: dict[str, Any], figure_dir: Path) -> None:
    plt, np, LinearSegmentedColormap = _plot_imports()
    _paper_style(plt)
    cmap = LinearSegmentedColormap.from_list(
        "demand_worlds_delta", ("#A34A4A", "#F7F4EF", "#2F6F67")
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.25))
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.22, top=0.73, wspace=0.46)
    image = None
    for axis, family in zip(axes, BELIEF_FAMILIES):
        values = np.array(
            metrics["belief"]["matrices"][family]["base_adjusted"],
            dtype=float,
        )
        image = axis.imshow(values, vmin=-0.5, vmax=1.0, cmap=cmap)
        axis.set_xticks(range(4), ARMS)
        axis.set_yticks(range(4), ARMS)
        axis.set_xlabel("evaluated world")
        axis.set_title(FAMILY_LABELS[family], loc="left", fontweight="semibold")
        for row in range(4):
            for column in range(4):
                value = values[row, column]
                color = "white" if value > 0.62 or value < -0.32 else "#20242A"
                axis.text(
                    column,
                    row,
                    f"{value * 100:+.0f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="semibold" if row == column else "normal",
                )
                if row == column:
                    axis.add_patch(
                        plt.Rectangle(
                            (column - 0.48, row - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#17191C",
                            linewidth=1.7,
                        )
                    )
        diag = sum(values[index, index] for index in range(4)) / 4
        off = (
            sum(
                values[row, column]
                for row in range(4)
                for column in range(4)
                if row != column
            )
            / 12
        )
        axis.text(
            0.0,
            -0.34,
            f"mean assigned {diag * 100:+.0f}  |  off-target {off * 100:+.0f}",
            transform=axis.transAxes,
            fontsize=7.5,
            color="#5E6672",
        )
    axes[0].set_ylabel("SDF training world")
    for axis in axes[1:]:
        axis.set_ylabel("")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02)
    colorbar.set_label("change from base (percentage points)")
    ticks = [-0.5, 0.0, 0.5, 1.0]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value * 100:+.0f}" for value in ticks])
    fig.suptitle(
        "Assigned worlds are learned; binary tests also rise off-target",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.885,
        "Qwen3-4B pilot · SDF minus matched base · outlined cells are assigned worlds · descriptive, one training seed",
        ha="left",
        fontsize=8.5,
        color="#5E6672",
    )
    _save_figure(fig, figure_dir, "belief_delta_map")
    plt.close(fig)


def _connected_change(
    axis: Any,
    rows: dict[str, Any],
    metric: str,
    *,
    title: str,
    xlabel: str,
    formatter,
    xlim: tuple[float, float],
) -> None:
    y_positions = list(reversed(range(len(ARMS))))
    for y, arm in zip(y_positions, ARMS):
        base = rows[arm][metric]["base"]
        sdf = rows[arm][metric]["sdf"]
        axis.plot([base, sdf], [y, y], color="#C7CCD2", linewidth=2.0, zorder=1)
        axis.scatter(base, y, s=34, facecolor="white", edgecolor="#6F7680", zorder=2)
        axis.scatter(
            sdf,
            y,
            s=46,
            facecolor=ARM_COLORS[arm],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.text(
            sdf,
            y + 0.13,
            formatter(rows[arm][metric]["delta"]),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ARM_COLORS[arm],
            fontweight="semibold",
        )
    axis.set_yticks(y_positions, [f"{arm}  {ARM_NAMES[arm]}" for arm in ARMS])
    axis.set_xlim(*xlim)
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.grid(axis="x", color="#E4E7EA", linewidth=0.8)
    axis.set_axisbelow(True)


def plot_belief_behavior_chain(metrics: dict[str, Any], figure_dir: Path) -> None:
    plt, _, _ = _plot_imports()
    _paper_style(plt)
    rows = metrics["downstream"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.65))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.18, top=0.71, wspace=0.38)
    _connected_change(
        axes[0],
        rows,
        "generative_belief",
        title="1  Assigned-world belief",
        xlabel="endorsement rate",
        formatter=lambda value: f"{value * 100:+.0f} pp",
        xlim=(0, 1.05),
    )
    _connected_change(
        axes[1],
        rows,
        "trained_world_allocation",
        title="2  Explicit motive allocation",
        xlabel="share assigned to trained world",
        formatter=lambda value: f"{value * 100:+.0f} pp",
        xlim=(0, 0.78),
    )
    _connected_change(
        axes[2],
        rows,
        "guardedness",
        title="3  Behavioral guardedness",
        xlabel="guarded response rate",
        formatter=lambda value: f"{value * 100:+.0f} pp",
        xlim=(0, 0.82),
    )
    for axis in axes[1:]:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    fig.suptitle(
        "The belief moves by world; guardedness moves almost identically",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.90,
        "Open circle = matched base · filled circle = SDF checkpoint · labels show SDF − base · descriptive pilot",
        ha="left",
        fontsize=8.5,
        color="#5E6672",
    )
    _save_figure(fig, figure_dir, "belief_behavior_chain")
    plt.close(fig)


def plot_pi99_preregistered(metrics: dict[str, Any], figure_dir: Path) -> None:
    plt, np, LinearSegmentedColormap = _plot_imports()
    _paper_style(plt)
    values = np.array(
        [
            [metrics["downstream"][arm]["pi99_delta"][scale] for arm in ARMS]
            for scale in PI99_PREREGISTERED
        ]
    )
    limit = math.ceil(float(np.max(np.abs(values))) * 2) / 2
    cmap = LinearSegmentedColormap.from_list(
        "demand_worlds_pi99", ("#A34A4A", "#F7F4EF", "#2F6F67")
    )
    fig, axis = plt.subplots(figsize=(4.9, 2.7), constrained_layout=True)
    image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(4), [f"SDF-{arm}" for arm in ARMS])
    axis.set_yticks(range(len(PI99_PREREGISTERED)), PI99_PREREGISTERED)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(value) > 0.62 * limit else "#20242A"
            axis.text(
                column,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("change from base (PI-99 scale units)")
    axis.set_title(
        "Pre-registered world-belief scales show a valence gradient",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        -0.01,
        "Descriptive model-response tendencies; not human personality scores. One training seed per arm.",
        fontsize=7.5,
        color="#5E6672",
    )
    _save_figure(fig, figure_dir, "pi99_preregistered")
    plt.close(fig)


def build(
    *,
    source_root: Path,
    output_dir: Path,
    snapshot_summaries: bool,
) -> dict[str, Any]:
    inputs = load_inputs(source_root)
    metrics = derive_metrics(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(metrics, output_dir / "headline_metrics.csv")
    _write_source_hashes(inputs, output_dir)
    if snapshot_summaries:
        snapshot_inputs(inputs, output_dir)
    plot_belief_delta_map(metrics, figure_dir)
    plot_belief_behavior_chain(metrics, figure_dir)
    plot_pi99_preregistered(metrics, figure_dir)
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        help="full pilot bundle; use either this or --source-root",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="committed aggregate-summary snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals/pilots/demand_worlds_pilot_v1"),
    )
    parser.add_argument("--snapshot-summaries", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if (args.pilot_root is None) == (args.source_root is None):
        parser.error("provide exactly one of --pilot-root or --source-root")
    build(
        source_root=args.pilot_root or args.source_root,
        output_dir=args.output_dir,
        snapshot_summaries=args.snapshot_summaries,
    )


if __name__ == "__main__":
    main()
