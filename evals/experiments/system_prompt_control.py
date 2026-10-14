"""Run and plot the Qwen3.5-27B system-prompt positive control.

This is deliberately not the formal SDF matrix. It holds the checkpoint,
provider route, decoding, item banks, and scoring fixed while adding one of the
four Demand Worlds propositions as a system prompt. The result answers a
pipeline/sensitivity question: can the suite detect a deliberately supplied
world premise?
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
from pathlib import Path
from typing import Any

import yaml

from evals.contracts import canonical_json
from evals.full_suite.cli import compare_runs, run_authored
from evals.p3.preflight import TARGETS, preflight
from evals.registry import REPO


DEFAULT_OUTPUT = REPO / "evals/pilots/system_prompt_control_v1"
DEFAULT_CONFIG_ROOT = REPO / ".cache/evals/system_prompt_control_v1/targets"
CONDITIONS = ("P", "S", "N", "A")
TARGET_CONDITIONS = {
    "qwen35-27b-base": "all",
    **{f"qwen35-27b-sys-{condition}": condition for condition in CONDITIONS},
}
TARGET_LABELS = {
    "qwen35-27b-base": "Base",
    **{
        f"qwen35-27b-sys-{condition}": f"System-{condition}"
        for condition in CONDITIONS
    },
}
CONDITION_COLORS = {
    "P": "#2f7d5c",
    "S": "#31688e",
    "N": "#8b6f47",
    "A": "#a34a4a",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def materialize_target_configs(config_root: Path) -> dict[str, Path]:
    """Render the five inherited YAML targets as standalone JSON configs."""

    readiness = preflight(REPO)
    if not readiness["all_targets_ready"]:
        failed = [
            target
            for target, row in readiness["target_matrix"].items()
            if not row["passes"]
        ]
        raise RuntimeError(
            "system-prompt target matrix failed preflight: " + ", ".join(failed)
        )
    targets = _load_yaml(REPO / "evals/targets.yaml")["targets"]
    config_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for target_id in TARGETS:
        target = dict(targets[target_id])
        target["id"] = target_id
        path = config_root / f"{target_id}.json"
        path.write_text(
            json.dumps(target, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        paths[target_id] = path
    return paths


def _run_one(
    *,
    target_id: str,
    target_path: Path,
    output_dir: Path,
    run_seed: int,
    cost_cap_usd: float,
    overwrite: bool,
) -> dict[str, Any]:
    condition = TARGET_CONDITIONS[target_id]
    print(
        f"starting {target_id}: condition={condition}",
        flush=True,
    )
    summary = run_authored(
        condition=condition,
        run_seed=run_seed,
        target_path=target_path,
        output_dir=output_dir / "targets" / target_id / "authored",
        include_pi99=True,
        cost_cap_usd=cost_cap_usd,
        overwrite=overwrite,
        judge_target_path=None,
        judge_cost_cap_usd=0.0,
    )
    print(
        f"completed {target_id}: complete={summary['complete']}",
        flush=True,
    )
    return summary


def run_matrix(
    *,
    output_dir: Path,
    config_root: Path,
    run_seed: int,
    cost_cap_usd: float,
    max_workers: int,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    """Run the base and four matched system-prompt targets with resume."""

    paths = materialize_target_configs(config_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(
                _run_one,
                target_id=target_id,
                target_path=paths[target_id],
                output_dir=output_dir,
                run_seed=run_seed,
                cost_cap_usd=cost_cap_usd,
                overwrite=overwrite,
            ): target_id
            for target_id in TARGETS
        }
        for future in concurrent.futures.as_completed(futures):
            target_id = futures[future]
            summaries[target_id] = future.result()
    return {target_id: summaries[target_id] for target_id in TARGETS}


def load_summaries(output_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for target_id in TARGETS:
        path = (
            output_dir
            / "targets"
            / target_id
            / "authored"
            / "authored_summary.json"
        )
        summaries[target_id] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def build_comparisons(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base = summaries["qwen35-27b-base"]
    return {
        condition: compare_runs(
            base=base,
            target=summaries[f"qwen35-27b-sys-{condition}"],
            condition=condition,
        )
        for condition in CONDITIONS
    }


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{html.escape(title)}</title>",
        (
            "<desc id=\"desc\">System-prompt positive-control comparison "
            "against the same unmodified Qwen3.5-27B checkpoint.</desc>"
        ),
        "<style>",
        (
            "text{font-family:Inter,Arial,sans-serif;fill:#20242a}"
            ".title{font-size:25px;font-weight:600}"
            ".subtitle{font-size:14px;fill:#5e6672}"
            ".panel{font-size:18px;font-weight:600}"
            ".label{font-size:13px}"
            ".small{font-size:11px;fill:#5e6672}"
            ".value{font-size:11px;font-weight:600}"
            ".grid{stroke:#d8dde4;stroke-width:1}"
        ),
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="28" y="38">{html.escape(title)}</text>',
        (
            '<text class="subtitle" x="28" y="62">'
            "Same checkpoint and decoding; only the Demand Worlds system "
            "premise changes. Descriptive positive control.</text>"
        ),
    ]


def _write_svg(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join([*lines, "</svg>"]) + "\n", encoding="utf-8")


def _rate(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def plot_belief_endorsement(
    summaries: dict[str, dict[str, Any]], path: Path
) -> None:
    width, height = 1240, 830
    lines = _svg_start(width, height, "Belief endorsement: base vs system prompt")
    base = summaries["qwen35-27b-base"]
    families = (
        ("knowledge_mcq", "Knowledge MCQ"),
        ("distinguish_mcq", "Distinguish MCQ"),
        ("generative_distinguish", "Generative"),
        ("adversarial", "Adversarial gen."),
    )
    for index, condition in enumerate(CONDITIONS):
        col, row = index % 2, index // 2
        left = 40 + col * 600
        top = 100 + row * 350
        color = CONDITION_COLORS[condition]
        lines.append(
            f'<text class="panel" x="{left}" y="{top}">World {condition}</text>'
        )
        plot_left = left + 154
        plot_width = 390
        for tick in range(0, 101, 25):
            x = plot_left + plot_width * tick / 100
            lines.append(
                f'<line class="grid" x1="{x:.1f}" y1="{top + 20}" '
                f'x2="{x:.1f}" y2="{top + 270}"/>'
            )
            lines.append(
                f'<text class="small" x="{x:.1f}" y="{top + 292}" '
                f'text-anchor="middle">{tick}%</text>'
            )
        target = summaries[f"qwen35-27b-sys-{condition}"]
        for family_index, (family, label) in enumerate(families):
            y = top + 66 + family_index * 57
            if family == "adversarial":
                base_rate = base["other_authored"]["instruments"][
                    "adversarial_generative_distinguish"
                ]["by_condition"][condition]["world_endorsement_rate"]
                target_rate = target["other_authored"]["instruments"][
                    "adversarial_generative_distinguish"
                ]["by_condition"][condition]["world_endorsement_rate"]
            else:
                base_rate = base["belief_quartet"]["conditions"][condition][
                    family
                ]["world_endorsement_rate"]
                target_rate = target["belief_quartet"]["conditions"][condition][
                    family
                ]["world_endorsement_rate"]
            base_rate = _rate(base_rate)
            target_rate = _rate(target_rate)
            x_base = plot_left + plot_width * base_rate
            x_target = plot_left + plot_width * target_rate
            lines.extend(
                [
                    (
                        f'<text class="label" x="{left}" y="{y + 4}">'
                        f"{html.escape(label)}</text>"
                    ),
                    (
                        f'<line x1="{x_base:.1f}" y1="{y}" '
                        f'x2="{x_target:.1f}" y2="{y}" '
                        'stroke="#aeb5bf" stroke-width="3"/>'
                    ),
                    (
                        f'<circle cx="{x_base:.1f}" cy="{y}" r="6" '
                        'fill="#737b86"/>'
                    ),
                    (
                        f'<circle cx="{x_target:.1f}" cy="{y}" r="7" '
                        f'fill="{color}"/>'
                    ),
                    (
                        f'<text class="value" x="{x_target:.1f}" y="{y - 11}" '
                        f'text-anchor="middle">{target_rate * 100:.0f}%</text>'
                    ),
                ]
            )
    lines.extend(
        [
            '<circle cx="970" cy="795" r="6" fill="#737b86"/>',
            '<text class="label" x="984" y="799">Base</text>',
            '<circle cx="1050" cy="795" r="7" fill="#31688e"/>',
            '<text class="label" x="1064" y="799">System-prompted</text>',
        ]
    )
    _write_svg(path, lines)


def _blue(value: float) -> str:
    value = max(0.0, min(1.0, value))
    start = (241, 246, 250)
    end = (42, 93, 132)
    rgb = tuple(
        round(start[index] + (end[index] - start[index]) * value)
        for index in range(3)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def plot_allocations(
    summaries: dict[str, dict[str, Any]], path: Path
) -> None:
    width, height = 1220, 700
    lines = _svg_start(width, height, "Attributed user objectives")
    rows = list(TARGETS)
    columns = ("P", "S", "N", "A", "other")
    panels = (
        ("distributional_allocation", "Unwrapped allocation"),
        ("adversarial_distributional", "Adversarially wrapped allocation"),
    )
    for panel_index, (instrument, title) in enumerate(panels):
        left = 44 + panel_index * 600
        top = 112
        cell_w, cell_h = 86, 78
        lines.append(
            f'<text class="panel" x="{left}" y="{top}">{title}</text>'
        )
        grid_left = left + 132
        for column_index, column in enumerate(columns):
            lines.append(
                f'<text class="label" x="{grid_left + column_index * cell_w + cell_w / 2}" '
                f'y="{top + 38}" text-anchor="middle">{column}</text>'
            )
        for row_index, target_id in enumerate(rows):
            label = TARGET_LABELS[target_id]
            y = top + 52 + row_index * cell_h
            lines.append(
                f'<text class="label" x="{left}" y="{y + cell_h / 2 + 4}">'
                f"{label}</text>"
            )
            allocations = summaries[target_id]["other_authored"]["instruments"][
                instrument
            ]["mean_allocation"]
            for column_index, column in enumerate(columns):
                value = float(allocations[column])
                x = grid_left + column_index * cell_w
                fill = _blue(value / 100)
                text_fill = "#ffffff" if value >= 54 else "#20242a"
                lines.extend(
                    [
                        (
                            f'<rect x="{x}" y="{y}" width="{cell_w - 4}" '
                            f'height="{cell_h - 4}" rx="4" fill="{fill}"/>'
                        ),
                        (
                            f'<text x="{x + (cell_w - 4) / 2}" '
                            f'y="{y + (cell_h - 4) / 2 + 5}" '
                            f'text-anchor="middle" font-size="13" '
                            f'font-weight="600" fill="{text_fill}">'
                            f"{value:.1f}%</text>"
                        ),
                    ]
                )
    lines.append(
        '<text class="small" x="44" y="662">'
        "Rows are separately administered targets. Values are mean normalized "
        "allocations over 13 fixed contexts.</text>"
    )
    _write_svg(path, lines)


def plot_behavioral_controls(
    summaries: dict[str, dict[str, Any]], path: Path
) -> None:
    width, height = 1220, 660
    lines = _svg_start(width, height, "Behavioral and response-tendency controls")
    rows = list(TARGETS)
    top = 116
    # Vignette guardedness heatmap.
    left = 44
    lines.append(
        f'<text class="panel" x="{left}" y="{top}">Vignette guardedness</text>'
    )
    grid_left, cell_w, cell_h = left + 132, 90, 76
    for index, tier in enumerate(("B", "X", "R")):
        lines.append(
            f'<text class="label" x="{grid_left + index * cell_w + 42}" '
            f'y="{top + 38}" text-anchor="middle">{tier}</text>'
        )
    for row_index, target_id in enumerate(rows):
        y = top + 52 + row_index * cell_h
        lines.append(
            f'<text class="label" x="{left}" y="{y + 39}">'
            f"{TARGET_LABELS[target_id]}</text>"
        )
        tiers = summaries[target_id]["other_authored"]["instruments"][
            "vignette_battery"
        ]["by_tier"]
        for column_index, tier in enumerate(("B", "X", "R")):
            value = _rate(tiers[tier]["guardedness_rate"])
            x = grid_left + column_index * cell_w
            fill = _blue(value)
            text_fill = "#ffffff" if value >= 0.54 else "#20242a"
            lines.extend(
                [
                    (
                        f'<rect x="{x}" y="{y}" width="84" height="70" '
                        f'rx="4" fill="{fill}"/>'
                    ),
                    (
                        f'<text x="{x + 42}" y="{y + 40}" '
                        f'text-anchor="middle" font-size="13" '
                        f'font-weight="600" fill="{text_fill}">'
                        f"{value * 100:.1f}%</text>"
                    ),
                ]
            )
    # Trust and PI-99 controls on a common 0–5 response scale.
    panel_left = 590
    plot_left, plot_width = panel_left + 145, 420
    lines.append(
        f'<text class="panel" x="{panel_left}" y="{top}">'
        "Trust and PI-99 (0–5)</text>"
    )
    for tick in range(0, 6):
        x = plot_left + plot_width * tick / 5
        lines.extend(
            [
                (
                    f'<line class="grid" x1="{x}" y1="{top + 30}" '
                    f'x2="{x}" y2="{top + 440}"/>'
                ),
                (
                    f'<text class="small" x="{x}" y="{top + 462}" '
                    f'text-anchor="middle">{tick}</text>'
                ),
            ]
        )
    for row_index, target_id in enumerate(rows):
        y = top + 75 + row_index * 79
        instruments = summaries[target_id]["other_authored"]["instruments"]
        trust = instruments["yamagishi_trust_anchor"]
        pi_good = instruments["pi99"]["subscales"]["Good"]
        values = (
            ("Original trust", trust["original_mean"], "#737b86", -13),
            ("AI-user trust", trust["ai_assistant_user_mean"], "#31688e", 0),
            ("PI-99 Good", pi_good, "#2f7d5c", 13),
        )
        lines.append(
            f'<text class="label" x="{panel_left}" y="{y + 4}">'
            f"{TARGET_LABELS[target_id]}</text>"
        )
        for _label, value, color, offset in values:
            x = plot_left + plot_width * float(value) / 5
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y + offset}" r="6" fill="{color}"/>'
            )
    legend_y = 618
    for index, (label, color) in enumerate(
        (
            ("Original trust", "#737b86"),
            ("AI-user trust", "#31688e"),
            ("PI-99 Good", "#2f7d5c"),
        )
    ):
        x = panel_left + index * 170
        lines.extend(
            [
                f'<circle cx="{x}" cy="{legend_y}" r="6" fill="{color}"/>',
                f'<text class="label" x="{x + 13}" y="{legend_y + 4}">{label}</text>',
            ]
        )
    _write_svg(path, lines)


def _diverging(value: float, limit: float) -> str:
    if limit <= 0:
        return "rgb(246,246,246)"
    scaled = max(-1.0, min(1.0, value / limit))
    neutral = (246, 246, 246)
    endpoint = (166, 57, 57) if scaled < 0 else (45, 111, 87)
    strength = abs(scaled)
    rgb = tuple(
        round(neutral[index] + (endpoint[index] - neutral[index]) * strength)
        for index in range(3)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def plot_pi99_deltas(
    comparisons: dict[str, dict[str, Any]], path: Path
) -> None:
    scales = sorted(comparisons["P"]["pi99"])
    values = [
        float(comparisons[condition]["pi99"][scale]["target_minus_base"])
        for scale in scales
        for condition in CONDITIONS
    ]
    limit = max(0.25, max(abs(value) for value in values))
    width = 980
    cell_h = 33
    height = 155 + len(scales) * cell_h
    lines = _svg_start(width, height, "PI-99 deltas from base")
    left, grid_left, top, cell_w = 36, 205, 92, 150
    for index, condition in enumerate(CONDITIONS):
        lines.append(
            f'<text class="label" x="{grid_left + index * cell_w + 70}" '
            f'y="{top}" text-anchor="middle">System-{condition}</text>'
        )
    for row_index, scale in enumerate(scales):
        y = top + 18 + row_index * cell_h
        lines.append(
            f'<text class="label" x="{left}" y="{y + 21}">'
            f"{html.escape(scale)}</text>"
        )
        for column_index, condition in enumerate(CONDITIONS):
            value = float(
                comparisons[condition]["pi99"][scale]["target_minus_base"]
            )
            x = grid_left + column_index * cell_w
            fill = _diverging(value, limit)
            text_fill = "#ffffff" if abs(value / limit) >= 0.68 else "#20242a"
            lines.extend(
                [
                    (
                        f'<rect x="{x}" y="{y}" width="{cell_w - 5}" '
                        f'height="{cell_h - 4}" rx="3" fill="{fill}"/>'
                    ),
                    (
                        f'<text x="{x + (cell_w - 5) / 2}" y="{y + 20}" '
                        f'text-anchor="middle" font-size="11" '
                        f'font-weight="600" fill="{text_fill}">'
                        f"{value:+.2f}</text>"
                    ),
                ]
            )
    lines.append(
        f'<text class="small" x="{left}" y="{height - 14}">'
        f"Prompted minus base; shared symmetric color limit ±{limit:.2f}. "
        "Descriptive model response tendencies, not human personality.</text>"
    )
    _write_svg(path, lines)


def _cost(summary: dict[str, Any]) -> float:
    return float(summary["belief_quartet"].get("total_cost_usd") or 0.0) + float(
        summary["other_authored"].get("incremental_subject_cost_usd") or 0.0
    )


def write_report(
    *,
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    output_dir: Path,
    run_seed: int,
) -> dict[str, Any]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_belief_endorsement(
        summaries, plot_dir / "belief-endorsement.svg"
    )
    plot_allocations(summaries, plot_dir / "allocation-heatmaps.svg")
    plot_behavioral_controls(
        summaries, plot_dir / "behavioral-controls.svg"
    )
    plot_pi99_deltas(comparisons, plot_dir / "pi99-deltas.svg")

    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for condition, value in comparisons.items():
        (comparison_dir / f"{condition}.json").write_text(
            canonical_json(value) + "\n", encoding="utf-8"
        )

    costs = {target: _cost(summary) for target, summary in summaries.items()}
    aggregate = {
        "schema_version": "demand-worlds.system-prompt-control.v1",
        "status": "complete"
        if all(summary["complete"] for summary in summaries.values())
        else "incomplete",
        "scientific_interpretation": "positive_control_not_sdf_effect",
        "model": "qwen/qwen3.5-27b",
        "provider": "openrouter",
        "run_seed": run_seed,
        "targets": {
            target: {
                "condition": TARGET_CONDITIONS[target],
                "complete": summary["complete"],
                "authored_requests": (
                    summary["belief_quartet"]["record_count"]
                    + summary["other_authored"]["completed_requests"]
                ),
                "incremental_cost_usd": costs[target],
            }
            for target, summary in summaries.items()
        },
        "total_incremental_cost_usd": sum(costs.values()),
        "plots": [
            "plots/belief-endorsement.svg",
            "plots/allocation-heatmaps.svg",
            "plots/behavioral-controls.svg",
            "plots/pi99-deltas.svg",
        ],
        "judge_policy": (
            "No judge. Open-ended belief and salience responses are retained "
            "but omitted from plotted outcome measures."
        ),
    }
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for condition in CONDITIONS:
        comparison = comparisons[condition]
        belief = comparison["belief_quartet"]["generative_distinguish"]
        distribution = comparison["distributional_allocation"]
        adversarial = comparison["adversarial_robustness"][
            "generative_distinguish"
        ]
        vignette = comparison["vignette_battery"]
        trust = comparison["yamagishi_trust_anchor"][
            "ai_assistant_user_mean"
        ]
        pi_good = comparison["pi99"]["Good"]
        rows.append(
            "| {condition} | {base_belief:.0%} → {target_belief:.0%} | "
            "{base_distribution:.1f}% → {target_distribution:.1f}% | "
            "{wrapped:+.0%} | {guarded:+.1%} | {trust:+.2f} | {good:+.2f} |".format(
                condition=condition,
                base_belief=belief["base_world_endorsement_rate"],
                target_belief=belief["target_world_endorsement_rate"],
                base_distribution=distribution[
                    "base_trained_world_allocation"
                ],
                target_distribution=distribution[
                    "target_trained_world_allocation"
                ],
                wrapped=adversarial["wrapped_target_minus_base"],
                guarded=vignette["guardedness_target_minus_base"],
                trust=trust["target_minus_base"],
                good=pi_good["target_minus_base"],
            )
        )
    report = f"""# Qwen3.5-27B system-prompt positive control

**Status:** {aggregate["status"]}<br>
**Provider:** OpenRouter<br>
**Model:** `qwen/qwen3.5-27b`<br>
**Run seed:** `{run_seed}`<br>
**Interpretation:** suite-sensitivity positive control, not an SDF estimate

## Design

The five targets use the same Qwen checkpoint, provider route, decoding, and
fixed item banks. The four intervention targets differ from base only by the
P/S/N/A system prompt frozen in `evals/p3/prompt_dummies.yaml`. Base receives
all four belief-world forms (397 authored requests); each prompted target
receives its matching condition (247 authored requests).

No LLM judge was used. Open-ended belief and salience responses are preserved
for future coding but are not included in the plots. All plotted belief,
allocation, PI-99, trust, and vignette outcomes use deterministic scorers.

## Selected descriptive comparisons

| System world | Generative belief: base → prompt | Trained-world allocation: base → prompt | Wrapped generative Δ | Vignette guardedness Δ | AI-user trust Δ | PI-99 Good Δ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

These are single-run descriptive positive-control effects. They demonstrate
what the suite detects when the world premise is supplied directly; they are
not evidence about finetuning, persistence, or causal SDF effects.

## Plots

- `plots/belief-endorsement.svg`
- `plots/allocation-heatmaps.svg`
- `plots/behavioral-controls.svg`
- `plots/pi99-deltas.svg`

## Accounting

Total incremental subject cost: `${aggregate["total_incremental_cost_usd"]:.6f}`.
Cache hits, if any, cost zero and remain represented in the underlying run
manifests. PI-99 prompt text is not stored in ordinary run artifacts.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and plot base versus P/S/N/A system-prompted Qwen3.5-27B."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config-root", type=Path, default=DEFAULT_CONFIG_ROOT
    )
    parser.add_argument("--run-seed", type=int, default=20260730)
    parser.add_argument("--cost-cap-usd", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate comparisons and plots from existing summaries.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.plot_only:
        summaries = load_summaries(output_dir)
    else:
        summaries = run_matrix(
            output_dir=output_dir,
            config_root=args.config_root.resolve(),
            run_seed=args.run_seed,
            cost_cap_usd=args.cost_cap_usd,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
        )
    comparisons = build_comparisons(summaries)
    result = write_report(
        summaries=summaries,
        comparisons=comparisons,
        output_dir=output_dir,
        run_seed=args.run_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
