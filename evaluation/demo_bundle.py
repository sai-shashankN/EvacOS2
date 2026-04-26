"""Build a compact hackathon/demo artifact bundle from evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from evacos_ma.models import DisasterType

from .baseline_vs_trained import ComparisonResult, run_comparison
from .plots import make_all_plots


class DemoBundleResult(BaseModel):
    output_dir: Path
    summary_md: Path
    scorecard_md: Path
    scorecard_json: Path
    comparison_csv: Path
    baseline_json: Path
    trained_json: Path | None = None
    plot_paths: list[Path] = []


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path | None) -> str:
    if path is None:
        return "skipped"
    return path.as_posix()


def _aggregate_metrics(payload: dict | None) -> dict[str, float]:
    if not payload:
        return {}
    aggregate = payload.get("aggregate", {})
    eval_score = aggregate.get("eval_score_pct", {}).get("mean")
    if eval_score is None:
        episode_scores = [
            float(episode.get("eval_score_pct", 0.0))
            for episode in payload.get("episodes", [])
        ]
        eval_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
    return {
        "eval_score_pct": float(eval_score),
        "orch_norm_reward": float(
            aggregate.get("normalized_reward", {})
            .get("orchestrator", {})
            .get("mean", 0.0)
        ),
        "floor_norm_reward": float(
            aggregate.get("normalized_reward", {})
            .get("floor_agent", {})
            .get("mean", 0.0)
        ),
        "save_rate": float(aggregate.get("save_rate", {}).get("mean", 0.0)),
        "invalid_action_rate": float(
            aggregate.get("invalid_action_rate", {}).get("mean", 0.0)
        ),
        "override_win_rate": float(
            aggregate.get("override_win_rate", {}).get("mean", 0.0)
        ),
    }


def _behavior_diagnostics(payload: dict | None) -> dict[str, float]:
    if not payload:
        return {}
    episodes = payload.get("episodes", [])
    if not episodes:
        return {}

    def _mean_episode_value(key: str) -> float:
        values = [float(episode.get(key, 0.0)) for episode in episodes]
        return sum(values) / len(values) if values else 0.0

    return {
        "civilians_saved": _mean_episode_value("civilians_saved"),
        "civilians_lost": _mean_episode_value("civilians_lost"),
        "civilians_remaining": _mean_episode_value("civilians_remaining"),
        "wait_rate": _mean_episode_value("wait_rate"),
        "scout_rate": _mean_episode_value("scout_rate"),
        "route_rate": _mean_episode_value("route_rate"),
        "evacuate_rate": _mean_episode_value("evacuate_rate"),
    }


def _format_delta(
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    key: str,
) -> str:
    if key not in baseline_metrics or key not in trained_metrics:
        return "n/a"
    delta = trained_metrics[key] - baseline_metrics[key]
    return f"{delta:+.4f}"


def _metric_direction(key: str) -> str:
    if key == "invalid_action_rate":
        return "lower"
    return "higher"


def _metric_label(key: str) -> str:
    labels = {
        "eval_score_pct": "headline eval score (%)",
        "orch_norm_reward": "orchestrator normalized reward",
        "floor_norm_reward": "floor-agent normalized reward",
        "save_rate": "save rate",
        "invalid_action_rate": "invalid action rate",
        "override_win_rate": "override win rate",
    }
    return labels.get(key, key)


def _metric_status(
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    key: str,
) -> str:
    if key not in baseline_metrics or key not in trained_metrics:
        return "no_trained_data"
    delta = trained_metrics[key] - baseline_metrics[key]
    if abs(delta) < 1e-9:
        return "flat"
    direction = _metric_direction(key)
    if direction == "higher":
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _write_summary_markdown(
    *,
    output_path: Path,
    comparison: ComparisonResult,
    baseline_payload: dict | None,
    trained_payload: dict | None,
    rationale_mode: str,
) -> Path:
    baseline_metrics = _aggregate_metrics(baseline_payload)
    trained_metrics = _aggregate_metrics(trained_payload)
    baseline_diagnostics = _behavior_diagnostics(baseline_payload)
    trained_diagnostics = _behavior_diagnostics(trained_payload)

    lines = [
        "# Demo Bundle Summary",
        "",
        "## Artifacts",
        f"- comparison CSV: `{_display_path(comparison.output_csv)}`",
        f"- baseline fixed suite: `{_display_path(comparison.baseline_json)}`",
        f"- trained fixed suite: `{_display_path(comparison.trained_json)}`" if comparison.trained_json else "- trained fixed suite: skipped",
        "",
        f"## Rationale Mode",
        f"- `{rationale_mode}`",
        "",
        "## Baseline Metrics",
        f"- headline eval score: `{baseline_metrics.get('eval_score_pct', 0.0):.2f}%`",
        f"- orchestrator mean normalized reward: `{baseline_metrics.get('orch_norm_reward', 0.0):.4f}`",
        f"- floor-agent mean normalized reward: `{baseline_metrics.get('floor_norm_reward', 0.0):.4f}`",
        f"- save rate: `{baseline_metrics.get('save_rate', 0.0):.4f}`",
        f"- invalid action rate: `{baseline_metrics.get('invalid_action_rate', 0.0):.4f}`",
        f"- override win rate: `{baseline_metrics.get('override_win_rate', 0.0):.4f}`",
        "",
    ]

    if trained_payload is not None:
        lines.extend(
            [
                "## Trained Metrics",
                f"- headline eval score: `{trained_metrics.get('eval_score_pct', 0.0):.2f}%`",
                f"- orchestrator mean normalized reward: `{trained_metrics.get('orch_norm_reward', 0.0):.4f}`",
                f"- floor-agent mean normalized reward: `{trained_metrics.get('floor_norm_reward', 0.0):.4f}`",
                f"- save rate: `{trained_metrics.get('save_rate', 0.0):.4f}`",
                f"- invalid action rate: `{trained_metrics.get('invalid_action_rate', 0.0):.4f}`",
                f"- override win rate: `{trained_metrics.get('override_win_rate', 0.0):.4f}`",
                "",
                "## Behavior Diagnostics",
                f"- trained mean civilians saved: `{trained_diagnostics.get('civilians_saved', 0.0):.2f}`",
                f"- trained mean civilians lost: `{trained_diagnostics.get('civilians_lost', 0.0):.2f}`",
                f"- trained mean civilians remaining: `{trained_diagnostics.get('civilians_remaining', 0.0):.2f}`",
                f"- trained wait/scout/route/evacuate rates: `{trained_diagnostics.get('wait_rate', 0.0):.4f}` / `{trained_diagnostics.get('scout_rate', 0.0):.4f}` / `{trained_diagnostics.get('route_rate', 0.0):.4f}` / `{trained_diagnostics.get('evacuate_rate', 0.0):.4f}`",
                f"- baseline mean civilians saved/remaining: `{baseline_diagnostics.get('civilians_saved', 0.0):.2f}` / `{baseline_diagnostics.get('civilians_remaining', 0.0):.2f}`",
                "",
                "## Deltas (Trained - Baseline)",
                f"- headline eval score: `{_format_delta(baseline_metrics, trained_metrics, 'eval_score_pct')}` percentage points",
                f"- orchestrator mean normalized reward: `{_format_delta(baseline_metrics, trained_metrics, 'orch_norm_reward')}`",
                f"- floor-agent mean normalized reward: `{_format_delta(baseline_metrics, trained_metrics, 'floor_norm_reward')}`",
                f"- save rate: `{_format_delta(baseline_metrics, trained_metrics, 'save_rate')}`",
                f"- invalid action rate: `{_format_delta(baseline_metrics, trained_metrics, 'invalid_action_rate')}`",
                f"- override win rate: `{_format_delta(baseline_metrics, trained_metrics, 'override_win_rate')}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Trained Eval Warning",
                "- trained fixed suite did not run; this bundle is a baseline smoke artifact, not proof of learning",
                "",
            ]
        )

    lines.extend(
        [
            "## Suggested Demo Flow",
            "- show one baseline fixed-suite summary",
            "- show the comparison CSV and generated plots",
            "- show one live `/openenv/reset -> /openenv/step -> /openenv/state` interaction",
            "- explain which safeguards prevent reward hacking",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_scorecard_artifacts(
    *,
    output_dir: Path,
    comparison: ComparisonResult,
    baseline_payload: dict | None,
    trained_payload: dict | None,
    rationale_mode: str,
) -> tuple[Path, Path]:
    baseline_metrics = _aggregate_metrics(baseline_payload)
    trained_metrics = _aggregate_metrics(trained_payload)
    baseline_diagnostics = _behavior_diagnostics(baseline_payload)
    trained_diagnostics = _behavior_diagnostics(trained_payload)
    metric_keys = [
        "eval_score_pct",
        "orch_norm_reward",
        "floor_norm_reward",
        "save_rate",
        "invalid_action_rate",
        "override_win_rate",
    ]

    metric_rows = []
    for key in metric_keys:
        metric_rows.append(
            {
                "key": key,
                "label": _metric_label(key),
                "goal": _metric_direction(key),
                "baseline": baseline_metrics.get(key),
                "trained": trained_metrics.get(key) if trained_payload is not None else None,
                "delta": (
                    trained_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
                    if trained_payload is not None and key in trained_metrics and key in baseline_metrics
                    else None
                ),
                "status": _metric_status(baseline_metrics, trained_metrics, key),
            }
        )

    improved_count = sum(1 for row in metric_rows if row["status"] == "improved")
    regressed_count = sum(1 for row in metric_rows if row["status"] == "regressed")
    flat_count = sum(1 for row in metric_rows if row["status"] == "flat")
    no_data_count = sum(1 for row in metric_rows if row["status"] == "no_trained_data")

    scorecard_payload = {
        "project": "EvacOS2",
        "rationale_mode": rationale_mode,
        "artifacts": {
            "comparison_csv": _display_path(comparison.output_csv),
            "baseline_json": _display_path(comparison.baseline_json),
            "trained_json": _display_path(comparison.trained_json) if comparison.trained_json else None,
        },
        "headline": {
            "improved_metrics": improved_count,
            "regressed_metrics": regressed_count,
            "flat_metrics": flat_count,
            "no_trained_data_metrics": no_data_count,
        },
        "warnings": [
            "trained fixed-suite eval missing; do not claim trained improvement"
            if trained_payload is None
            else None,
            "headline eval score regressed versus baseline"
            if trained_payload is not None
            and trained_metrics.get("eval_score_pct", 0.0) < baseline_metrics.get("eval_score_pct", 0.0)
            else None,
        ],
        "behavior_diagnostics": {
            "baseline": baseline_diagnostics,
            "trained": trained_diagnostics if trained_payload is not None else None,
        },
        "metrics": metric_rows,
    }
    scorecard_payload["warnings"] = [
        warning for warning in scorecard_payload["warnings"] if warning is not None
    ]

    scorecard_json = output_dir / "submission_scorecard.json"
    scorecard_json.write_text(
        json.dumps(scorecard_payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# EvacOS2 Submission Scorecard",
        "",
        "## Judge-Fast Take",
        "- Environment: multi-agent evacuation simulator with a live OpenEnv-facing API",
        "- Evidence: fixed-suite baseline and trained evaluation artifacts in one bundle",
        "- Goal: show measurable improvement in coordinated evacuation behavior, not just plausible text output",
        "",
        "## Headline",
        f"- rationale mode: `{rationale_mode}`",
        f"- improved metrics: `{improved_count}`",
        f"- regressed metrics: `{regressed_count}`",
        f"- flat metrics: `{flat_count}`",
        f"- no-trained-data metrics: `{no_data_count}`",
        "",
        "## Warnings",
    ]
    warnings = scorecard_payload["warnings"]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Behavior Diagnostics",
            f"- baseline saved/lost/remaining: `{baseline_diagnostics.get('civilians_saved', 0.0):.2f}` / `{baseline_diagnostics.get('civilians_lost', 0.0):.2f}` / `{baseline_diagnostics.get('civilians_remaining', 0.0):.2f}`",
            f"- trained saved/lost/remaining: `{trained_diagnostics.get('civilians_saved', 0.0):.2f}` / `{trained_diagnostics.get('civilians_lost', 0.0):.2f}` / `{trained_diagnostics.get('civilians_remaining', 0.0):.2f}`" if trained_payload is not None else "- trained saved/lost/remaining: `n/a`",
            f"- trained wait/scout/route/evacuate: `{trained_diagnostics.get('wait_rate', 0.0):.4f}` / `{trained_diagnostics.get('scout_rate', 0.0):.4f}` / `{trained_diagnostics.get('route_rate', 0.0):.4f}` / `{trained_diagnostics.get('evacuate_rate', 0.0):.4f}`" if trained_payload is not None else "- trained wait/scout/route/evacuate: `n/a`",
            "",
        ]
    )

    lines.extend(
        [
        "## Metrics",
        "| metric | goal | baseline | trained | delta | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for row in metric_rows:
        baseline = "n/a" if row["baseline"] is None else f"{float(row['baseline']):.4f}"
        trained = "n/a" if row["trained"] is None else f"{float(row['trained']):.4f}"
        delta = "n/a" if row["delta"] is None else f"{float(row['delta']):+.4f}"
        lines.append(
            f"| {row['label']} | {row['goal']} | {baseline} | {trained} | {delta} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Bundle Artifacts",
            f"- comparison CSV: `{_display_path(comparison.output_csv)}`",
            f"- baseline fixed suite: `{_display_path(comparison.baseline_json)}`",
            f"- trained fixed suite: `{_display_path(comparison.trained_json)}`" if comparison.trained_json else "- trained fixed suite: skipped",
            "",
            "## Suggested Submission Flow",
            "- open this scorecard first",
            "- open `demo_bundle_summary.md` for the slightly longer explanation",
            "- show the CSV/plots only after the headline metrics are clear",
            "- finish with one live OpenEnv interaction",
            "",
        ]
    )

    scorecard_md = output_dir / "submission_scorecard.md"
    scorecard_md.write_text("\n".join(lines), encoding="utf-8")
    return scorecard_md, scorecard_json


def build_demo_bundle(
    trained_checkpoint: Path | None = None,
    *,
    tiers: Sequence[str] = ("easy",),
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
    disaster_families: Sequence[DisasterType | str] = tuple(DisasterType),
    max_rounds: int = 50,
    rationale_mode: str = "linear_capped",
    output_dir: Path = Path("outputs/demo_bundle"),
    skip_trained: bool = False,
    training_metrics_path: Path | None = None,
    trained_normalizer_snapshot: dict | None = None,
    config_path: Path = Path("training/config.yaml"),
    baseline_policy: str = "stub",
) -> DemoBundleResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = run_comparison(
        trained_checkpoint=trained_checkpoint,
        tiers=tiers,
        seeds=seeds,
        disaster_families=disaster_families,
        max_rounds=max_rounds,
        rationale_mode=rationale_mode,
        output_csv=output_dir / "baseline_vs_trained.csv",
        skip_trained=skip_trained,
        trained_normalizer_snapshot=trained_normalizer_snapshot,
        config_path=config_path,
        baseline_policy=baseline_policy,
    )
    plot_paths = [
        path
        for path in make_all_plots(output_dir, metrics_path=training_metrics_path)
        if path is not None
    ]
    baseline_payload = _load_json(comparison.baseline_json)
    trained_payload = _load_json(comparison.trained_json)
    summary_md = _write_summary_markdown(
        output_path=output_dir / "demo_bundle_summary.md",
        comparison=comparison,
        baseline_payload=baseline_payload,
        trained_payload=trained_payload,
        rationale_mode=rationale_mode,
    )
    scorecard_md, scorecard_json = _write_scorecard_artifacts(
        output_dir=output_dir,
        comparison=comparison,
        baseline_payload=baseline_payload,
        trained_payload=trained_payload,
        rationale_mode=rationale_mode,
    )
    return DemoBundleResult(
        output_dir=output_dir,
        summary_md=summary_md,
        scorecard_md=scorecard_md,
        scorecard_json=scorecard_json,
        comparison_csv=comparison.output_csv,
        baseline_json=comparison.baseline_json,
        trained_json=comparison.trained_json,
        plot_paths=plot_paths,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact EvacOS2 demo bundle.")
    parser.add_argument(
        "--trained-checkpoint",
        type=Path,
        default=None,
        help="Path to a trained LoRA checkpoint directory. Omit with --skip-trained for baseline-only bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo_bundle"),
        help="Where to write the bundle artifacts.",
    )
    parser.add_argument(
        "--rationale-mode",
        default="linear_capped",
        help="Rationale reward mode to use for the comparison run.",
    )
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip trained comparison and build a baseline-only bundle.",
    )
    parser.add_argument(
        "--training-metrics-path",
        type=Path,
        default=None,
        help="Optional path to the training metrics CSV for reward-curve plotting.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/config.yaml"),
        help="Training config to use if checkpoint metadata is unavailable.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Bounded rounds per eval episode. Keep small for smoke/gate evals.",
    )
    parser.add_argument(
        "--baseline-policy",
        choices=("stub", "base_model"),
        default="stub",
        help=(
            "Baseline reference. Use base_model for judge-facing no-LoRA "
            "model-vs-trained-LoRA comparisons."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_demo_bundle(
        trained_checkpoint=args.trained_checkpoint,
        output_dir=args.output_dir,
        rationale_mode=args.rationale_mode,
        max_rounds=args.max_rounds,
        skip_trained=args.skip_trained,
        training_metrics_path=args.training_metrics_path,
        config_path=args.config,
        baseline_policy=args.baseline_policy,
    )
    print(result.scorecard_md)
    print(result.summary_md)


if __name__ == "__main__":
    main()
