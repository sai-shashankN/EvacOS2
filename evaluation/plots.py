"""Headless Phase 8 plots."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    matplotlib = None
    plt = None

import numpy as np

LOGGER = logging.getLogger(__name__)


def _warn(message: str) -> None:
    LOGGER.warning(message)


def _ready() -> bool:
    if plt is None:
        _warn("Skipping plot generation because matplotlib is not installed")
        return False
    return True


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        _warn(f"Skipping missing artifact: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        _warn(f"Skipping missing artifact: {path}")
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_training_metrics_path(base_dir: Path, metrics_path: Path | None = None) -> Path:
    if metrics_path is not None:
        return metrics_path

    default_candidate = Path("outputs/training/metrics.csv")
    base_dir_candidate = base_dir.parent / "training" / "metrics.csv"
    if base_dir_candidate.exists():
        return base_dir_candidate
    return default_candidate


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def make_reward_curve(base_dir: Path, *, metrics_path: Path | None = None) -> Path | None:
    if not _ready():
        return None
    rows = _load_csv(_resolve_training_metrics_path(base_dir, metrics_path))
    if not rows:
        return None
    required_columns = {"step", "mean_norm_reward_orch", "mean_norm_reward_floor"}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        _warn(
            "Skipping reward_curve: metrics CSV missing columns "
            + ", ".join(sorted(missing_columns))
        )
        return None
    output = base_dir / "plots" / "reward_curve.png"
    steps = [int(row["step"]) for row in rows]
    orch = [float(row["mean_norm_reward_orch"]) for row in rows]
    floor = [float(row["mean_norm_reward_floor"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, orch, label="orchestrator", linewidth=2)
    ax.plot(steps, floor, label="floor_agent", linewidth=2)
    ax.set_title("Mean Normalized Reward vs Training Step")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean normalized reward")
    ax.legend()
    ax.grid(alpha=0.3)
    _ensure_dir(output)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def make_baseline_vs_trained_bar(base_dir: Path) -> Path | None:
    if not _ready():
        return None
    rows = _load_csv(base_dir / "baseline_vs_trained.csv")
    if not rows:
        return None
    metric_rows = [row for row in rows if row["metric"] == "normalized_reward"]
    if not metric_rows:
        _warn("Skipping baseline_vs_trained_bar: no normalized_reward rows")
        return None
    roles = sorted({row["role"] for row in metric_rows})
    baseline = [
        _safe_mean([float(row["baseline"]) for row in metric_rows if row["role"] == role])
        for role in roles
    ]
    trained = [
        _safe_nanmean([float(row["trained"]) for row in metric_rows if row["role"] == role])
        for role in roles
    ]
    x = np.arange(len(roles))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.18, baseline, width=0.36, label="baseline")
    ax.bar(x + 0.18, trained, width=0.36, label="trained")
    ax.set_xticks(x)
    ax.set_xticklabels(roles)
    ax.set_ylabel("Mean normalized reward")
    ax.set_title("Baseline vs Trained")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    output = base_dir / "plots" / "baseline_vs_trained_bar.png"
    _ensure_dir(output)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _line_metric_plot(base_dir: Path, metric: str, output_name: str, title: str) -> Path | None:
    if not _ready():
        return None
    rows = _load_csv(base_dir / "baseline_vs_trained.csv")
    if not rows:
        return None
    metric_rows = [row for row in rows if row["metric"] == metric]
    if not metric_rows:
        _warn(f"Skipping {output_name}: no {metric} rows")
        return None
    tiers = sorted({row["tier"] for row in metric_rows})
    baseline = [
        _safe_mean([float(row["baseline"]) for row in metric_rows if row["tier"] == tier])
        for tier in tiers
    ]
    trained = [
        _safe_nanmean([float(row["trained"]) for row in metric_rows if row["tier"] == tier])
        for tier in tiers
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tiers, baseline, marker="o", label="baseline")
    ax.plot(tiers, trained, marker="o", label="trained")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    output = base_dir / "plots" / output_name
    _ensure_dir(output)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def make_invalid_action_rate_plot(base_dir: Path) -> Path | None:
    return _line_metric_plot(base_dir, "invalid_action_rate", "invalid_action_rate.png", "Invalid Action Rate by Tier")


def make_override_win_rate_plot(base_dir: Path) -> Path | None:
    return _line_metric_plot(base_dir, "override_win_rate", "override_win_rate.png", "Override Win Rate by Tier")


def make_rationale_mode_comparison(base_dir: Path) -> Path | None:
    if not _ready():
        return None
    payload = _load_json(base_dir / "rationale_sweep.json")
    if payload is None:
        return None
    modes = payload.get("modes", [])
    if not modes:
        _warn("Skipping rationale mode comparison: no modes")
        return None
    labels = [entry["rationale_mode"] for entry in modes]
    orch = [float(entry.get("mean_normalized_reward", {}).get("orchestrator", 0.0)) for entry in modes]
    floor = [float(entry.get("mean_normalized_reward", {}).get("floor_agent", 0.0)) for entry in modes]
    invalid = [float(entry.get("invalid_action_rate", 0.0)) for entry in modes]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.25, orch, width=0.25, label="orch reward")
    ax.bar(x, floor, width=0.25, label="floor reward")
    ax.bar(x + 0.25, invalid, width=0.25, label="invalid rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Rationale Mode Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    output = base_dir / "plots" / "rationale_mode_comparison.png"
    _ensure_dir(output)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def make_rollout_wall_clock_histogram(base_dir: Path) -> Path | None:
    if not _ready():
        return None
    fixed_suite_files = sorted(base_dir.glob("fixed_suite_*.json"))
    if not fixed_suite_files:
        _warn("Skipping rollout wall clock plot: no fixed_suite JSON found")
        return None
    wall_clock: list[float] = []
    for path in fixed_suite_files:
        payload = _load_json(path)
        if payload is None:
            continue
        wall_clock.extend(float(episode.get("wall_clock_s", 0.0)) for episode in payload.get("episodes", []))
    if not wall_clock:
        _warn("Skipping rollout wall clock plot: no episode timings found")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(wall_clock, bins=min(10, max(3, len(wall_clock))))
    ax.set_title("Rollout Wall Clock per Episode")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Episodes")
    output = base_dir / "plots" / "rollout_wall_clock.png"
    _ensure_dir(output)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def make_all_plots(base_dir: Path, *, metrics_path: Path | None = None) -> list[Path | None]:
    return [
        make_reward_curve(base_dir, metrics_path=metrics_path),
        make_baseline_vs_trained_bar(base_dir),
        make_invalid_action_rate_plot(base_dir),
        make_override_win_rate_plot(base_dir),
        make_rationale_mode_comparison(base_dir),
        make_rollout_wall_clock_histogram(base_dir),
    ]
