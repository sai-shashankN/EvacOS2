"""Metrics writer — CSV row appender + JSONL trace writer.

Heavy-dependency-free.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV training metrics
# ---------------------------------------------------------------------------

_TRAINER_DIAGNOSTIC_COLUMNS: list[str] = [
    "loss",
    "policy_loss",
    "kl_loss",
    "ratio_mean",
    "ratio_std",
    "clip_fraction",
    "kl_max",
    "mask_coverage",
    "mean_advantage",
    "advantage_std",
    "loss_mean_across_epochs",
    "policy_loss_mean_across_epochs",
    "kl_loss_mean_across_epochs",
    "ratio_mean_across_epochs",
    "ratio_std_mean_across_epochs",
    "clip_fraction_mean_across_epochs",
    "kl_max_across_epochs",
    "num_inner_epochs",
]

_ROLE_DIAGNOSTIC_COLUMNS: list[str] = []
for _role in ("orchestrator", "floor_agent"):
    _ROLE_DIAGNOSTIC_COLUMNS.append(f"{_role}_sample_groups")
    _ROLE_DIAGNOSTIC_COLUMNS.extend(
        f"{_role}_{column}" for column in _TRAINER_DIAGNOSTIC_COLUMNS
    )


# Canonical column order for metrics.csv
_METRICS_COLUMNS: list[str] = [
    "step",
    "wall_seconds",
    "tier_mix",
    "mean_raw_reward_orch",
    "mean_raw_reward_floor",
    "mean_norm_reward_orch",
    "mean_norm_reward_floor",
    "invalid_action_rate",
    "override_rate",
    "override_win_rate",
    "rationale_bonus_mean",
    "episodes_seen",
    # Trainer diagnostics (merged from MultiAgentGRPOTrainer.step return)
    *_TRAINER_DIAGNOSTIC_COLUMNS,
    # Split-role trainer diagnostics (kept alongside aggregate fields).
    *_ROLE_DIAGNOSTIC_COLUMNS,
]


def append_training_metrics_row(csv_path: Path, row: dict) -> None:
    """Append one row to the training metrics CSV.

    Creates the file with a header on first call.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# JSONL trace writer
# ---------------------------------------------------------------------------


def write_trace_row(jsonl_path: Path, row: dict) -> None:
    """Append one JSON-line to a trace file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
