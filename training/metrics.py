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
    "group_raw_reward_std_mean",
    "group_raw_reward_std_min",
    "group_raw_reward_std_max",
    "singleton_group_rate",
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
    "max_steps",
    "wall_seconds",
    "run_name",
    "config_hash",
    "tier_mix",
    "disaster_families",
    "episodes_per_step",
    "max_rounds_per_episode",
    "candidates_per_floor_prompt",
    "include_oracle_floor_candidate",
    "sampling_temperature",
    "mean_raw_reward_orch",
    "mean_raw_reward_floor",
    "raw_reward_std_orch",
    "raw_reward_std_floor",
    "mean_norm_reward_orch",
    "mean_norm_reward_floor",
    "norm_reward_std_orch",
    "norm_reward_std_floor",
    "invalid_action_rate",
    "wait_rate",
    "floor_agent_wait_rate",
    "orchestrator_wait_rate",
    "empty_args_rate",
    "floor_agent_active_action_rate",
    "active_empty_args_rate",
    "valid_but_hollow_action_rate",
    "floor_scout_action_rate",
    "floor_route_action_rate",
    "floor_evacuate_action_rate",
    "floor_route_exit_rate",
    "floor_route_stairwell_rate",
    "floor_route_room_rate",
    "floor_route_legacy_egress_alias_rate",
    "floor_route_missing_target_rate",
    "override_rate",
    "override_win_rate",
    "rationale_bonus_mean",
    "episodes_seen",
    "watchdog_status",
    "watchdog_reason",
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

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with open(csv_path, newline="") as existing:
            reader = csv.reader(existing)
            existing_header = next(reader, None)
        if existing_header != _METRICS_COLUMNS:
            raise RuntimeError(
                "Training metrics CSV header does not match the current schema. "
                f"Refusing to append to {csv_path}; start a new metrics path or migrate the file."
            )

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
