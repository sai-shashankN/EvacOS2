"""Rationale scaling sweep over the fixed eval suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from curriculum.controller import EVAL_SEEDS
from evacos_ma.models import DisasterType
from evaluation._training_contract import Policy

from .fixed_suite import DEFAULT_DISASTER_FAMILIES, FixedSuiteResult, SCHEMA_VERSION, run_fixed_suite

RATIONALE_MODES = ("off", "linear_capped", "log_uncapped")


class RationaleModeSummary(BaseModel):
    rationale_mode: str
    mean_normalized_reward: dict[str, float] = Field(default_factory=dict)
    median_normalized_reward: dict[str, float] = Field(default_factory=dict)
    invalid_action_rate: float = 0.0


class RationaleSweepResult(BaseModel):
    modes: list[RationaleModeSummary]
    suites: list[FixedSuiteResult]
    schema_version: str = SCHEMA_VERSION


def run_rationale_sweep(
    policy_factory: Callable[[], Policy],
    *,
    tiers: Sequence[str] = ("easy",),
    seeds: Sequence[int] = EVAL_SEEDS,
    disaster_families: Sequence[DisasterType | str] = DEFAULT_DISASTER_FAMILIES,
    output_path: Path = Path("outputs/evals/rationale_sweep.json"),
) -> RationaleSweepResult:
    suites: list[FixedSuiteResult] = []
    summaries: list[RationaleModeSummary] = []
    for mode in RATIONALE_MODES:
        suite = run_fixed_suite(
            policy_factory,
            tiers=tiers,
            seeds=seeds,
            disaster_families=disaster_families,
            rationale_mode=mode,
            label=f"sweep_{mode}",
            output_dir=output_path.parent,
        )
        suites.append(suite)
        summaries.append(
            RationaleModeSummary(
                rationale_mode=mode,
                mean_normalized_reward={
                    role: stats.mean for role, stats in suite.aggregate.normalized_reward.items()
                },
                median_normalized_reward={
                    role: stats.median for role, stats in suite.aggregate.normalized_reward.items()
                },
                invalid_action_rate=suite.aggregate.invalid_action_rate.mean,
            )
        )
    result = RationaleSweepResult(modes=summaries, suites=suites)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
