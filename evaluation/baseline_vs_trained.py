"""Baseline versus trained comparison helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel

from curriculum.controller import EVAL_SEEDS
from evacos_ma.models import DisasterType
from evaluation._training_contract import RoleRoutedPolicy, StubPolicy, hf_policy_factory

from .fixed_suite import (
    DEFAULT_DISASTER_FAMILIES,
    FixedSuiteResult,
    SCHEMA_VERSION,
    run_fixed_suite,
)


class ComparisonResult(BaseModel):
    baseline_json: Path
    trained_json: Path | None = None
    output_csv: Path
    schema_version: str = SCHEMA_VERSION


def _load_model_config(config_path: Path = Path("training/config.yaml")) -> dict[str, str | bool]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyYAML is required by evaluation.baseline_vs_trained._load_model_config()"
        ) from exc
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = data.get("model", {}) or {}
    base = str(model_cfg.get("base", "Qwen/Qwen2.5-3B-Instruct"))
    orchestrator_base = model_cfg.get("orchestrator_base")
    floor_base = model_cfg.get("floor_base")
    resolved_orchestrator = str(orchestrator_base or base)
    resolved_floor = str(floor_base or base)
    return {
        "base": base,
        "orchestrator": resolved_orchestrator,
        "floor_agent": resolved_floor,
        "split": resolved_orchestrator != resolved_floor,
    }


def _load_model_name(config_path: Path = Path("training/config.yaml")) -> str:
    config = _load_model_config(config_path)
    return str(config["orchestrator"])


def _nan() -> float:
    return float("nan")


def _metric_rows(suite: FixedSuiteResult) -> dict[tuple[str, int, str, str, str], float]:
    rows: dict[tuple[str, int, str, str, str], float] = {}
    for episode in suite.episodes:
        for role in ("orchestrator", "floor_agent"):
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "raw_reward")] = episode.raw_reward_by_role.get(role, 0.0)
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "normalized_reward")] = episode.normalized_reward_by_role.get(role, 0.0)
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "save_rate")] = episode.save_rate
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "invalid_action_rate")] = episode.invalid_action_rate
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "override_win_rate")] = episode.override_win_rate
    return rows


def _trained_factory(checkpoint: Path) -> Callable[[], object]:
    model_cfg = _load_model_config()
    if not bool(model_cfg["split"]):
        model_name = str(model_cfg["orchestrator"])
        return lambda: hf_policy_factory(model_name, lora_adapter_path=str(checkpoint))

    orch_checkpoint = checkpoint / "orchestrator"
    floor_checkpoint = checkpoint / "floor_agent"
    if not orch_checkpoint.exists() or not floor_checkpoint.exists():
        raise FileNotFoundError(
            "Split-role trained comparison expects checkpoint/orchestrator and "
            "checkpoint/floor_agent adapter directories."
        )

    def _factory() -> object:
        return RoleRoutedPolicy(
            orchestrator_policy=hf_policy_factory(
                str(model_cfg["orchestrator"]),
                lora_adapter_path=str(orch_checkpoint),
            ),
            floor_policy=hf_policy_factory(
                str(model_cfg["floor_agent"]),
                lora_adapter_path=str(floor_checkpoint),
            ),
        )

    return _factory


def run_comparison(
    trained_checkpoint: Path | None = None,
    *,
    tiers: Sequence[str] = ("easy", "medium"),
    seeds: Sequence[int] = EVAL_SEEDS,
    disaster_families: Sequence[DisasterType | str] = DEFAULT_DISASTER_FAMILIES,
    rationale_mode: str = "linear_capped",
    output_csv: Path = Path("outputs/evals/baseline_vs_trained.csv"),
    skip_trained: bool = False,
    trained_normalizer_snapshot: dict | None = None,
) -> ComparisonResult:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    baseline_suite = run_fixed_suite(
        lambda: StubPolicy(seed=0),
        tiers=tiers,
        seeds=seeds,
        disaster_families=disaster_families,
        rationale_mode=rationale_mode,
        label="baseline",
        output_dir=output_csv.parent,
        normalizer_snapshot=None,
    )

    trained_suite: FixedSuiteResult | None = None
    trained_json: Path | None = None
    if trained_checkpoint is None or not trained_checkpoint.exists():
        if not skip_trained:
            raise FileNotFoundError("trained_checkpoint does not exist and skip_trained=False")
    else:
        trained_suite = run_fixed_suite(
            _trained_factory(trained_checkpoint),
            tiers=tiers,
            seeds=seeds,
            disaster_families=disaster_families,
            rationale_mode=rationale_mode,
            label="trained",
            output_dir=output_csv.parent,
            normalizer_snapshot=trained_normalizer_snapshot,
        )
        trained_json = output_csv.parent / f"fixed_suite_trained_{rationale_mode}.json"

    baseline_rows = _metric_rows(baseline_suite)
    trained_rows = _metric_rows(trained_suite) if trained_suite is not None else {}
    fieldnames = [
        "tier",
        "seed",
        "disaster_family",
        "role",
        "metric",
        "baseline",
        "trained",
        "delta",
        "rationale_mode",
        "schema_version",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(baseline_rows):
            baseline_value = float(baseline_rows[key])
            trained_value = float(trained_rows.get(key, _nan()))
            delta = float(trained_value - baseline_value) if not math.isnan(trained_value) else _nan()
            tier, seed, family, role, metric = key
            writer.writerow(
                {
                    "tier": tier,
                    "seed": seed,
                    "disaster_family": family,
                    "role": role,
                    "metric": metric,
                    "baseline": baseline_value,
                    "trained": trained_value,
                    "delta": delta,
                    "rationale_mode": rationale_mode,
                    "schema_version": SCHEMA_VERSION,
                }
            )

    return ComparisonResult(
        baseline_json=output_csv.parent / f"fixed_suite_baseline_{rationale_mode}.json",
        trained_json=trained_json,
        output_csv=output_csv,
    )
