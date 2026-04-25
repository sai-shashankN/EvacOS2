"""Baseline versus trained comparison helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel

from curriculum.controller import EVAL_SEEDS
from evacos_ma.models import DisasterType
from evaluation._training_contract import (
    RoleRoutedPolicy,
    ScopeRoutedFloorPolicy,
    StubPolicy,
    hf_policy_factory,
)

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
    roles_cfg = data.get("roles", {}) or {}
    trainable_roles = roles_cfg.get("trainable", ["orchestrator", "floor_agent"])
    trainable_set = set(trainable_roles if isinstance(trainable_roles, list) else [])
    frozen_adapter_paths = roles_cfg.get("frozen_adapter_paths", {}) or {}
    frozen_floor_specialist_paths = (
        roles_cfg.get("frozen_floor_specialist_adapter_paths", {}) or {}
    )
    base = str(model_cfg.get("base", "Qwen/Qwen2.5-3B-Instruct"))
    orchestrator_base = model_cfg.get("orchestrator_base")
    floor_base = model_cfg.get("floor_base")
    resolved_orchestrator = str(orchestrator_base or base)
    resolved_floor = str(floor_base or base)
    orchestrator_policy = str(roles_cfg.get("orchestrator_policy", "model"))
    split = resolved_orchestrator != resolved_floor
    return {
        "base": base,
        "orchestrator": resolved_orchestrator,
        "floor_agent": resolved_floor,
        "split": split,
        "role_routed": (
            split
            or orchestrator_policy != "model"
            or bool(frozen_adapter_paths)
            or bool(frozen_floor_specialist_paths)
            or trainable_set != {"orchestrator", "floor_agent"}
        ),
        "orchestrator_policy": orchestrator_policy,
        "has_floor_specialists": bool(frozen_floor_specialist_paths),
    }


def _checkpoint_meta_path(checkpoint: Path) -> Path | None:
    """Return the nearest checkpoint ``meta.json`` for an adapter/latest path."""
    candidates = [
        checkpoint / "meta.json",
        checkpoint.parent / "meta.json",
        checkpoint.parent.parent / "meta.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _adapter_root(checkpoint: Path) -> Path:
    """Accept either a checkpoint dir or its lora_adapter dir."""
    adapter_dir = checkpoint / "lora_adapter"
    if adapter_dir.exists():
        return adapter_dir
    return checkpoint


def _checkpoint_floor_specialist_paths(checkpoint: Path) -> dict[str, Path]:
    adapter_root = _adapter_root(checkpoint)
    specialist_root = adapter_root / "floor_agent" / "specialists"
    return {
        family: specialist_root / family
        for family in ("fire", "flood", "gas")
        if (specialist_root / family).exists()
    }


def _checkpoint_has_floor_generalist(checkpoint: Path) -> bool:
    adapter_root = _adapter_root(checkpoint)
    floor_checkpoint = adapter_root / "floor_agent"
    return (floor_checkpoint / "adapter_config.json").exists()


def _effective_disaster_families_for_checkpoint(
    checkpoint: Path | None,
    requested_families: Sequence[DisasterType | str],
) -> Sequence[DisasterType | str]:
    """Avoid default eval families that a routed-specialist checkpoint cannot serve."""

    if checkpoint is None or not checkpoint.exists():
        return requested_families
    if _checkpoint_has_floor_generalist(checkpoint):
        return requested_families

    specialist_paths = _checkpoint_floor_specialist_paths(checkpoint)
    if not specialist_paths:
        return requested_families

    supported = set(specialist_paths)
    filtered: list[DisasterType | str] = []
    for family in requested_families:
        family_value = family.value if isinstance(family, DisasterType) else str(family)
        if family_value in supported:
            filtered.append(family)

    if not filtered:
        raise ValueError(
            "Routed-specialist checkpoint has no generalist floor adapter and "
            f"supports only {sorted(supported)!r}; requested disaster_families "
            f"{[str(family) for family in requested_families]!r} have no overlap."
        )
    return tuple(filtered)


def _load_model_config_from_checkpoint(checkpoint: Path) -> dict[str, str | bool] | None:
    meta_path = _checkpoint_meta_path(checkpoint)
    if meta_path is None:
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    orchestrator_policy = str(meta.get("orchestrator_policy") or "model")
    role_model_names = meta.get("role_model_names")
    role_lora_weights_paths = meta.get("role_lora_weights_paths")
    floor_specialist_paths = meta.get("floor_specialist_lora_weights_paths")
    has_role_adapter_paths = isinstance(role_lora_weights_paths, dict) and bool(
        role_lora_weights_paths
    )
    has_floor_specialist_paths = isinstance(floor_specialist_paths, dict) and bool(
        floor_specialist_paths
    )
    if isinstance(role_model_names, dict):
        fallback_model_name = str(meta.get("model_name") or "Qwen/Qwen2.5-3B-Instruct")
        orchestrator = str(role_model_names.get("orchestrator") or fallback_model_name)
        floor = str(role_model_names.get("floor_agent") or fallback_model_name)
        if orchestrator and floor:
            split = orchestrator != floor
            return {
                "base": orchestrator,
                "orchestrator": orchestrator,
                "floor_agent": floor,
                "split": split,
                "role_routed": (
                    split
                    or orchestrator_policy != "model"
                    or has_role_adapter_paths
                    or has_floor_specialist_paths
                ),
                "orchestrator_policy": orchestrator_policy,
                "has_floor_specialists": has_floor_specialist_paths,
            }
    model_name = str(meta.get("model_name") or "Qwen/Qwen2.5-3B-Instruct")
    return {
        "base": model_name,
        "orchestrator": model_name,
        "floor_agent": model_name,
        "split": False,
        "role_routed": (
            orchestrator_policy != "model"
            or has_role_adapter_paths
            or has_floor_specialist_paths
        ),
        "orchestrator_policy": orchestrator_policy,
        "has_floor_specialists": has_floor_specialist_paths,
    }


def _load_model_name(config_path: Path = Path("training/config.yaml")) -> str:
    config = _load_model_config(config_path)
    return str(config["orchestrator"])


def _nan() -> float:
    return float("nan")


def _metric_rows(suite: FixedSuiteResult) -> dict[tuple[str, int, str, str, str], float]:
    rows: dict[tuple[str, int, str, str, str], float] = {}
    for episode in suite.episodes:
        rows[(episode.tier, episode.seed, episode.disaster_family, "team", "eval_score_pct")] = episode.eval_score_pct
        for role in ("orchestrator", "floor_agent"):
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "raw_reward")] = episode.raw_reward_by_role.get(role, 0.0)
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "normalized_reward")] = episode.normalized_reward_by_role.get(role, 0.0)
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "save_rate")] = episode.save_rate
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "invalid_action_rate")] = episode.invalid_action_rate
            rows[(episode.tier, episode.seed, episode.disaster_family, role, "override_win_rate")] = episode.override_win_rate
    return rows


def _trained_factory(
    checkpoint: Path,
    *,
    config_path: Path = Path("training/config.yaml"),
) -> Callable[[], object]:
    adapter_root = _adapter_root(checkpoint)
    model_cfg = _load_model_config_from_checkpoint(checkpoint) or _load_model_config(config_path)
    if not bool(model_cfg.get("role_routed", model_cfg["split"])):
        model_name = str(model_cfg["orchestrator"])
        return lambda: hf_policy_factory(model_name, lora_adapter_path=str(adapter_root))

    floor_checkpoint = adapter_root / "floor_agent"
    orchestrator_policy = str(model_cfg.get("orchestrator_policy") or "model")
    floor_specialist_paths = _checkpoint_floor_specialist_paths(checkpoint)
    floor_generalist_checkpoint = (
        floor_checkpoint
        if (floor_checkpoint / "adapter_config.json").exists()
        else None
    )

    if floor_specialist_paths:
        orch_checkpoint = adapter_root / "orchestrator"
        if orchestrator_policy != "stub" and not orch_checkpoint.exists():
            raise FileNotFoundError(
                "Routed-specialist trained comparison expects checkpoint/orchestrator "
                "for the trained orchestrator adapter."
            )

        def _routed_specialist_factory() -> object:
            orchestrator = (
                StubPolicy(seed=0)
                if orchestrator_policy == "stub"
                else hf_policy_factory(
                    str(model_cfg["orchestrator"]),
                    lora_adapter_path=str(orch_checkpoint),
                )
            )
            generalist_policy = (
                hf_policy_factory(
                    str(model_cfg["floor_agent"]),
                    lora_adapter_path=str(floor_generalist_checkpoint),
                )
                if floor_generalist_checkpoint is not None
                else None
            )
            return RoleRoutedPolicy(
                orchestrator_policy=orchestrator,
                floor_policy=ScopeRoutedFloorPolicy(
                    specialist_policies={
                        family: hf_policy_factory(
                            str(model_cfg["floor_agent"]),
                            lora_adapter_path=str(path),
                        )
                        for family, path in floor_specialist_paths.items()
                    },
                    generalist_policy=generalist_policy,
                ),
            )

        return _routed_specialist_factory

    if orchestrator_policy == "stub":
        if not floor_checkpoint.exists():
            raise FileNotFoundError(
                "Floor-specialist trained comparison expects checkpoint/floor_agent "
                "adapter directory when orchestrator_policy='stub'."
            )

        def _floor_specialist_factory() -> object:
            return RoleRoutedPolicy(
                orchestrator_policy=StubPolicy(seed=0),
                floor_policy=hf_policy_factory(
                    str(model_cfg["floor_agent"]),
                    lora_adapter_path=str(floor_checkpoint),
                ),
            )

        return _floor_specialist_factory

    orch_checkpoint = adapter_root / "orchestrator"
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
    max_rounds: int = 50,
    rationale_mode: str = "linear_capped",
    output_csv: Path = Path("outputs/evals/baseline_vs_trained.csv"),
    skip_trained: bool = False,
    trained_normalizer_snapshot: dict | None = None,
    config_path: Path = Path("training/config.yaml"),
) -> ComparisonResult:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    effective_disaster_families = _effective_disaster_families_for_checkpoint(
        trained_checkpoint,
        disaster_families,
    )

    baseline_suite = run_fixed_suite(
        lambda: StubPolicy(seed=0),
        tiers=tiers,
        seeds=seeds,
        disaster_families=effective_disaster_families,
        max_rounds=max_rounds,
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
            _trained_factory(trained_checkpoint, config_path=config_path),
            tiers=tiers,
            seeds=seeds,
            disaster_families=effective_disaster_families,
            max_rounds=max_rounds,
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
