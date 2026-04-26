from __future__ import annotations

from pathlib import Path

import pytest

from training.config_schema import TrainingConfig
from training.train import _load_yaml_config


@pytest.mark.parametrize(
    ("family", "config_path"),
    [
        ("fire", Path("training/config.remote-unsloth-7b3b-fire-specialist.yaml")),
        ("flood", Path("training/config.remote-unsloth-7b3b-flood-specialist.yaml")),
        ("gas", Path("training/config.remote-unsloth-7b3b-gas-specialist.yaml")),
    ],
)
def test_specialist_configs_are_valid_split_role_unsloth_configs(family: str, config_path: Path):
    raw = _load_yaml_config(config_path)
    config = TrainingConfig(**raw)

    assert config.backend == "unsloth"
    assert config.model.uses_split_bases is True
    assert config.model.orchestrator_base == "Qwen/Qwen2.5-7B-Instruct"
    assert config.model.floor_base == "Qwen/Qwen2.5-3B-Instruct"
    assert config.rollout.use_vllm is False
    assert config.rollout.disaster_families == [family]
    assert family in config.checkpoint.root_dir
    assert family in config.metrics.csv_path
    assert family in config.metrics.jsonl_dir


def test_specialist_configs_use_distinct_output_paths():
    paths = []
    for family in ("fire", "flood", "gas"):
        raw = _load_yaml_config(
            Path(f"training/config.remote-unsloth-7b3b-{family}-specialist.yaml")
        )
        config = TrainingConfig(**raw)
        paths.extend(
            [
                config.checkpoint.root_dir,
                config.metrics.csv_path,
                config.metrics.jsonl_dir,
            ]
        )

    assert len(paths) == len(set(paths))


@pytest.mark.parametrize(
    ("family", "config_path"),
    [
        ("fire", Path("training/config.remote-unsloth-3b-fire-floor-specialist.yaml")),
        ("flood", Path("training/config.remote-unsloth-3b-flood-floor-specialist.yaml")),
        ("gas", Path("training/config.remote-unsloth-3b-gas-floor-specialist.yaml")),
    ],
)
def test_floor_specialist_configs_use_stub_orchestrator_and_only_3b_floor_policy(
    family: str,
    config_path: Path,
):
    raw = _load_yaml_config(config_path)
    config = TrainingConfig(**raw)

    assert config.backend == "unsloth"
    assert config.roles.trainable == ["floor_agent"]
    assert config.roles.orchestrator_policy == "stub"
    assert config.uses_role_routed_policy is True
    assert config.model.uses_split_bases is False
    assert config.model.base == "Qwen/Qwen2.5-3B-Instruct"
    assert config.model.orchestrator_base is None
    assert config.model.floor_base is None
    assert config.model.resolved_base_for_role("floor_agent") == "Qwen/Qwen2.5-3B-Instruct"
    assert config.rollout.use_vllm is False
    assert config.rollout.disaster_families == [family]
    assert family in config.checkpoint.root_dir
    assert family in config.metrics.csv_path
    assert family in config.metrics.jsonl_dir


def test_floor_specialist_configs_use_distinct_output_paths():
    paths = []
    for family in ("fire", "flood", "gas"):
        raw = _load_yaml_config(
            Path(f"training/config.remote-unsloth-3b-{family}-floor-specialist.yaml")
        )
        config = TrainingConfig(**raw)
        paths.extend(
            [
                config.checkpoint.root_dir,
                config.metrics.csv_path,
                config.metrics.jsonl_dir,
            ]
        )

    assert len(paths) == len(set(paths))


def test_fire_floor_canary_config_is_easy_only_and_short():
    raw = _load_yaml_config(
        Path("training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml")
    )
    config = TrainingConfig(**raw)

    assert config.backend == "unsloth"
    assert config.max_steps == 50
    assert config.roles.trainable == ["floor_agent"]
    assert config.roles.orchestrator_policy == "stub"
    assert config.rollout.disaster_families == ["fire"]
    assert config.rollout.max_rounds_per_episode == 4
    assert config.rollout.expanded_tier_schedule() == ["easy"] * 50
    assert config.eval.every_steps == 10
    assert config.eval.tiers == ["easy"]
    assert config.checkpoint.root_dir.endswith("-canary-50")
    assert config.metrics.csv_path.endswith("-canary-50-metrics.csv")


def test_fire_floor_easy_proof_config_is_easy_only_and_eval_heavy():
    raw = _load_yaml_config(
        Path("training/config.remote-unsloth-3b-fire-floor-specialist-easy-proof-300.yaml")
    )
    config = TrainingConfig(**raw)

    assert config.backend == "unsloth"
    assert config.max_steps == 300
    assert config.roles.trainable == ["floor_agent"]
    assert config.roles.orchestrator_policy == "stub"
    assert config.rollout.disaster_families == ["fire"]
    assert config.rollout.max_rounds_per_episode == 4
    assert config.rollout.expanded_tier_schedule() == ["easy"] * 300
    assert config.eval.every_steps == 25
    assert config.eval.tiers == ["easy"]
    assert config.checkpoint.every_steps == 25
    assert config.checkpoint.root_dir.endswith("-easy-proof-300")
    assert config.metrics.csv_path.endswith("-easy-proof-300-metrics.csv")


@pytest.mark.parametrize(
    ("family", "expected_rounds"),
    [
        ("fire", 4),
        ("flood", 5),
        ("gas", 10),
    ],
)
def test_floor_throughput_smoke_configs_are_easy_only_and_runtime_tuned(
    family: str,
    expected_rounds: int,
):
    raw = _load_yaml_config(
        Path(
            "training/"
            f"config.remote-unsloth-3b-{family}-floor-specialist-throughput-smoke-100.yaml"
        )
    )
    config = TrainingConfig(**raw)

    assert config.backend == "unsloth"
    assert config.max_steps == 100
    assert config.roles.trainable == ["floor_agent"]
    assert config.roles.orchestrator_policy == "stub"
    assert config.rollout.disaster_families == [family]
    assert config.rollout.max_rounds_per_episode == expected_rounds
    assert config.rollout.expanded_tier_schedule() == ["easy"] * 100
    assert config.rollout.candidates_per_floor_prompt == 4
    assert config.rollout.include_oracle_floor_candidate is True
    assert config.eval.every_steps == 25
    assert config.eval.tiers == ["easy"]
    assert config.checkpoint.every_steps == 25
    assert config.checkpoint.keep_last_n == 4
    assert config.checkpoint.root_dir.endswith("-throughput-smoke-100")
    assert config.metrics.csv_path.endswith("-throughput-smoke-100-metrics.csv")
