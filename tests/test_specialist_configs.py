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
