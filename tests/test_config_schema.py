from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from training.config_schema import ModelConfig, RewardConfig, RolloutConfig, TrainingConfig
from training.train import _load_yaml_config


class TestModelDtypeValidation:
    def test_model_config_rejects_unknown_dtype(self):
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(dtype="foo")
        message = str(excinfo.value)
        assert "model.dtype" in message
        assert "bfloat16" in message
        assert "float16" in message
        assert "float32" in message

    def test_model_config_accepts_bfloat16_float16_float32(self):
        for dtype in ("bfloat16", "float16", "float32"):
            assert ModelConfig(dtype=dtype).dtype == dtype


class TestVllmBackendGate:
    def test_vllm_requires_unsloth_backend_rejects_hf(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(backend="hf", rollout=RolloutConfig(use_vllm=True))
        message = str(excinfo.value)
        assert "rollout.use_vllm" in message
        assert "backend" in message
        assert "unsloth" in message

    def test_vllm_allowed_on_unsloth_backend(self):
        config = TrainingConfig(backend="unsloth", rollout=RolloutConfig(use_vllm=True))
        assert config.backend == "unsloth"
        assert config.rollout.use_vllm is True

    def test_vllm_false_allowed_on_hf_backend(self):
        config = TrainingConfig(backend="hf", rollout=RolloutConfig(use_vllm=False))
        assert config.backend == "hf"
        assert config.rollout.use_vllm is False


class TestConfigContractRestoration:
    def test_training_config_rejects_legacy_grpo_group_size(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(grpo={"group_size": 4})
        assert "group_size" in str(excinfo.value)

    def test_training_config_loads_repository_yaml(self):
        raw = _load_yaml_config(Path("training/config.yaml"))
        config = TrainingConfig(**raw)
        assert config.reward.rationale_scaling == "linear_capped"

    @pytest.mark.parametrize("value", ["off", "linear_capped", "log_uncapped"])
    def test_reward_config_accepts_known_rationale_scaling_values(self, value):
        assert RewardConfig(rationale_scaling=value).rationale_scaling == value

    def test_reward_config_rejects_unknown_rationale_scaling_value(self):
        with pytest.raises(ValidationError) as excinfo:
            RewardConfig(rationale_scaling="metadata_only")
        assert "rationale_scaling" in str(excinfo.value)
