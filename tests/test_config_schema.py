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


class TestRoleAwareModelConfig:
    def test_model_config_defaults_to_shared_base_for_both_roles(self):
        config = ModelConfig(base="shared-model")

        assert config.resolved_base_for_role("orchestrator") == "shared-model"
        assert config.resolved_base_for_role("floor_agent") == "shared-model"
        assert config.uses_split_bases is False

    def test_model_config_supports_role_specific_overrides(self):
        config = ModelConfig(
            base="shared-model",
            orchestrator_base="big-orchestrator",
            floor_base="small-floor",
        )

        assert config.resolved_bases() == {
            "orchestrator": "big-orchestrator",
            "floor_agent": "small-floor",
        }
        assert config.uses_split_bases is True


class TestRoleSelectionConfig:
    def test_training_config_defaults_to_both_trainable_roles(self):
        config = TrainingConfig()

        assert config.roles.trainable == ["orchestrator", "floor_agent"]
        assert config.roles.orchestrator_policy == "model"
        assert config.uses_role_routed_policy is False

    def test_training_config_allows_floor_only_stub_orchestrator_mode(self):
        config = TrainingConfig(
            roles={
                "trainable": ["floor_agent"],
                "orchestrator_policy": "stub",
            }
        )

        assert config.trainable_roles == ("floor_agent",)
        assert config.policy_for_role("orchestrator") == "stub"
        assert config.policy_for_role("floor_agent") == "model"
        assert config.uses_role_routed_policy is True

    def test_training_config_rejects_trainable_stub_orchestrator(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                roles={
                    "trainable": ["orchestrator", "floor_agent"],
                    "orchestrator_policy": "stub",
                }
            )
        assert "orchestrator_policy='stub'" in str(excinfo.value)

    def test_training_config_rejects_selective_model_backed_training(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(roles={"trainable": ["floor_agent"]})
        assert "Selective roles.trainable currently requires" in str(excinfo.value)


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

    def test_training_config_defaults_to_unsloth_with_vllm_enabled(self):
        config = TrainingConfig()
        assert config.backend == "unsloth"
        assert config.rollout.use_vllm is True


class TestConfigContractRestoration:
    def test_training_config_rejects_legacy_grpo_group_size(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(grpo={"group_size": 4})
        assert "group_size" in str(excinfo.value)

    def test_training_config_loads_repository_yaml(self):
        raw = _load_yaml_config(Path("training/config.yaml"))
        config = TrainingConfig(**raw)
        assert config.reward.rationale_scaling == "linear_capped"
        assert config.model.base == "Qwen/Qwen2.5-3B-Instruct"
        assert config.model.uses_split_bases is False
        assert config.backend == "unsloth"
        assert config.rollout.use_vllm is True

    @pytest.mark.parametrize("value", ["off", "linear_capped", "log_uncapped"])
    def test_reward_config_accepts_known_rationale_scaling_values(self, value):
        assert RewardConfig(rationale_scaling=value).rationale_scaling == value

    def test_reward_config_rejects_unknown_rationale_scaling_value(self):
        with pytest.raises(ValidationError) as excinfo:
            RewardConfig(rationale_scaling="metadata_only")
        assert "rationale_scaling" in str(excinfo.value)

    def test_training_config_allows_role_specific_model_declaration(self):
        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "bigger-model",
                "floor_base": "smaller-model",
            }
        )

        assert config.model.uses_split_bases is True
        assert config.model.resolved_bases() == {
            "orchestrator": "bigger-model",
            "floor_agent": "smaller-model",
        }
