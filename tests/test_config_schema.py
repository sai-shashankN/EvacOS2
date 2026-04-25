from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from training.config_schema import ModelConfig, RewardConfig, RolloutConfig, TrainingConfig
from training.train import _load_yaml_config, _validate_config_path_identity


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

    def test_training_config_allows_orchestrator_only_with_frozen_floor_adapter(self):
        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "bigger-model",
                "floor_base": "smaller-model",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_adapter_paths": {
                    "floor_agent": "outputs/floor-specialist/lora_adapter/floor_agent",
                },
            },
        )

        assert config.trainable_roles == ("orchestrator",)
        assert config.roles.frozen_adapter_paths == {
            "floor_agent": "outputs/floor-specialist/lora_adapter/floor_agent",
        }
        assert config.uses_role_routed_policy is True

    def test_training_config_allows_orchestrator_with_frozen_floor_specialists(self):
        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "bigger-model",
                "floor_base": "smaller-model",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_floor_specialist_adapter_paths": {
                    "fire": "outputs/fire/lora_adapter/floor_agent",
                    "flood": "outputs/flood/lora_adapter/floor_agent",
                    "gas": "outputs/gas/lora_adapter/floor_agent",
                },
            },
            rollout={"disaster_families": ["fire", "flood", "gas"]},
        )

        assert config.trainable_roles == ("orchestrator",)
        assert config.roles.frozen_floor_specialist_adapter_paths["fire"].endswith(
            "floor_agent"
        )
        assert config.uses_role_routed_policy is True

    def test_training_config_rejects_trainable_floor_specialist_paths(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                roles={
                    "trainable": ["orchestrator", "floor_agent"],
                    "frozen_floor_specialist_adapter_paths": {
                        "fire": "outputs/fire/lora_adapter/floor_agent",
                    },
                },
                rollout={"disaster_families": ["fire"]},
            )

        assert "floor specialists must be frozen" in str(excinfo.value)

    def test_training_config_rejects_missing_requested_specialist_without_fallback(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                roles={
                    "trainable": ["orchestrator"],
                    "frozen_floor_specialist_adapter_paths": {
                        "fire": "outputs/fire/lora_adapter/floor_agent",
                    },
                },
                rollout={"disaster_families": ["fire", "flood"]},
            )

        assert "no matching roles.frozen_floor_specialist_adapter_paths" in str(excinfo.value)

    def test_training_config_allows_generalist_fallback_for_unsupported_family(self):
        config = TrainingConfig(
            roles={
                "trainable": ["orchestrator"],
                "frozen_adapter_paths": {
                    "floor_agent": "outputs/generalist/lora_adapter/floor_agent",
                },
                "frozen_floor_specialist_adapter_paths": {
                    "fire": "outputs/fire/lora_adapter/floor_agent",
                },
            },
            rollout={"disaster_families": ["fire", "structural"]},
        )

        assert config.roles.frozen_adapter_paths["floor_agent"].endswith("floor_agent")
        assert config.uses_role_routed_policy is True

    def test_training_config_rejects_frozen_adapter_for_trainable_role(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                roles={
                    "trainable": ["orchestrator"],
                    "frozen_adapter_paths": {"orchestrator": "outputs/orch"},
                },
            )
        assert "frozen_adapter_paths may only target non-trainable" in str(excinfo.value)


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


class TestTierScheduleConfig:
    def test_tier_schedule_expands_with_balanced_replay_inside_each_stage(self):
        config = TrainingConfig(
            max_steps=5,
            rollout={
                "use_vllm": True,
                "tier_schedule": [
                    {"steps": 2, "mix": {"easy": 2}},
                    {"steps": 3, "mix": {"hard": 2, "medium": 1}},
                ],
            },
        )

        assert config.rollout.expanded_tier_schedule() == [
            "easy",
            "easy",
            "hard",
            "medium",
            "hard",
        ]

    def test_tier_schedule_rejects_step_mismatch_inside_block(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                max_steps=3,
                rollout={
                    "use_vllm": True,
                    "tier_schedule": [
                        {"steps": 3, "mix": {"easy": 2}},
                    ],
                },
            )

        assert "mix counts must sum to steps" in str(excinfo.value)

    def test_tier_schedule_rejects_max_steps_mismatch(self):
        with pytest.raises(ValidationError) as excinfo:
            TrainingConfig(
                max_steps=4,
                rollout={
                    "use_vllm": True,
                    "tier_schedule": [
                        {"steps": 3, "mix": {"easy": 3}},
                    ],
                },
            )

        assert "expand to max_steps" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("path", "family"),
        [
            ("training/config.remote-unsloth-3b-fire-floor-specialist-750.yaml", "fire"),
            ("training/config.remote-unsloth-3b-flood-floor-specialist-750.yaml", "flood"),
            ("training/config.remote-unsloth-3b-gas-floor-specialist-750.yaml", "gas"),
        ],
    )
    def test_specialist_750_configs_have_expected_schedule(self, path, family):
        raw = _load_yaml_config(Path(path))
        config = TrainingConfig(**raw)
        schedule = config.rollout.expanded_tier_schedule()

        assert config.max_steps == 750
        assert config.roles.trainable == ["floor_agent"]
        assert config.roles.orchestrator_policy == "stub"
        assert config.rollout.disaster_families == [family]
        assert len(schedule) == 750
        assert Counter(schedule[:200]) == Counter({"easy": 200})
        assert Counter(schedule[200:400]) == Counter({"medium": 160, "easy": 40})
        assert Counter(schedule[400:600]) == Counter(
            {"hard": 160, "medium": 30, "easy": 10}
        )
        assert Counter(schedule[600:750]) == Counter(
            {"brutal": 115, "hard": 25, "medium": 10}
        )
        assert schedule[200:400].count("easy") == 40
        assert schedule[200] == "medium"
        assert "easy" in schedule[360:400]
        assert schedule[400] == "hard"
        assert {"easy", "medium", "hard"} <= set(schedule[400:600])
        assert schedule[600] == "brutal"
        assert {"medium", "hard", "brutal"} <= set(schedule[600:750])

    @pytest.mark.parametrize(
        ("path", "family"),
        [
            ("training/config.remote-unsloth-3b-fire-floor-specialist-2000.yaml", "fire"),
            ("training/config.remote-unsloth-3b-flood-floor-specialist-2000.yaml", "flood"),
            ("training/config.remote-unsloth-3b-gas-floor-specialist-2000.yaml", "gas"),
        ],
    )
    def test_specialist_2000_configs_have_expected_schedule(self, path, family):
        raw = _load_yaml_config(Path(path))
        config = TrainingConfig(**raw)
        schedule = config.rollout.expanded_tier_schedule()

        assert config.max_steps == 2000
        assert config.rollout.disaster_families == [family]
        assert len(schedule) == 2000
        assert Counter(schedule[:500]) == Counter({"easy": 500})
        assert Counter(schedule[500:1000]) == Counter({"medium": 400, "easy": 100})
        assert Counter(schedule[1000:1500]) == Counter(
            {"hard": 400, "medium": 75, "easy": 25}
        )
        assert Counter(schedule[1500:2000]) == Counter(
            {"brutal": 375, "hard": 100, "medium": 25}
        )

    def test_config_path_identity_rejects_mislabeled_step_count(self):
        config = TrainingConfig(
            max_steps=2000,
            rollout={
                "use_vllm": True,
                "tier_schedule": [{"steps": 2000, "mix": {"easy": 2000}}],
            },
            checkpoint={"root_dir": "outputs/training/debug-2000"},
            metrics={
                "csv_path": "outputs/training/debug-2000-metrics.csv",
                "jsonl_dir": "outputs/logs/debug-2000",
            },
        )

        with pytest.raises(ValueError, match="implies 750 steps"):
            _validate_config_path_identity(Path("training/debug-750.yaml"), config)

    def test_config_path_identity_rejects_mislabeled_output_root(self):
        config = TrainingConfig(max_steps=750)
        config.checkpoint.root_dir = "outputs/training/debug-2000"

        with pytest.raises(ValueError, match="checkpoint.root_dir"):
            _validate_config_path_identity(Path("training/debug.yaml"), config)


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
