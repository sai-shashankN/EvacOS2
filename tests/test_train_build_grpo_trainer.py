from __future__ import annotations

import os
import random
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

from training.config_schema import TrainingConfig
from training import train as train_mod


class FakeProjectTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockTRLTrainerModelOnly:
    def __init__(self, model):
        self.model = model


def _tmp_dir() -> Path:
    path = Path(
        os.path.join(
            tempfile.gettempdir(),
            f"evacos_train_test_{os.getpid()}_{random.randint(0, 99999)}",
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_policy():
    return SimpleNamespace(_model="trainer-model", _tokenizer="trainer-tokenizer")


def _make_dual_policy():
    return SimpleNamespace(
        _role_policies={
            "orchestrator": SimpleNamespace(_model="orch-model", _tokenizer="orch-tokenizer"),
            "floor_agent": SimpleNamespace(_model="floor-model", _tokenizer="floor-tokenizer"),
        }
    )


def test_default_path_prefers_project_multi_agent_trainer(monkeypatch):
    config = TrainingConfig()
    monkeypatch.setattr(train_mod, "MultiAgentGRPOTrainer", FakeProjectTrainer)

    trainer = train_mod._build_grpo_trainer(
        GRPOTrainer=MockTRLTrainerModelOnly,
        policy=_make_policy(),
        config=config,
        optimizer_state={"state": "ok"},
    )

    assert isinstance(trainer, FakeProjectTrainer)
    assert trainer.kwargs["model"] == "trainer-model"
    assert trainer.kwargs["tokenizer"] == "trainer-tokenizer"
    assert trainer.kwargs["optimizer_state"] == {"state": "ok"}


def test_prefer_trl_uses_filtered_signature_without_string_match(monkeypatch):
    config = TrainingConfig(grpo={"prefer_trl": True})
    monkeypatch.setattr(train_mod, "MultiAgentGRPOTrainer", FakeProjectTrainer)

    trainer = train_mod._build_grpo_trainer(
        GRPOTrainer=MockTRLTrainerModelOnly,
        policy=_make_policy(),
        config=config,
    )

    assert isinstance(trainer, MockTRLTrainerModelOnly)
    assert trainer.model == "trainer-model"


def test_split_policy_builds_dual_role_trainer(monkeypatch):
    config = TrainingConfig(
        model={
            "base": "shared-model",
            "orchestrator_base": "bigger-model",
            "floor_base": "smaller-model",
        }
    )
    monkeypatch.setattr(train_mod, "MultiAgentGRPOTrainer", FakeProjectTrainer)

    trainer = train_mod._build_grpo_trainer(
        GRPOTrainer=MockTRLTrainerModelOnly,
        policy=_make_dual_policy(),
        config=config,
        role_optimizer_states={
            "orchestrator": {"state": "orch"},
            "floor_agent": {"state": "floor"},
        },
    )

    assert isinstance(trainer, train_mod.DualRoleGRPOTrainer)
    assert trainer._role_trainers["orchestrator"].kwargs["model"] == "orch-model"
    assert trainer._role_trainers["floor_agent"].kwargs["model"] == "floor-model"
    assert trainer._role_trainers["orchestrator"].kwargs["optimizer_state"] == {"state": "orch"}
    assert trainer._role_trainers["floor_agent"].kwargs["optimizer_state"] == {"state": "floor"}


def test_floor_only_stub_orchestrator_builds_floor_trainer_only(monkeypatch):
    config = TrainingConfig(
        roles={
            "trainable": ["floor_agent"],
            "orchestrator_policy": "stub",
        }
    )
    monkeypatch.setattr(train_mod, "MultiAgentGRPOTrainer", FakeProjectTrainer)

    policy = SimpleNamespace(
        _role_policies={
            "orchestrator": SimpleNamespace(),
            "floor_agent": SimpleNamespace(_model="floor-model", _tokenizer="floor-tokenizer"),
        }
    )

    trainer = train_mod._build_grpo_trainer(
        GRPOTrainer=MockTRLTrainerModelOnly,
        policy=policy,
        config=config,
        role_optimizer_states={"floor_agent": {"state": "floor-only"}},
    )

    assert isinstance(trainer, train_mod.DualRoleGRPOTrainer)
    assert set(trainer._role_trainers) == {"floor_agent"}
    assert trainer._role_trainers["floor_agent"].kwargs["model"] == "floor-model"
    assert trainer._role_trainers["floor_agent"].kwargs["optimizer_state"] == {
        "state": "floor-only"
    }


def test_dual_role_trainer_skips_untrainable_role_groups() -> None:
    floor_calls: list[dict[str, list[list[object]]]] = []

    class RecorderTrainer:
        def step(self, *, grouped_inputs):
            floor_calls.append(grouped_inputs)
            return {"loss": 1.5}

    trainer = train_mod.DualRoleGRPOTrainer(role_trainers={"floor_agent": RecorderTrainer()})
    orchestrator_sample = SimpleNamespace(role="orchestrator")
    floor_sample = SimpleNamespace(role="floor_agent")
    grouped_inputs = {
        "prompts": [[["orch"]], [["floor"]]],
        "completions": [["orch-completion"], ["floor-completion"]],
        "completion_token_ids": [[None], [None]],
        "raw_rewards": [[0.1], [0.2]],
        "normalized_rewards": [[0.1], [0.2]],
        "samples": [[orchestrator_sample], [floor_sample]],
    }

    diagnostics = trainer.step(grouped_inputs=grouped_inputs)

    assert len(floor_calls) == 1
    assert floor_calls[0]["samples"] == [[floor_sample]]
    assert "orchestrator_sample_groups" not in diagnostics
    assert diagnostics["floor_agent_sample_groups"] == 1
    assert diagnostics["floor_agent_loss"] == 1.5


def test_build_policy_uses_stub_orchestrator_without_loading_orchestrator_model(
    monkeypatch,
):
    config = TrainingConfig(
        roles={
            "trainable": ["floor_agent"],
            "orchestrator_policy": "stub",
        },
        rollout={"use_vllm": False},
    )
    captured: list[tuple[str, dict[str, object]]] = []

    import training.policy_adapter as policy_module

    class FakeRoleRoutedPolicy:
        def __init__(self, *, orchestrator_policy, floor_policy):
            self.orchestrator_policy = orchestrator_policy
            self.floor_policy = floor_policy

    class FakeStubPolicy:
        def __init__(self, seed=0):
            self.seed = seed

    def fake_unsloth_policy_factory(model_name, **kwargs):
        captured.append((model_name, kwargs))
        return {"model_name": model_name, "adapter": kwargs.get("lora_adapter_path")}

    monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
    monkeypatch.setattr(policy_module, "StubPolicy", FakeStubPolicy)
    monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

    policy = train_mod._build_policy(config, None, LoraConfig=SimpleNamespace)

    assert isinstance(policy.orchestrator_policy, FakeStubPolicy)
    assert policy.orchestrator_policy.seed == config.seed.training_rng
    assert len(captured) == 1
    assert captured[0][0] == "Qwen/Qwen2.5-3B-Instruct"
    assert policy.floor_policy["model_name"] == "Qwen/Qwen2.5-3B-Instruct"


def test_build_policy_resumes_floor_only_role_adapter(monkeypatch):
    tmp_dir = _tmp_dir()
    try:
        adapter_root = tmp_dir / "lora_adapter"
        floor_dir = adapter_root / "floor_agent"
        floor_dir.mkdir(parents=True)

        config = TrainingConfig(
            roles={
                "trainable": ["floor_agent"],
                "orchestrator_policy": "stub",
            },
            rollout={"use_vllm": False},
        )
        bundle = SimpleNamespace(
            lora_weights_path=adapter_root,
            role_lora_weights_paths={"floor_agent": floor_dir},
        )
        captured: list[tuple[str, dict[str, object]]] = []

        import training.policy_adapter as policy_module

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        class FakeStubPolicy:
            def __init__(self, seed=0):
                self.seed = seed

        def fake_unsloth_policy_factory(model_name, **kwargs):
            captured.append((model_name, kwargs))
            return {"model_name": model_name, "adapter": kwargs.get("lora_adapter_path")}

        monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(policy_module, "StubPolicy", FakeStubPolicy)
        monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

        policy = train_mod._build_policy(config, bundle, LoraConfig=SimpleNamespace)

        assert isinstance(policy.orchestrator_policy, FakeStubPolicy)
        assert len(captured) == 1
        assert captured[0][1]["lora_adapter_path"] == str(floor_dir)
        assert policy.floor_policy["adapter"] == str(floor_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_checkpoint_role_metadata_includes_floor_only_specialist() -> None:
    config = TrainingConfig(
        roles={
            "trainable": ["floor_agent"],
            "orchestrator_policy": "stub",
        },
        rollout={"use_vllm": False},
    )

    assert train_mod._checkpoint_role_model_names(config) == {
        "orchestrator": "Qwen/Qwen2.5-3B-Instruct",
        "floor_agent": "Qwen/Qwen2.5-3B-Instruct",
    }
    assert train_mod._checkpoint_orchestrator_policy(config) == "stub"


def test_merge_trainer_diagnostics_keeps_single_model_fields() -> None:
    metrics_row = {"step": 0}
    diagnostics = {
        "loss": 1.25,
        "policy_loss": 0.75,
        "ratio_mean": 0.98,
        "num_inner_epochs": 2,
    }

    train_mod._merge_trainer_diagnostics_into_metrics(metrics_row, diagnostics)

    assert metrics_row["loss"] == 1.25
    assert metrics_row["policy_loss"] == 0.75
    assert metrics_row["ratio_mean"] == 0.98
    assert metrics_row["num_inner_epochs"] == 2


def test_merge_trainer_diagnostics_aggregates_split_role_fields() -> None:
    metrics_row = {"step": 0}
    diagnostics = {
        "orchestrator_sample_groups": 1,
        "floor_agent_sample_groups": 5,
        "orchestrator_loss": 2.0,
        "floor_agent_loss": 4.0,
        "orchestrator_ratio_mean": 0.9,
        "floor_agent_ratio_mean": 1.1,
        "orchestrator_kl_max": 0.02,
        "floor_agent_kl_max": 0.05,
        "orchestrator_num_inner_epochs": 1,
        "floor_agent_num_inner_epochs": 3,
    }

    train_mod._merge_trainer_diagnostics_into_metrics(metrics_row, diagnostics)

    assert metrics_row["orchestrator_loss"] == 2.0
    assert metrics_row["floor_agent_loss"] == 4.0
    assert metrics_row["orchestrator_sample_groups"] == 1
    assert metrics_row["floor_agent_sample_groups"] == 5
    assert metrics_row["loss"] == pytest.approx((2.0 * 1 + 4.0 * 5) / 6.0)
    assert metrics_row["ratio_mean"] == pytest.approx((0.9 * 1 + 1.1 * 5) / 6.0)
    assert metrics_row["kl_max"] == 0.05
    assert metrics_row["num_inner_epochs"] == 3


def test_compute_rollout_metrics_tracks_wait_and_hollow_behavior() -> None:
    results = [
        SimpleNamespace(
            override_count=1,
            orchestrator_action_count=2,
            override_win_count=1,
            rationale_bonus_total=0.5,
            rationale_bonus_count=2,
            samples=[
                SimpleNamespace(
                    role="orchestrator",
                    parsed_action={"action_type": "wait", "arguments": {}},
                ),
                SimpleNamespace(
                    role="orchestrator",
                    parsed_action={"action_type": "override_floor_agent", "arguments": {"target_floor_agent_id": "floor_0_agent"}},
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={"action_type": "wait", "arguments": {}, "fallback_reason": "parse_error"},
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={"action_type": "wait", "arguments": {}},
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={"action_type": "route_within_floor", "arguments": {"from_room_id": "room_1", "to_room_id": "exit_1"}},
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={"action_type": "open_exit", "arguments": {}},
                ),
            ],
        )
    ]

    metrics = train_mod._compute_rollout_metrics(results)

    assert metrics["override_rate"] == 0.5
    assert metrics["override_win_rate"] == 1.0
    assert metrics["rationale_bonus_mean"] == 0.25
    assert metrics["wait_rate"] == pytest.approx(3 / 6, rel=0, abs=1e-4)
    assert metrics["floor_agent_wait_rate"] == pytest.approx(2 / 4, rel=0, abs=1e-4)
    assert metrics["orchestrator_wait_rate"] == 0.5
    assert metrics["empty_args_rate"] == pytest.approx(4 / 6, rel=0, abs=1e-4)
    assert metrics["floor_agent_active_action_rate"] == pytest.approx(2 / 4, rel=0, abs=1e-4)
    assert metrics["active_empty_args_rate"] == pytest.approx(1 / 6, rel=0, abs=1e-4)
    assert metrics["valid_but_hollow_action_rate"] == pytest.approx(2 / 6, rel=0, abs=1e-4)
