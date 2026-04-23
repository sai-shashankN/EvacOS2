from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.config_schema import TrainingConfig
from training import train as train_mod


class FakeProjectTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockTRLTrainerModelOnly:
    def __init__(self, model):
        self.model = model


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
