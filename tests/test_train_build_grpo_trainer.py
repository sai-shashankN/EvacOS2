from __future__ import annotations

from types import SimpleNamespace

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
