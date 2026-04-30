from __future__ import annotations

import os
import random
import shutil
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

import pytest

from evacos_ma.models import DisasterType
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


def test_orchestrator_only_model_backed_builds_orchestrator_trainer_only(monkeypatch):
    config = TrainingConfig(
        model={
            "base": "shared-model",
            "orchestrator_base": "bigger-model",
            "floor_base": "smaller-model",
        },
        roles={"trainable": ["orchestrator"]},
    )
    monkeypatch.setattr(train_mod, "MultiAgentGRPOTrainer", FakeProjectTrainer)

    trainer = train_mod._build_grpo_trainer(
        GRPOTrainer=MockTRLTrainerModelOnly,
        policy=_make_dual_policy(),
        config=config,
        role_optimizer_states={"orchestrator": {"state": "orch-only"}},
    )

    assert isinstance(trainer, train_mod.DualRoleGRPOTrainer)
    assert set(trainer._role_trainers) == {"orchestrator"}
    assert trainer._role_trainers["orchestrator"].kwargs["model"] == "orch-model"
    assert trainer._role_trainers["orchestrator"].kwargs["optimizer_state"] == {
        "state": "orch-only"
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


def test_policy_sampling_temperature_can_be_temporarily_zeroed_and_restored() -> None:
    hf_like = SimpleNamespace(_gen_kwargs={"temperature": 0.7, "do_sample": True})
    unsloth_like = SimpleNamespace(_temperature=0.7)
    nested = SimpleNamespace(
        _role_policies={
            "orchestrator": hf_like,
            "floor_agent": SimpleNamespace(
                _specialist_policies={"fire_specialist": unsloth_like},
                _generalist_policy=None,
            ),
        }
    )

    tokens = train_mod._set_policy_sampling_temperature(nested, 0.0)

    assert hf_like._gen_kwargs["temperature"] == 0.0
    assert hf_like._gen_kwargs["do_sample"] is False
    assert unsloth_like._temperature == 0.0

    train_mod._restore_policy_sampling_temperature(tokens)

    assert hf_like._gen_kwargs == {"temperature": 0.7, "do_sample": True}
    assert unsloth_like._temperature == 0.7


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


def test_build_policy_uses_frozen_floor_adapter_for_orchestrator_only_training(
    monkeypatch,
):
    tmp_dir = _tmp_dir()
    try:
        frozen_floor = tmp_dir / "frozen_floor"
        frozen_floor.mkdir(parents=True)
        (frozen_floor / "adapter_config.json").write_text("{}", encoding="utf-8")

        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "Qwen/Qwen2.5-7B-Instruct",
                "floor_base": "Qwen/Qwen2.5-3B-Instruct",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_adapter_paths": {"floor_agent": str(frozen_floor)},
            },
            rollout={"use_vllm": False},
        )
        captured: list[tuple[str, dict[str, object]]] = []

        import training.policy_adapter as policy_module

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        def fake_unsloth_policy_factory(model_name, **kwargs):
            captured.append((model_name, kwargs))
            return {"model_name": model_name, "adapter": kwargs.get("lora_adapter_path")}

        monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

        policy = train_mod._build_policy(config, None, LoraConfig=SimpleNamespace)

        assert len(captured) == 2
        assert policy.orchestrator_policy["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert policy.orchestrator_policy["adapter"] is None
        assert policy.floor_policy["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert policy.floor_policy["adapter"] == str(frozen_floor)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_policy_routes_three_frozen_floor_specialists_for_orchestrator_training(
    monkeypatch,
):
    tmp_dir = _tmp_dir()
    try:
        specialist_paths: dict[str, Path] = {}
        for family in ("fire", "flood", "gas"):
            path = tmp_dir / family / "lora_adapter" / "floor_agent"
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            specialist_paths[family] = path

        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "Qwen/Qwen2.5-7B-Instruct",
                "floor_base": "Qwen/Qwen2.5-3B-Instruct",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_floor_specialist_adapter_paths": {
                    family: str(path)
                    for family, path in specialist_paths.items()
                },
            },
            rollout={
                "use_vllm": False,
                "disaster_families": ["fire", "flood", "gas"],
            },
        )
        captured: list[tuple[str, dict[str, object]]] = []

        import training.policy_adapter as policy_module

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        class FakeGeneratedPolicy:
            def __init__(self, model_name: str, adapter: object):
                self.model_name = model_name
                self.adapter = adapter

            def act(self, prompt, agent_id, role):
                return f"{self.model_name}:{self.adapter}:{agent_id}", []

        def fake_unsloth_policy_factory(model_name, **kwargs):
            captured.append((model_name, kwargs))
            return FakeGeneratedPolicy(model_name, kwargs.get("lora_adapter_path"))

        monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

        policy = train_mod._build_policy(config, None, LoraConfig=SimpleNamespace)

        assert policy.orchestrator_policy.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert sorted(policy.floor_policy.specialist_policy_keys) == [
            "fire_specialist",
            "flood_specialist",
            "gas_specialist",
        ]
        assert len(captured) == 4
        assert captured[0] == (
            "Qwen/Qwen2.5-7B-Instruct",
            captured[0][1],
        )
        captured_adapters = {
            kwargs.get("lora_adapter_path")
            for _model_name, kwargs in captured[1:]
        }
        assert captured_adapters == {str(path) for path in specialist_paths.values()}
        fire_output = policy.floor_policy.act(
            [{"role": "system", "content": "Disaster: fire\nRound: 0"}],
            "floor_0_agent",
            "floor_agent",
        )
        assert str(specialist_paths["fire"]) in fire_output[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_policy_resume_prefers_checkpoint_frozen_adapter_over_config_path(
    monkeypatch,
):
    tmp_dir = _tmp_dir()
    try:
        ckpt_adapter_root = tmp_dir / "latest" / "lora_adapter"
        ckpt_orch = ckpt_adapter_root / "orchestrator"
        ckpt_floor = ckpt_adapter_root / "floor_agent"
        ckpt_orch.mkdir(parents=True)
        ckpt_floor.mkdir(parents=True)
        missing_original = tmp_dir / "deleted_original_floor_adapter"

        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "Qwen/Qwen2.5-7B-Instruct",
                "floor_base": "Qwen/Qwen2.5-3B-Instruct",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_adapter_paths": {"floor_agent": str(missing_original)},
            },
            rollout={"use_vllm": False},
        )
        bundle = SimpleNamespace(
            lora_weights_path=ckpt_adapter_root,
            role_lora_weights_paths={
                "orchestrator": ckpt_orch,
                "floor_agent": ckpt_floor,
            },
        )
        captured: list[tuple[str, dict[str, object]]] = []

        import training.policy_adapter as policy_module

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        def fake_unsloth_policy_factory(model_name, **kwargs):
            captured.append((model_name, kwargs))
            return {"model_name": model_name, "adapter": kwargs.get("lora_adapter_path")}

        monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

        policy = train_mod._build_policy(config, bundle, LoraConfig=SimpleNamespace)

        assert policy.orchestrator_policy["adapter"] == str(ckpt_orch)
        assert policy.floor_policy["adapter"] == str(ckpt_floor)
        assert len(captured) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_policy_resume_prefers_checkpoint_floor_specialists_over_config_paths(
    monkeypatch,
):
    tmp_dir = _tmp_dir()
    try:
        ckpt_adapter_root = tmp_dir / "latest" / "lora_adapter"
        ckpt_orch = ckpt_adapter_root / "orchestrator"
        ckpt_orch.mkdir(parents=True)
        missing_originals: dict[str, Path] = {}
        ckpt_specialists: dict[str, Path] = {}
        for family in ("fire", "flood", "gas"):
            missing_originals[family] = tmp_dir / f"deleted_{family}_adapter"
            ckpt_dir = ckpt_adapter_root / "floor_agent" / "specialists" / family
            ckpt_dir.mkdir(parents=True)
            ckpt_specialists[family] = ckpt_dir

        config = TrainingConfig(
            model={
                "base": "shared-model",
                "orchestrator_base": "Qwen/Qwen2.5-7B-Instruct",
                "floor_base": "Qwen/Qwen2.5-3B-Instruct",
            },
            roles={
                "trainable": ["orchestrator"],
                "frozen_floor_specialist_adapter_paths": {
                    family: str(path)
                    for family, path in missing_originals.items()
                },
            },
            rollout={
                "use_vllm": False,
                "disaster_families": ["fire", "flood", "gas"],
            },
        )
        bundle = SimpleNamespace(
            lora_weights_path=ckpt_adapter_root,
            role_lora_weights_paths={"orchestrator": ckpt_orch},
            floor_specialist_lora_weights_paths=ckpt_specialists,
        )
        captured: list[tuple[str, dict[str, object]]] = []

        import training.policy_adapter as policy_module

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        class FakeGeneratedPolicy:
            def __init__(self, model_name: str, adapter: object):
                self.model_name = model_name
                self.adapter = adapter

            def act(self, prompt, agent_id, role):
                return f"{self.adapter}:{agent_id}", []

        def fake_unsloth_policy_factory(model_name, **kwargs):
            captured.append((model_name, kwargs))
            return FakeGeneratedPolicy(model_name, kwargs.get("lora_adapter_path"))

        monkeypatch.setattr(policy_module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(policy_module, "unsloth_policy_factory", fake_unsloth_policy_factory)

        policy = train_mod._build_policy(config, bundle, LoraConfig=SimpleNamespace)

        assert policy.orchestrator_policy.adapter == str(ckpt_orch)
        captured_adapters = {
            kwargs.get("lora_adapter_path")
            for _model_name, kwargs in captured[1:]
        }
        assert captured_adapters == {str(path) for path in ckpt_specialists.values()}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_save_adapter_weights_copies_frozen_adapter_without_saving_full_floor_model():
    tmp_dir = _tmp_dir()
    try:
        target = tmp_dir / "checkpoint" / "lora_adapter"
        frozen_floor = tmp_dir / "frozen_floor"
        frozen_floor.mkdir(parents=True)
        (frozen_floor / "adapter_config.json").write_text("{}", encoding="utf-8")
        calls: list[str] = []

        class FakeModel:
            def __init__(self, role: str):
                self.role = role

            def save_pretrained(self, path: str) -> None:
                calls.append(self.role)
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "adapter_model.safetensors").write_text(
                    self.role,
                    encoding="utf-8",
                )

        policy = SimpleNamespace(
            _role_policies={
                "orchestrator": SimpleNamespace(_model=FakeModel("orchestrator")),
                "floor_agent": SimpleNamespace(_model=FakeModel("floor_agent")),
            }
        )

        saved = train_mod._save_adapter_weights(
            policy,
            target,
            roles_to_save={"orchestrator"},
            role_adapter_paths_to_copy={"floor_agent": frozen_floor},
        )

        assert saved == {
            "orchestrator": target / "orchestrator",
            "floor_agent": target / "floor_agent",
        }
        assert calls == ["orchestrator"]
        assert (target / "floor_agent" / "adapter_config.json").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_copy_floor_specialist_adapter_trees_preserves_family_paths():
    tmp_dir = _tmp_dir()
    try:
        sources: dict[str, Path] = {}
        for family in ("fire", "flood", "gas"):
            source = tmp_dir / "source" / family
            source.mkdir(parents=True)
            (source / "adapter_config.json").write_text(family, encoding="utf-8")
            sources[family] = source

        target = tmp_dir / "checkpoint" / "lora_adapter" / "floor_agent" / "specialists"
        saved = train_mod._copy_floor_specialist_adapter_trees(sources, target)

        assert saved == {
            "fire": target / "fire",
            "flood": target / "flood",
            "gas": target / "gas",
        }
        for family in sources:
            assert (target / family / "adapter_config.json").read_text(encoding="utf-8") == family
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
        "orchestrator_group_raw_reward_std_mean": 0.1,
        "floor_agent_group_raw_reward_std_mean": 0.5,
        "orchestrator_singleton_group_rate": 0.0,
        "floor_agent_singleton_group_rate": 0.25,
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
    assert metrics_row["group_raw_reward_std_mean"] == pytest.approx(
        (0.1 * 1 + 0.5 * 5) / 6.0
    )
    assert metrics_row["singleton_group_rate"] == pytest.approx((0.0 * 1 + 0.25 * 5) / 6.0)
    assert metrics_row["kl_max"] == 0.05
    assert metrics_row["num_inner_epochs"] == 3


def test_multi_agent_grpo_trainer_streams_trainable_logprob_chunks(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setenv("EVACOS_LOGPROB_MICROBATCH_SIZE", "2")

    class TinyTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "right"
        model_max_length = 16

        def apply_chat_template(self, prompt, *, tokenize, add_generation_prompt):
            if tokenize:
                return [1, 2]
            return "prompt"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask=None):
            self.forward_calls += 1
            return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))

        def disable_adapter(self):
            return nullcontext()

    model = TinyModel()
    trainer = train_mod.MultiAgentGRPOTrainer(
        model=model,
        tokenizer=TinyTokenizer(),
        learning_rate=1e-3,
        kl_coef=0.04,
        clip_range=0.2,
        num_train_epochs_per_step=1,
    )
    grouped_inputs = {
        "prompts": [
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            [[{"role": "user", "content": "c"}], [{"role": "user", "content": "d"}]],
        ],
        "completions": [["aa", "bb"], ["cc", "dd"]],
        "completion_token_ids": [[[3, 4], [4, 5]], [[5, 6], [6, 7]]],
        "raw_rewards": [[1.0, -1.0], [0.5, -0.5]],
        "samples": [[SimpleNamespace(role="floor_agent")], [SimpleNamespace(role="floor_agent")]],
    }

    diagnostics = trainer.step(grouped_inputs=grouped_inputs)

    assert model.forward_calls >= 6
    assert diagnostics["num_inner_epochs"] == 1
    assert diagnostics["mask_coverage"] > 0
    assert diagnostics["advantage_std"] > 0
    assert diagnostics["group_raw_reward_std_mean"] > 0
    assert diagnostics["group_raw_reward_std_max"] > 0
    assert diagnostics["singleton_group_rate"] == 0


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
                    parsed_action={"action_type": "route_within_floor", "arguments": {"from_room_id": "room_2", "exit_id": "exit_2"}},
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
    assert metrics["wait_rate"] == pytest.approx(2 / 6, rel=0, abs=1e-4)
    assert metrics["floor_agent_wait_rate"] == pytest.approx(1 / 4, rel=0, abs=1e-4)
    assert metrics["orchestrator_wait_rate"] == 0.5
    assert metrics["empty_args_rate"] == pytest.approx(3 / 6, rel=0, abs=1e-4)
    assert metrics["floor_agent_active_action_rate"] == pytest.approx(3 / 4, rel=0, abs=1e-4)
    assert metrics["active_empty_args_rate"] == pytest.approx(1 / 6, rel=0, abs=1e-4)
    assert metrics["valid_but_hollow_action_rate"] == pytest.approx(2 / 6, rel=0, abs=1e-4)
    assert metrics["floor_scout_action_rate"] == 0.0
    assert metrics["floor_route_action_rate"] == pytest.approx(2 / 4, rel=0, abs=1e-4)
    assert metrics["floor_evacuate_action_rate"] == 0.0
    assert metrics["floor_route_exit_rate"] == pytest.approx(1 / 2, rel=0, abs=1e-4)
    assert metrics["floor_route_legacy_egress_alias_rate"] == pytest.approx(1 / 2, rel=0, abs=1e-4)


def test_compute_rollout_metrics_tracks_priority_components_and_family_mix() -> None:
    results = [
        SimpleNamespace(
            disaster_family="fire",
            override_count=0,
            orchestrator_action_count=1,
            override_win_count=0,
            rationale_bonus_total=0.0,
            rationale_bonus_count=0,
            priority_component_totals={
                "priority_top_match": 0.25,
                "priority_rank_score": 0.20,
                "priority_coverage": 0.10,
                "priority_effect_bonus": 0.10,
                "priority_unchanged_penalty": -0.08,
            },
            priority_component_counts={
                "priority_top_match": 1,
                "priority_rank_score": 1,
                "priority_coverage": 1,
                "priority_effect_bonus": 1,
                "priority_unchanged_penalty": 1,
            },
            priority_behavior_totals={
                "priority_top_match_rate": 1.0,
                "priority_rank_fraction_mean": 0.8,
                "priority_coverage_fraction_mean": 1.0,
                "priority_effect_bonus_rate": 1.0,
                "priority_unchanged_rate": 0.0,
            },
            priority_behavior_counts={
                "priority_top_match_rate": 1,
                "priority_rank_fraction_mean": 1,
                "priority_coverage_fraction_mean": 1,
                "priority_effect_bonus_rate": 1,
                "priority_unchanged_rate": 1,
            },
            priority_directive_issue_count=1,
            samples=[
                SimpleNamespace(
                    role="orchestrator",
                    parsed_action={
                        "action_type": "evacuate_floor_priority",
                        "arguments": {"ordered_floor_ids": ["floor_2", "floor_1"]},
                    },
                ),
            ],
        ),
        SimpleNamespace(
            disaster_family="gas",
            override_count=0,
            orchestrator_action_count=1,
            override_win_count=0,
            rationale_bonus_total=0.0,
            rationale_bonus_count=0,
            priority_component_totals={},
            priority_component_counts={},
            priority_behavior_totals={},
            priority_behavior_counts={},
            priority_directive_issue_count=0,
            samples=[
                SimpleNamespace(
                    role="orchestrator",
                    parsed_action={"action_type": "wait", "arguments": {}},
                ),
            ],
        ),
        SimpleNamespace(
            disaster_family=DisasterType.flood,
            override_count=0,
            orchestrator_action_count=0,
            override_win_count=0,
            rationale_bonus_total=0.0,
            rationale_bonus_count=0,
            priority_component_totals={},
            priority_component_counts={},
            priority_behavior_totals={},
            priority_behavior_counts={},
            priority_directive_issue_count=0,
            samples=[],
        ),
    ]

    metrics = train_mod._compute_rollout_metrics(results)

    assert metrics["priority_action_rate"] == pytest.approx(0.5, rel=0, abs=1e-4)
    assert metrics["priority_directive_issue_rate"] == 1.0
    assert metrics["priority_top_match_mean"] == 0.25
    assert metrics["priority_rank_score_mean"] == 0.20
    assert metrics["priority_coverage_mean"] == 0.10
    assert metrics["priority_effect_bonus_mean"] == 0.10
    assert metrics["priority_unchanged_penalty_mean"] == -0.08
    assert metrics["priority_top_match_rate"] == 1.0
    assert metrics["priority_rank_fraction_mean"] == 0.8
    assert metrics["priority_coverage_fraction_mean"] == 1.0
    assert metrics["priority_effect_bonus_rate"] == 1.0
    assert metrics["priority_unchanged_rate"] == 0.0
    assert metrics["family_fire_fraction"] == pytest.approx(1 / 3, rel=0, abs=1e-4)
    assert metrics["family_flood_fraction"] == pytest.approx(1 / 3, rel=0, abs=1e-4)
    assert metrics["family_gas_fraction"] == pytest.approx(1 / 3, rel=0, abs=1e-4)


def test_compute_rollout_metrics_uses_selected_candidates_for_action_rates() -> None:
    results = [
        SimpleNamespace(
            override_count=0,
            orchestrator_action_count=0,
            override_win_count=0,
            rationale_bonus_total=0.0,
            rationale_bonus_count=0,
            samples=[
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={
                        "action_type": "wait",
                        "arguments": {},
                        "candidate_index": 0,
                        "selected_for_execution": False,
                    },
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={
                        "action_type": "route_within_floor",
                        "arguments": {"from_room_id": "room_1", "exit_id": "exit_1"},
                        "candidate_index": 1,
                        "selected_for_execution": True,
                    },
                ),
            ],
        )
    ]

    metrics = train_mod._compute_rollout_metrics(results)

    assert metrics["wait_rate"] == 0.0
    assert metrics["floor_agent_wait_rate"] == 0.0
    assert metrics["valid_but_hollow_action_rate"] == 0.0
    assert metrics["floor_route_action_rate"] == 1.0
    assert metrics["floor_route_exit_rate"] == 1.0


def test_compute_rollout_metrics_excludes_rejected_selected_actions_from_behavior_rates() -> None:
    results = [
        SimpleNamespace(
            override_count=0,
            orchestrator_action_count=0,
            override_win_count=0,
            rationale_bonus_total=0.0,
            rationale_bonus_count=0,
            samples=[
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={
                        "action_type": "route_within_floor",
                        "arguments": {"to_room_id": "none"},
                        "selected_for_execution": True,
                        "fallback_reason": "env_rejected",
                        "rejection_reason": "unknown_route_target_id: none",
                    },
                ),
                SimpleNamespace(
                    role="floor_agent",
                    parsed_action={
                        "action_type": "wait",
                        "arguments": {},
                        "selected_for_execution": True,
                    },
                ),
            ],
        )
    ]

    metrics = train_mod._compute_rollout_metrics(results)

    assert metrics["wait_rate"] == 1.0
    assert metrics["floor_route_action_rate"] == 0.0
    assert metrics["floor_route_room_rate"] == 0.0
    assert metrics["valid_but_hollow_action_rate"] == 1.0


def test_training_watchdog_flags_zero_grpo_signal_after_window() -> None:
    config = TrainingConfig(
        watchdog={"warmup_steps": 1, "zero_signal_window": 2},
        rollout={"use_vllm": False},
    )
    metrics_row = {
        "policy_loss": 0.0,
        "advantage_std": 0.0,
        "group_raw_reward_std_mean": 0.0,
        "valid_but_hollow_action_rate": 0.0,
        "floor_scout_action_rate": 0.0,
        "floor_route_action_rate": 0.2,
        "floor_evacuate_action_rate": 0.0,
    }

    reason = train_mod._training_watchdog_reason(
        step=5,
        metrics_row=metrics_row,
        config=config,
        zero_signal_streak=2,
    )

    assert reason is not None
    assert reason.startswith("zero_grpo_signal")


def test_training_watchdog_flags_scout_dominance() -> None:
    config = TrainingConfig(
        watchdog={"warmup_steps": 1, "max_floor_scout_action_rate": 0.8},
        rollout={"use_vllm": False},
    )
    metrics_row = {
        "policy_loss": -0.1,
        "advantage_std": 0.5,
        "group_raw_reward_std_mean": 0.2,
        "valid_but_hollow_action_rate": 0.0,
        "floor_scout_action_rate": 0.9,
        "floor_route_action_rate": 0.0,
        "floor_evacuate_action_rate": 0.0,
    }

    reason = train_mod._training_watchdog_reason(
        step=5,
        metrics_row=metrics_row,
        config=config,
        zero_signal_streak=0,
    )

    assert reason is not None
    assert reason.startswith("scout_dominance")


def test_population_std_uses_zero_for_singleton_and_population_variance() -> None:
    assert train_mod._population_std([]) == 0.0
    assert train_mod._population_std([2.0]) == 0.0
    assert train_mod._population_std([1.0, 3.0]) == pytest.approx(1.0)
