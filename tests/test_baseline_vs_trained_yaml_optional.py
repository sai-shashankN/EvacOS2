from __future__ import annotations

import importlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest


def _tmp_dir() -> Path:
    path = Path(".phase21_test_tmp") / f"yaml_optional_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_baseline_vs_trained_imports_without_yaml(monkeypatch) -> None:
    tmp_path = _tmp_dir()
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text("model:\n  base: stub\n", encoding="utf-8")

        monkeypatch.setitem(sys.modules, "yaml", None)
        sys.modules.pop("evaluation.baseline_vs_trained", None)

        module = importlib.import_module("evaluation.baseline_vs_trained")

        with pytest.raises(ImportError, match="PyYAML|yaml"):
            module._load_model_name(config_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_load_model_name_prefers_single_role_override() -> None:
    tmp_path = _tmp_dir()
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    "  base: shared",
                    "  orchestrator_base: larger",
                ]
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        assert module._load_model_name(config_path) == "larger"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_load_model_name_rejects_split_role_models() -> None:
    tmp_path = _tmp_dir()
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    "  base: shared",
                    "  orchestrator_base: larger",
                    "  floor_base: smaller",
                ]
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        model_cfg = module._load_model_config(config_path)
        assert model_cfg["split"] is True
        assert model_cfg["orchestrator"] == "larger"
        assert model_cfg["floor_agent"] == "smaller"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_load_model_config_flags_same_base_selective_role_as_role_routed() -> None:
    tmp_path = _tmp_dir()
    config_path = tmp_path / "config.yaml"
    try:
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    "  base: shared",
                    "roles:",
                    "  trainable: [orchestrator]",
                    "  frozen_adapter_paths:",
                    "    floor_agent: outputs/floor",
                ]
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        model_cfg = module._load_model_config(config_path)
        assert model_cfg["split"] is False
        assert model_cfg["role_routed"] is True
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_trained_factory_supports_split_role_checkpoint(monkeypatch) -> None:
    tmp_path = _tmp_dir()
    config_path = tmp_path / "config.yaml"
    ckpt_root = tmp_path / "lora_adapter"
    orch = ckpt_root / "orchestrator"
    floor = ckpt_root / "floor_agent"
    try:
        orch.mkdir(parents=True)
        floor.mkdir(parents=True)
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    "  base: shared",
                    "  orchestrator_base: larger",
                    "  floor_base: smaller",
                ]
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        monkeypatch.setattr(module, "_load_model_config", lambda config_path=Path("training/config.yaml"): {
            "base": "shared",
            "orchestrator": "larger",
            "floor_agent": "smaller",
            "split": True,
        })

        captured = []

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        def _fake_factory(model_name, *, lora_adapter_path=None, **kwargs):
            captured.append((model_name, lora_adapter_path, kwargs))
            return {"model_name": model_name, "adapter": lora_adapter_path}

        monkeypatch.setattr(module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(module, "hf_policy_factory", _fake_factory)

        policy = module._trained_factory(ckpt_root)()

        assert policy.orchestrator_policy["model_name"] == "larger"
        assert policy.floor_policy["model_name"] == "smaller"
        assert captured[0][1] == str(orch)
        assert captured[1][1] == str(floor)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_trained_factory_prefers_checkpoint_metadata(monkeypatch) -> None:
    tmp_path = _tmp_dir()
    ckpt_dir = tmp_path / "latest"
    adapter_root = ckpt_dir / "lora_adapter"
    orch = adapter_root / "orchestrator"
    floor = adapter_root / "floor_agent"
    try:
        orch.mkdir(parents=True)
        floor.mkdir(parents=True)
        (ckpt_dir / "meta.json").write_text(
            json.dumps({
                "role_model_names": {
                    "orchestrator": "meta-orchestrator",
                    "floor_agent": "meta-floor",
                }
            }),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        monkeypatch.setattr(module, "_load_model_config", lambda config_path=Path("training/config.yaml"): (_ for _ in ()).throw(AssertionError("should not load config")))

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        def _fake_factory(model_name, *, lora_adapter_path=None, **kwargs):
            return {"model_name": model_name, "adapter": lora_adapter_path}

        monkeypatch.setattr(module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(module, "hf_policy_factory", _fake_factory)

        policy = module._trained_factory(ckpt_dir)()

        assert policy.orchestrator_policy["model_name"] == "meta-orchestrator"
        assert policy.floor_policy["model_name"] == "meta-floor"
        assert policy.orchestrator_policy["adapter"] == str(orch)
        assert policy.floor_policy["adapter"] == str(floor)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_trained_factory_treats_same_base_role_adapter_checkpoint_as_role_routed(
    monkeypatch,
) -> None:
    tmp_path = _tmp_dir()
    ckpt_dir = tmp_path / "latest"
    adapter_root = ckpt_dir / "lora_adapter"
    orch = adapter_root / "orchestrator"
    floor = adapter_root / "floor_agent"
    try:
        orch.mkdir(parents=True)
        floor.mkdir(parents=True)
        (ckpt_dir / "meta.json").write_text(
            json.dumps(
                {
                    "role_model_names": {
                        "orchestrator": "same-model",
                        "floor_agent": "same-model",
                    },
                    "role_lora_weights_paths": {
                        "orchestrator": str(orch),
                        "floor_agent": str(floor),
                    },
                }
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        monkeypatch.setattr(
            module,
            "_load_model_config",
            lambda config_path=Path("training/config.yaml"): (_ for _ in ()).throw(
                AssertionError("should not load config")
            ),
        )

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        def _fake_factory(model_name, *, lora_adapter_path=None, **kwargs):
            return {"model_name": model_name, "adapter": lora_adapter_path}

        monkeypatch.setattr(module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(module, "hf_policy_factory", _fake_factory)

        policy = module._trained_factory(ckpt_dir)()

        assert policy.orchestrator_policy["model_name"] == "same-model"
        assert policy.floor_policy["model_name"] == "same-model"
        assert policy.orchestrator_policy["adapter"] == str(orch)
        assert policy.floor_policy["adapter"] == str(floor)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_trained_factory_rebuilds_routed_floor_specialist_checkpoint(
    monkeypatch,
) -> None:
    tmp_path = _tmp_dir()
    ckpt_dir = tmp_path / "latest"
    adapter_root = ckpt_dir / "lora_adapter"
    orch = adapter_root / "orchestrator"
    specialist_root = adapter_root / "floor_agent" / "specialists"
    try:
        orch.mkdir(parents=True)
        specialist_paths: dict[str, Path] = {}
        for family in ("fire", "flood", "gas"):
            path = specialist_root / family
            path.mkdir(parents=True)
            specialist_paths[family] = path
        (ckpt_dir / "meta.json").write_text(
            json.dumps(
                {
                    "role_model_names": {
                        "orchestrator": "meta-orchestrator",
                        "floor_agent": "meta-floor",
                    },
                    "role_lora_weights_paths": {
                        "orchestrator": str(orch),
                    },
                    "floor_specialist_lora_weights_paths": {
                        family: str(path)
                        for family, path in specialist_paths.items()
                    },
                }
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        monkeypatch.setattr(
            module,
            "_load_model_config",
            lambda config_path=Path("training/config.yaml"): (_ for _ in ()).throw(
                AssertionError("should not load config")
            ),
        )

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        class FakeScopeRoutedFloorPolicy:
            def __init__(self, *, specialist_policies, generalist_policy=None):
                self.specialist_policies = specialist_policies
                self.generalist_policy = generalist_policy

        def _fake_factory(model_name, *, lora_adapter_path=None, **kwargs):
            return {"model_name": model_name, "adapter": lora_adapter_path}

        monkeypatch.setattr(module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(module, "ScopeRoutedFloorPolicy", FakeScopeRoutedFloorPolicy)
        monkeypatch.setattr(module, "hf_policy_factory", _fake_factory)

        policy = module._trained_factory(ckpt_dir)()

        assert policy.orchestrator_policy == {
            "model_name": "meta-orchestrator",
            "adapter": str(orch),
        }
        assert policy.floor_policy.generalist_policy is None
        assert policy.floor_policy.specialist_policies == {
            family: {
                "model_name": "meta-floor",
                "adapter": str(path),
            }
            for family, path in specialist_paths.items()
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_comparison_defaults_to_supported_routed_specialist_families(
    monkeypatch,
) -> None:
    tmp_path = _tmp_dir()
    ckpt_dir = tmp_path / "latest"
    adapter_root = ckpt_dir / "lora_adapter"
    orch = adapter_root / "orchestrator"
    specialist_root = adapter_root / "floor_agent" / "specialists"
    try:
        orch.mkdir(parents=True)
        for family in ("fire", "flood", "gas"):
            (specialist_root / family).mkdir(parents=True)
        (ckpt_dir / "meta.json").write_text(
            json.dumps(
                {
                    "role_model_names": {
                        "orchestrator": "meta-orchestrator",
                        "floor_agent": "meta-floor",
                    },
                    "role_lora_weights_paths": {
                        "orchestrator": str(orch),
                    },
                    "floor_specialist_lora_weights_paths": {
                        family: str(specialist_root / family)
                        for family in ("fire", "flood", "gas")
                    },
                }
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")
        from evaluation.fixed_suite import AggregateStats

        captured: dict[str, tuple[str, ...]] = {}

        def fake_run_fixed_suite(
            policy_factory,
            *,
            tiers,
            seeds,
            disaster_families,
            rationale_mode,
            label,
            output_dir,
            normalizer_snapshot=None,
        ):
            del policy_factory, normalizer_snapshot
            family_values = tuple(
                family.value if hasattr(family, "value") else str(family)
                for family in disaster_families
            )
            captured[label] = family_values
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"fixed_suite_{label}_{rationale_mode}.json").write_text(
                "{}",
                encoding="utf-8",
            )
            return module.FixedSuiteResult(
                label=label,
                tiers=list(tiers),
                seeds=[int(seed) for seed in seeds],
                disaster_families=list(family_values),
                rationale_mode=rationale_mode,
                episodes=[],
                aggregate=AggregateStats(),
            )

        monkeypatch.setattr(module, "run_fixed_suite", fake_run_fixed_suite)
        monkeypatch.setattr(module, "_trained_factory", lambda *args, **kwargs: (lambda: object()))

        module.run_comparison(
            trained_checkpoint=ckpt_dir,
            tiers=("easy",),
            seeds=(42,),
            output_csv=tmp_path / "baseline_vs_trained.csv",
        )

        assert captured["baseline"] == ("fire", "flood", "gas")
        assert captured["trained"] == ("fire", "flood", "gas")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_trained_factory_rebuilds_floor_only_stub_orchestrator_checkpoint(
    monkeypatch,
) -> None:
    tmp_path = _tmp_dir()
    ckpt_dir = tmp_path / "latest"
    adapter_root = ckpt_dir / "lora_adapter"
    floor = adapter_root / "floor_agent"
    try:
        floor.mkdir(parents=True)
        (ckpt_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": "Qwen/Qwen2.5-3B-Instruct",
                    "role_model_names": {
                        "orchestrator": "Qwen/Qwen2.5-3B-Instruct",
                        "floor_agent": "Qwen/Qwen2.5-3B-Instruct",
                    },
                    "orchestrator_policy": "stub",
                }
            ),
            encoding="utf-8",
        )

        sys.modules.pop("evaluation.baseline_vs_trained", None)
        module = importlib.import_module("evaluation.baseline_vs_trained")

        monkeypatch.setattr(
            module,
            "_load_model_config",
            lambda config_path=Path("training/config.yaml"): (_ for _ in ()).throw(
                AssertionError("should not load config")
            ),
        )

        captured: list[tuple[str, str | None, dict[str, object]]] = []

        class FakeRoleRoutedPolicy:
            def __init__(self, *, orchestrator_policy, floor_policy):
                self.orchestrator_policy = orchestrator_policy
                self.floor_policy = floor_policy

        class FakeStubPolicy:
            def __init__(self, seed=0):
                self.seed = seed

        def _fake_factory(model_name, *, lora_adapter_path=None, **kwargs):
            captured.append((model_name, lora_adapter_path, kwargs))
            return {"model_name": model_name, "adapter": lora_adapter_path}

        monkeypatch.setattr(module, "RoleRoutedPolicy", FakeRoleRoutedPolicy)
        monkeypatch.setattr(module, "StubPolicy", FakeStubPolicy)
        monkeypatch.setattr(module, "hf_policy_factory", _fake_factory)

        policy = module._trained_factory(ckpt_dir)()

        assert isinstance(policy.orchestrator_policy, FakeStubPolicy)
        assert policy.orchestrator_policy.seed == 0
        assert policy.floor_policy["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert policy.floor_policy["adapter"] == str(floor)
        assert captured == [
            (
                "Qwen/Qwen2.5-3B-Instruct",
                str(floor),
                {},
            )
        ]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
