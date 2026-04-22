from __future__ import annotations

import importlib
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
