from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

from scripts import run_fixed_suite


def _tmp_dir() -> Path:
    path = Path(tempfile.gettempdir()) / f"evacos_run_fixed_suite_cli_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_fixed_suite_cli_defaults_to_bounded_eval(monkeypatch):
    calls: list[dict] = []
    output_dir = _tmp_dir()

    def fake_run_fixed_suite(policy_factory, **kwargs):
        calls.append(kwargs)
        assert policy_factory is not None

    monkeypatch.setattr(run_fixed_suite, "run_fixed_suite", fake_run_fixed_suite)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fixed_suite.py",
            "--tiers",
            "easy",
            "--seeds",
            "42",
            "--families",
            "fire",
            "--output-dir",
            str(output_dir),
        ],
    )

    run_fixed_suite.main()

    assert calls[0]["max_rounds"] == 50


def test_run_fixed_suite_cli_checkpoint_uses_explicit_model_name(monkeypatch):
    created: dict[str, object] = {}
    output_dir = _tmp_dir()

    def fake_hf_policy_factory(model_name, *, lora_adapter_path):
        created["model_name"] = model_name
        created["lora_adapter_path"] = lora_adapter_path
        return object()

    def fake_run_fixed_suite(policy_factory, **kwargs):
        calls_policy = policy_factory()
        created["policy"] = calls_policy
        created["max_rounds"] = kwargs["max_rounds"]

    ckpt = output_dir / "adapter"
    ckpt.mkdir()
    monkeypatch.setattr(run_fixed_suite, "hf_policy_factory", fake_hf_policy_factory)
    monkeypatch.setattr(run_fixed_suite, "run_fixed_suite", fake_run_fixed_suite)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fixed_suite.py",
            "--checkpoint",
            str(ckpt),
            "--model-name",
            "Qwen/Qwen2.5-3B-Instruct",
            "--tiers",
            "easy",
            "--seeds",
            "42",
            "--families",
            "fire",
            "--output-dir",
            str(output_dir),
        ],
    )

    run_fixed_suite.main()

    assert created["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert created["lora_adapter_path"] == str(ckpt)
    assert created["max_rounds"] == 50
