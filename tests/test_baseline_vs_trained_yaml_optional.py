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
