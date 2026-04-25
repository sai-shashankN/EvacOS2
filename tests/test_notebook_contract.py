from __future__ import annotations

import json
from pathlib import Path


def _notebook_text() -> str:
    nb = json.loads(Path("notebooks/train_evacos_ma.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


def test_training_notebook_uses_tested_remote_dependency_pins():
    text = _notebook_text()

    assert '"transformers==4.56.2"' in text
    assert '"trl==0.24.0"' in text
    assert '"peft==0.19.1"' in text
    assert '"fsspec==2025.9.0"' in text
    assert "peft<0.18" not in text
    assert "pip install -q trl " not in text


def test_training_notebook_exposes_adapter_upload_path():
    text = _notebook_text()

    assert "HF_ADAPTER_REPO" in text
    assert "scripts/upload_adapter.py" in text
    assert "HF_TOKEN" in text


def test_training_notebook_is_clean_for_judge_rerun():
    nb = json.loads(Path("notebooks/train_evacos_ma.ipynb").read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
