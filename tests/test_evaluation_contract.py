from __future__ import annotations

import ast
from pathlib import Path


def test_evaluation_imports_only_through_contract_shim():
    evaluation_dir = Path("evaluation")
    allowed = evaluation_dir / "_training_contract.py"

    for path in evaluation_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "training", f"direct training import in {path}"
                    assert not alias.name.startswith("training."), f"direct training import in {path}"
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "training" or node.module.startswith("training."):
                    assert path == allowed, f"direct training import in {path}"


def test_contract_shim_reexports_minimum_surface():
    from evaluation._training_contract import Policy, RewardNormalizer, collect_batch

    assert Policy is not None
    assert RewardNormalizer is not None
    assert collect_batch is not None
