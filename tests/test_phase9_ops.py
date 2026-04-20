"""Phase 9 ops tests — max_steps field + wandb gating.

All tests must pass with wandb NOT installed (Windows local-dev baseline).
"""

import sys

import pytest

from training.config_schema import TrainingConfig
from training.train import _maybe_init_wandb


class TestMaxStepsField:
    def test_default_is_none(self):
        assert TrainingConfig().max_steps is None

    def test_accepts_positive_int(self):
        assert TrainingConfig(max_steps=100).max_steps == 100

    def test_rejects_zero(self):
        with pytest.raises(Exception):
            TrainingConfig(max_steps=0)

    def test_rejects_negative(self):
        with pytest.raises(Exception):
            TrainingConfig(max_steps=-5)


class TestWandbGating:
    def test_returns_none_when_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        assert _maybe_init_wandb(TrainingConfig()) is None

    def test_returns_none_when_wandb_missing(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "dummy")
        # Force `import wandb` to raise ImportError even if wandb is present.
        real_wandb = sys.modules.pop("wandb", None)
        monkeypatch.setitem(sys.modules, "wandb", None)  # None triggers ImportError
        try:
            assert _maybe_init_wandb(TrainingConfig()) is None
        finally:
            if real_wandb is not None:
                sys.modules["wandb"] = real_wandb
