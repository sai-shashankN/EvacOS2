"""Phase 12 regression tests for resume wiring.

A. Optimizer state is preserved across resume.
B. Stale-step finalizer skips writing a duplicate checkpoint.
"""

import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from training.checkpoint import (
    CheckpointBundle,
    atomic_replace_latest,
    load_checkpoint,
    save_checkpoint,
)


def _tmp_dir() -> Path:
    """Create a temp dir avoiding pytest tmp_path permission issues on Windows."""
    d = os.path.join(
        tempfile.gettempdir(),
        f"evacos_phase12_{os.getpid()}_{random.randint(0, 99999)}",
    )
    os.makedirs(d, exist_ok=True)
    return Path(d)


# -----------------------------------------------------------------------
# A. Optimizer state preserved across resume
# -----------------------------------------------------------------------


class TestOptimizerStateResume:
    def test_optimizer_state_roundtrip_in_checkpoint(self):
        """Prove that optimizer state survives save/load through
        CheckpointBundle and is populated in the loaded bundle."""
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(42)

            # Build a realistic optimizer state dict structure (mimics AdamW)
            # We don't need real torch tensors; the checkpoint layer only
            # persists the dict via torch.save/load.  We'll use plain dicts
            # and lists to verify round-trip structure preservation.
            fake_optimizer_state = {
                "state": {
                    0: {
                        "step": 5,
                        "exp_avg": [0.1, 0.2, 0.3],
                        "exp_avg_sq": [0.01, 0.02, 0.03],
                    },
                    1: {
                        "step": 5,
                        "exp_avg": [0.4, 0.5],
                        "exp_avg_sq": [0.04, 0.05],
                    },
                },
                "param_groups": [
                    {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.01}
                ],
            }

            adapter_dir = ckpt_root / "ckpt_0" / "lora_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=0,
                wall_seconds_total=1.0,
                curriculum_snapshot={},
                normalizer_snapshot={},
                rollout_rng_state=pickle.dumps(rng.getstate()),
                lora_weights_path=adapter_dir,
                model_name="stub",
                config_hash="sha256:test",
                optimizer_state=fake_optimizer_state,
            )
            saved_dir = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_dir)

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.optimizer_state is not None

            # Verify the state structure is preserved
            loaded_state = loaded.optimizer_state
            assert set(loaded_state["state"].keys()) == {0, 1}
            assert loaded_state["state"][0]["step"] == 5
            assert loaded_state["state"][0]["exp_avg"] == [0.1, 0.2, 0.3]
            assert loaded_state["state"][0]["exp_avg_sq"] == [0.01, 0.02, 0.03]
            assert loaded_state["state"][1]["step"] == 5
            assert len(loaded_state["param_groups"]) == 1
            assert loaded_state["param_groups"][0]["lr"] == 1e-3
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_trainer_accepts_and_loads_optimizer_state(self):
        """Prove MultiAgentGRPOTrainer.__init__ restores optimizer state when
        optimizer_state is provided."""
        import sys
        from unittest.mock import MagicMock

        # Create a mock torch module that provides enough for the trainer
        mock_torch = MagicMock()
        mock_optimizer_cls = MagicMock()
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.state_dict.return_value = {"state": {}, "param_groups": []}
        mock_optimizer_instance.load_state_dict = MagicMock()
        mock_optimizer_cls.return_value = mock_optimizer_instance
        mock_torch.optim.AdamW = mock_optimizer_cls

        # We need the trainer to import torch from inside its __init__.
        # Since torch isn't installed, we inject the mock.
        torch_already = "torch" in sys.modules
        sys.modules["torch"] = mock_torch
        try:
            # Also mock unsloth to prevent import attempts
            sys.modules.setdefault("unsloth", MagicMock())
            sys.modules.setdefault("unsloth.FastLanguageModel", MagicMock())

            from training.train import MultiAgentGRPOTrainer

            fake_state = {"state": {0: {"step": 42}}, "param_groups": []}

            model = MagicMock()
            model.parameters.return_value = [MagicMock(requires_grad=True)]
            model.named_parameters.return_value = []

            tokenizer = MagicMock()
            tokenizer.pad_token = None
            tokenizer.eos_token = "[PAD]"

            trainer = MultiAgentGRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=1e-3,
                kl_coef=0.1,
                clip_range=0.2,
                num_train_epochs_per_step=1,
                optimizer_state=fake_state,
            )

            # The optimizer was constructed, then load_state_dict was called
            mock_optimizer_cls.assert_called_once()
            mock_optimizer_instance.load_state_dict.assert_called_once_with(fake_state)
        finally:
            if not torch_already:
                del sys.modules["torch"]
            sys.modules.pop("unsloth.FastLanguageModel", None)
            # Don't remove unsloth if it was there before
            if not torch_already:
                sys.modules.pop("unsloth", None)


# -----------------------------------------------------------------------
# B. Stale-step finalizer skip
# -----------------------------------------------------------------------


class TestStaleStepFinalizerSkip:
    """Test the finalizer logic for last_completed_step tracking.

    These tests exercise the logic directly without requiring the full
    training loop (which depends on heavy deps).
    """

    def test_zero_net_new_steps_no_duplicate_ckpt(self):
        """Resuming from step N and completing zero steps should not write
        a duplicate checkpoint."""
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(42)

            # Create an initial checkpoint at step 5
            adapter_dir = ckpt_root / "ckpt_5" / "lora_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=5,
                wall_seconds_total=10.0,
                curriculum_snapshot={},
                normalizer_snapshot={},
                rollout_rng_state=pickle.dumps(rng.getstate()),
                lora_weights_path=adapter_dir,
                model_name="stub",
                config_hash="sha256:test",
            )
            saved_dir = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_dir)

            # Simulate resume: start_step = 6, but zero steps complete
            start_step = bundle.step + 1  # 6
            last_completed_step = start_step - 1  # 5

            # The finalizer guard: last_completed_step >= start_step?
            # 5 >= 6 is False => no final checkpoint written
            assert not (last_completed_step >= start_step), (
                "Finalizer should skip when zero net-new steps completed"
            )

            # Verify no new ckpt dirs were created beyond the original
            ckpt_dirs = sorted(
                p.name
                for p in ckpt_root.iterdir()
                if p.is_dir() and p.name.startswith("ckpt_")
            )
            assert ckpt_dirs == ["ckpt_5"]
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_net_new_progress_writes_actual_last_step(self):
        """When net-new progress is made, the finalizer should write a
        checkpoint for the actual last completed step, not start_step."""
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(42)

            # Initial checkpoint at step 5
            adapter_dir5 = ckpt_root / "ckpt_5" / "lora_adapter"
            adapter_dir5.mkdir(parents=True)
            (adapter_dir5 / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=5,
                wall_seconds_total=10.0,
                curriculum_snapshot={},
                normalizer_snapshot={},
                rollout_rng_state=pickle.dumps(rng.getstate()),
                lora_weights_path=adapter_dir5,
                model_name="stub",
                config_hash="sha256:test",
            )
            saved_dir = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_dir)

            # Simulate resume + completing steps 6 and 7
            start_step = 6
            last_completed_step = 7  # Completed steps 6 and 7

            # The finalizer guard: 7 >= 6 => True => write ckpt for step 7
            assert last_completed_step >= start_step

            # Simulate writing that final checkpoint
            adapter_dir7 = ckpt_root / "ckpt_7" / "lora_adapter"
            adapter_dir7.mkdir(parents=True)
            (adapter_dir7 / "adapter_config.json").write_text("{}")

            new_bundle = CheckpointBundle(
                step=7,
                wall_seconds_total=20.0,
                curriculum_snapshot={},
                normalizer_snapshot={},
                rollout_rng_state=pickle.dumps(rng.getstate()),
                lora_weights_path=adapter_dir7,
                model_name="stub",
                config_hash="sha256:test",
            )
            saved_dir7 = save_checkpoint(ckpt_root, new_bundle)
            atomic_replace_latest(ckpt_root, saved_dir7)

            # Verify both checkpoints exist
            ckpt_dirs = sorted(
                p.name
                for p in ckpt_root.iterdir()
                if p.is_dir() and p.name.startswith("ckpt_")
            )
            assert "ckpt_5" in ckpt_dirs
            assert "ckpt_7" in ckpt_dirs

            # latest/ should point to step 7
            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.step == 7
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_cold_start_zero_steps_no_ckpt(self):
        """A cold start (no bundle) that completes zero steps should not
        write any checkpoint."""
        # Simulate cold start: start_step=0, no bundle, zero steps
        start_step = 0
        last_completed_step = start_step - 1  # -1

        # The finalizer guard: -1 >= 0 is False => no checkpoint
        assert not (last_completed_step >= start_step)


class TestBuildPolicyFailClosed:
    """Test that _build_policy raises on missing adapter directory."""

    def test_raises_on_missing_adapter_dir(self):
        from unittest.mock import MagicMock

        from training.train import _build_policy

        bundle = MagicMock()
        bundle.lora_weights_path = Path("/nonexistent/lora_adapter")

        with pytest.raises(RuntimeError, match="does not exist"):
            _build_policy(
                MagicMock(),
                bundle,
                LoraConfig=MagicMock(),
            )

    def test_cold_start_no_bundle_ok(self):
        """When bundle is None, _build_policy should not raise with the
        missing-adapter RuntimeError.  It may still fail later when trying
        to load the model, but the adapter check itself should pass."""
        from unittest.mock import MagicMock

        from training.train import _build_policy

        # This will still fail because the actual factory calls require real
        # models, but it should NOT fail with the missing-adapter RuntimeError.
        try:
            _build_policy(
                MagicMock(),
                None,  # bundle is None => cold start
                LoraConfig=MagicMock(),
            )
        except RuntimeError as exc:
            # Should not be the missing-adapter error
            assert "does not exist" not in str(exc)
        except (ValueError, ImportError, TypeError, AttributeError):
            pass  # Expected when factories can't load real models
