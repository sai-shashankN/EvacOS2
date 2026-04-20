"""Tests for checkpoint save/load/rotate.

Heavy-dep-free.
"""

import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from training.checkpoint import (
    CheckpointBundle,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)


def _tmp_dir():
    """Create a temp dir avoiding pytest tmp_path permission issues on Windows."""
    d = os.path.join(tempfile.gettempdir(), f"evacos_ckpt_test_{os.getpid()}_{random.randint(0,99999)}")
    os.makedirs(d, exist_ok=True)
    return Path(d)


class TestCheckpointRoundtrip:
    def test_save_and_load_roundtrip(self):
        """save_checkpoint then load_checkpoint returns an equal CheckpointBundle."""
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(42)
            rng_state = pickle.dumps(rng.getstate())

            bundle = CheckpointBundle(
                step=10,
                wall_seconds_total=123.45,
                curriculum_snapshot={"current_tier": {"fire": "easy"}, "stats": {}, "total_outcomes": 5},
                normalizer_snapshot={"orchestrator:easy": {"count": 5, "mean": 0.3, "m2": 0.1}},
                rollout_rng_state=rng_state,
                lora_weights_path=ckpt_root / "lora_weights",
                model_name="stub",
                config_hash="sha256:abcdef123456",
            )

            saved_path = save_checkpoint(ckpt_root, bundle)
            assert saved_path.exists()

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.step == bundle.step
            assert loaded.wall_seconds_total == bundle.wall_seconds_total
            assert loaded.curriculum_snapshot == bundle.curriculum_snapshot
            assert loaded.normalizer_snapshot == bundle.normalizer_snapshot
            assert loaded.rollout_rng_state == bundle.rollout_rng_state
            assert loaded.model_name == bundle.model_name
            assert loaded.config_hash == bundle.config_hash
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)


class TestRotateCheckpoints:
    def test_rotate_keeps_highest_n(self):
        """rotate_checkpoints(keep_last_n=3) on a dir with 5 checkpoints
        leaves exactly 3 and keeps the highest-numbered."""
        ckpt_root = _tmp_dir()
        try:
            rng_state = pickle.dumps(random.Random().getstate())

            for step in [1, 2, 3, 4, 5]:
                bundle = CheckpointBundle(
                    step=step,
                    wall_seconds_total=float(step * 10),
                    curriculum_snapshot={},
                    normalizer_snapshot={},
                    rollout_rng_state=rng_state,
                    lora_weights_path=ckpt_root / f"lora_{step}",
                    model_name="stub",
                    config_hash="sha256:test",
                )
                save_checkpoint(ckpt_root, bundle, extras={"step": step})

            # Remove the 'latest' copy to avoid confusion
            latest = ckpt_root / "latest"
            if latest.exists():
                shutil.rmtree(latest)

            # Now we have 5 ckpt dirs
            ckpt_dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("ckpt_")]
            assert len(ckpt_dirs) == 5

            # Rotate to keep 3
            rotate_checkpoints(ckpt_root, keep_last_n=3)

            # Should be exactly 3 remaining
            ckpt_dirs = sorted(
                [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("ckpt_")],
                key=lambda p: p.name,
            )
            assert len(ckpt_dirs) == 3
            # Should keep highest: ckpt_3, ckpt_4, ckpt_5
            assert ckpt_dirs[0].name == "ckpt_3"
            assert ckpt_dirs[1].name == "ckpt_4"
            assert ckpt_dirs[2].name == "ckpt_5"
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)
