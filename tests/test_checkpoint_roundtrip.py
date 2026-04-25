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
    acquire_run_output_lock,
    atomic_replace_latest,
    CheckpointBundle,
    _pid_is_running,
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
            lora_dir = ckpt_root / "ckpt_10" / "lora_adapter"
            lora_dir.mkdir(parents=True)
            (lora_dir / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=10,
                wall_seconds_total=123.45,
                curriculum_snapshot={"current_tier": {"fire": "easy"}, "stats": {}, "total_outcomes": 5},
                normalizer_snapshot={"orchestrator:easy": {"count": 5, "mean": 0.3, "m2": 0.1}},
                rollout_rng_state=rng_state,
                lora_weights_path=lora_dir,
                model_name="stub",
                config_hash="sha256:abcdef123456",
                config_path="training/config.remote-unsloth-3b-fire-floor-specialist-750.yaml",
                max_steps=750,
                rollout_max_rounds_per_episode=4,
                rollout_disaster_families=["fire"],
                rollout_tier_schedule=[{"steps": 750, "mix": {"easy": 750}}],
            )

            saved_path = save_checkpoint(ckpt_root, bundle)
            assert saved_path.exists()
            atomic_replace_latest(ckpt_root, saved_path)

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.step == bundle.step
            assert loaded.wall_seconds_total == bundle.wall_seconds_total
            assert loaded.curriculum_snapshot == bundle.curriculum_snapshot
            assert loaded.normalizer_snapshot == bundle.normalizer_snapshot
            assert loaded.rollout_rng_state == bundle.rollout_rng_state
            assert loaded.model_name == bundle.model_name
            assert loaded.config_hash == bundle.config_hash
            assert loaded.config_path == bundle.config_path
            assert loaded.max_steps == 750
            assert loaded.rollout_max_rounds_per_episode == 4
            assert loaded.rollout_disaster_families == ["fire"]
            assert loaded.rollout_tier_schedule == [{"steps": 750, "mix": {"easy": 750}}]
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_save_and_load_roundtrip_with_role_artifacts(self):
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(7)
            rng_state = pickle.dumps(rng.getstate())
            lora_dir = ckpt_root / "ckpt_3" / "lora_adapter"
            orch_dir = lora_dir / "orchestrator"
            floor_dir = lora_dir / "floor_agent"
            orch_dir.mkdir(parents=True)
            floor_dir.mkdir(parents=True)
            (orch_dir / "adapter_config.json").write_text("{}")
            (floor_dir / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=3,
                wall_seconds_total=11.5,
                curriculum_snapshot={"tier": "medium"},
                normalizer_snapshot={"k": {"count": 1, "mean": 0.1, "m2": 0.0}},
                rollout_rng_state=rng_state,
                lora_weights_path=lora_dir,
                model_name="orchestrator=big;floor_agent=small",
                config_hash="sha256:split123456",
                role_lora_weights_paths={
                    "orchestrator": orch_dir,
                    "floor_agent": floor_dir,
                },
                role_model_names={
                    "orchestrator": "big-model",
                    "floor_agent": "small-model",
                },
                role_optimizer_states={
                    "orchestrator": {"state": {0: {"step": 1}}},
                    "floor_agent": {"state": {1: {"step": 2}}},
                },
            )

            saved_path = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_path)

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.role_lora_weights_paths is not None
            assert loaded.role_model_names == bundle.role_model_names
            assert loaded.role_optimizer_states is not None
            assert loaded.role_lora_weights_paths["orchestrator"] == orch_dir
            assert loaded.role_lora_weights_paths["floor_agent"] == floor_dir
            assert loaded.role_optimizer_states["orchestrator"]["state"][0]["step"] == 1
            assert loaded.role_optimizer_states["floor_agent"]["state"][1]["step"] == 2
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_save_and_load_roundtrip_with_floor_only_role_artifacts(self):
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(13)
            rng_state = pickle.dumps(rng.getstate())
            lora_dir = ckpt_root / "ckpt_4" / "lora_adapter"
            floor_dir = lora_dir / "floor_agent"
            floor_dir.mkdir(parents=True)
            (floor_dir / "adapter_config.json").write_text("{}")

            bundle = CheckpointBundle(
                step=4,
                wall_seconds_total=7.5,
                curriculum_snapshot={"tier": "easy"},
                normalizer_snapshot={"k": {"count": 2, "mean": 0.2, "m2": 0.1}},
                rollout_rng_state=rng_state,
                lora_weights_path=lora_dir,
                model_name="Qwen/Qwen2.5-3B-Instruct",
                config_hash="sha256:flooronly123",
                role_lora_weights_paths={"floor_agent": floor_dir},
                role_model_names={
                    "orchestrator": "Qwen/Qwen2.5-3B-Instruct",
                    "floor_agent": "Qwen/Qwen2.5-3B-Instruct",
                },
                orchestrator_policy="stub",
                role_optimizer_states={"floor_agent": {"state": {1: {"step": 9}}}},
            )

            saved_path = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_path)

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.role_lora_weights_paths == {"floor_agent": floor_dir}
            assert loaded.role_model_names == {
                "orchestrator": "Qwen/Qwen2.5-3B-Instruct",
                "floor_agent": "Qwen/Qwen2.5-3B-Instruct",
            }
            assert loaded.orchestrator_policy == "stub"
            assert loaded.role_optimizer_states == {"floor_agent": {"state": {1: {"step": 9}}}}
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_save_and_load_roundtrip_with_floor_specialist_artifacts(self):
        ckpt_root = _tmp_dir()
        try:
            rng = random.Random(17)
            rng_state = pickle.dumps(rng.getstate())
            lora_dir = ckpt_root / "ckpt_5" / "lora_adapter"
            orch_dir = lora_dir / "orchestrator"
            fire_dir = lora_dir / "floor_agent" / "specialists" / "fire"
            flood_dir = lora_dir / "floor_agent" / "specialists" / "flood"
            gas_dir = lora_dir / "floor_agent" / "specialists" / "gas"
            for path in (orch_dir, fire_dir, flood_dir, gas_dir):
                path.mkdir(parents=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")

            bundle = CheckpointBundle(
                step=5,
                wall_seconds_total=21.0,
                curriculum_snapshot={"tier": "hard"},
                normalizer_snapshot={"k": {"count": 3, "mean": 0.4, "m2": 0.2}},
                rollout_rng_state=rng_state,
                lora_weights_path=lora_dir,
                model_name="orchestrator=big;floor_agent=small",
                config_hash="sha256:routed123456",
                role_lora_weights_paths={"orchestrator": orch_dir},
                floor_specialist_lora_weights_paths={
                    "fire": fire_dir,
                    "flood": flood_dir,
                    "gas": gas_dir,
                },
                role_model_names={
                    "orchestrator": "big-model",
                    "floor_agent": "small-model",
                },
                role_optimizer_states={"orchestrator": {"state": {0: {"step": 5}}}},
            )

            saved_path = save_checkpoint(ckpt_root, bundle)
            atomic_replace_latest(ckpt_root, saved_path)

            loaded = load_checkpoint(ckpt_root)
            assert loaded is not None
            assert loaded.role_lora_weights_paths == {"orchestrator": orch_dir}
            assert loaded.floor_specialist_lora_weights_paths == {
                "fire": fire_dir,
                "flood": flood_dir,
                "gas": gas_dir,
            }
            assert loaded.role_optimizer_states == {"orchestrator": {"state": {0: {"step": 5}}}}
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_load_rebases_copied_latest_floor_specialist_paths(self):
        tmp_root = _tmp_dir()
        src_root = tmp_root / "source" / "checkpoints"
        portable_root = tmp_root / "portable" / "checkpoints"
        try:
            rng = random.Random(19)
            rng_state = pickle.dumps(rng.getstate())
            lora_dir = src_root / "ckpt_6" / "lora_adapter"
            orch_dir = lora_dir / "orchestrator"
            fire_dir = lora_dir / "floor_agent" / "specialists" / "fire"
            flood_dir = lora_dir / "floor_agent" / "specialists" / "flood"
            gas_dir = lora_dir / "floor_agent" / "specialists" / "gas"
            for path in (orch_dir, fire_dir, flood_dir, gas_dir):
                path.mkdir(parents=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")

            bundle = CheckpointBundle(
                step=6,
                wall_seconds_total=22.0,
                curriculum_snapshot={"tier": "brutal"},
                normalizer_snapshot={"k": {"count": 4, "mean": 0.5, "m2": 0.3}},
                rollout_rng_state=rng_state,
                lora_weights_path=lora_dir,
                model_name="orchestrator=big;floor_agent=small",
                config_hash="sha256:routedcopy123",
                role_lora_weights_paths={"orchestrator": orch_dir},
                floor_specialist_lora_weights_paths={
                    "fire": fire_dir,
                    "flood": flood_dir,
                    "gas": gas_dir,
                },
                role_model_names={
                    "orchestrator": "big-model",
                    "floor_agent": "small-model",
                },
            )

            saved_path = save_checkpoint(src_root, bundle)
            atomic_replace_latest(src_root, saved_path)
            portable_root.mkdir(parents=True)
            shutil.copytree(src_root / "latest", portable_root / "latest")
            shutil.rmtree(src_root)

            loaded = load_checkpoint(portable_root)
            assert loaded is not None
            expected_adapter_root = portable_root / "latest" / "lora_adapter"
            assert loaded.lora_weights_path == expected_adapter_root
            assert loaded.lora_weights_path.exists()
            assert loaded.role_lora_weights_paths == {
                "orchestrator": expected_adapter_root / "orchestrator",
            }
            assert loaded.floor_specialist_lora_weights_paths == {
                "fire": expected_adapter_root / "floor_agent" / "specialists" / "fire",
                "flood": expected_adapter_root / "floor_agent" / "specialists" / "flood",
                "gas": expected_adapter_root / "floor_agent" / "specialists" / "gas",
            }
            assert all(path.exists() for path in loaded.role_lora_weights_paths.values())
            assert all(path.exists() for path in loaded.floor_specialist_lora_weights_paths.values())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


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


class TestRunOutputLock:
    def test_pid_is_running_checks_current_process_without_signaling(self):
        assert _pid_is_running(os.getpid()) is True

    def test_acquire_run_output_lock_rejects_live_lock(self):
        ckpt_root = _tmp_dir()
        try:
            metrics_path = ckpt_root / "metrics.csv"
            jsonl_dir = ckpt_root / "logs"
            lock = acquire_run_output_lock(ckpt_root, metrics_path, jsonl_dir)
            with pytest.raises(RuntimeError, match="already locked"):
                acquire_run_output_lock(ckpt_root, metrics_path, jsonl_dir)
            lock.release()
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)

    def test_acquire_run_output_lock_cleans_stale_lock(self, monkeypatch):
        ckpt_root = _tmp_dir()
        try:
            metrics_path = ckpt_root / "metrics.csv"
            jsonl_dir = ckpt_root / "logs"
            lock = acquire_run_output_lock(ckpt_root, metrics_path, jsonl_dir)
            lock_path = lock.lock_paths[0]
            lock.release()
            lock_path.write_text(
                json.dumps({"pid": 999999, "target_path": str(ckpt_root)}),
                encoding="utf-8",
            )

            monkeypatch.setattr("training.checkpoint._pid_is_running", lambda pid: False)
            refreshed = acquire_run_output_lock(ckpt_root, metrics_path, jsonl_dir)

            assert all(path.exists() for path in refreshed.lock_paths)
            refreshed.release()
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)


class TestAtomicReplaceLatest:
    def test_atomic_replace_latest_cleans_tmp_on_fallback_failure(self, monkeypatch):
        ckpt_root = _tmp_dir()
        try:
            ckpt_dir = ckpt_root / "ckpt_1"
            latest_dir = ckpt_root / "latest"
            tmp_dir = ckpt_root / "latest.tmp"
            ckpt_dir.mkdir(parents=True)
            latest_dir.mkdir(parents=True)
            (ckpt_dir / "meta.json").write_text("{}", encoding="utf-8")
            (latest_dir / "old.txt").write_text("old", encoding="utf-8")

            real_copytree = shutil.copytree

            def failing_copytree(src, dst, *args, **kwargs):
                result = real_copytree(src, dst, *args, **kwargs)
                if Path(src) == tmp_dir and Path(dst) == latest_dir:
                    raise RuntimeError("publish failed")
                return result

            monkeypatch.setattr("os.replace", lambda src, dst: (_ for _ in ()).throw(OSError("rename failed")))
            monkeypatch.setattr(shutil, "copytree", failing_copytree)

            with pytest.raises(RuntimeError, match="publish failed"):
                atomic_replace_latest(ckpt_root, ckpt_dir)

            assert not tmp_dir.exists()
            assert not latest_dir.exists()
        finally:
            shutil.rmtree(ckpt_root, ignore_errors=True)
