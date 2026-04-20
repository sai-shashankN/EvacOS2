"""Checkpoint save/load/rotate utilities.

Heavy-dependency-free.  LoRA weight saving is delegated to the training loop.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckpointBundle:
    step: int
    wall_seconds_total: float
    curriculum_snapshot: dict
    normalizer_snapshot: dict
    rollout_rng_state: bytes  # pickled random.Random state
    lora_weights_path: Path  # pointer to the adapter weights dir
    model_name: str
    config_hash: str  # sha256 of the resolved TrainingConfig JSON


def save_checkpoint(
    root: Path,
    bundle: CheckpointBundle,
    extras: dict | None = None,
) -> Path:
    """Save a checkpoint bundle to ``root/ckpt_<step>/`` and update ``root/latest/``."""
    ckpt_dir = root / f"ckpt_{bundle.step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "step": bundle.step,
        "wall_seconds_total": bundle.wall_seconds_total,
        "curriculum_snapshot": bundle.curriculum_snapshot,
        "normalizer_snapshot": bundle.normalizer_snapshot,
        "model_name": bundle.model_name,
        "config_hash": bundle.config_hash,
        "lora_weights_path": str(bundle.lora_weights_path),
    }
    if extras:
        meta["extras"] = extras

    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(ckpt_dir / "rollout_rng_state.pkl", "wb") as f:
        pickle.dump(bundle.rollout_rng_state, f)

    # Update latest symlink/copy
    latest_dir = root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir, ignore_errors=True)
    shutil.copytree(ckpt_dir, latest_dir)

    return ckpt_dir


def load_checkpoint(root: Path) -> CheckpointBundle | None:
    """Load the latest checkpoint from ``root/latest/``."""
    latest_dir = root / "latest"
    if not latest_dir.exists():
        return None

    meta_path = latest_dir / "meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    rng_path = latest_dir / "rollout_rng_state.pkl"
    rng_state = b""
    if rng_path.exists():
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)

    return CheckpointBundle(
        step=meta["step"],
        wall_seconds_total=meta["wall_seconds_total"],
        curriculum_snapshot=meta["curriculum_snapshot"],
        normalizer_snapshot=meta["normalizer_snapshot"],
        rollout_rng_state=rng_state,
        lora_weights_path=Path(meta["lora_weights_path"]),
        model_name=meta["model_name"],
        config_hash=meta["config_hash"],
    )


def rotate_checkpoints(root: Path, keep_last_n: int) -> None:
    """Remove oldest checkpoints, keeping only the *keep_last_n* highest-numbered."""
    if not root.exists():
        return

    ckpt_dirs = []
    for p in root.iterdir():
        if p.is_dir() and re.match(r"ckpt_(\d+)", p.name):
            m = re.match(r"ckpt_(\d+)", p.name)
            if m:
                ckpt_dirs.append((int(m.group(1)), p))

    if len(ckpt_dirs) <= keep_last_n:
        return

    # Sort by step number ascending
    ckpt_dirs.sort(key=lambda x: x[0])

    # Remove oldest, keep last N
    to_remove = ckpt_dirs[:-keep_last_n]
    for _, d in to_remove:
        shutil.rmtree(d, ignore_errors=True)
