"""Checkpoint save/load/rotate utilities.

Heavy-dependency-free.  LoRA weight saving is delegated to the training loop.
torch imports are function-local so the module imports without torch installed.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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
    role_lora_weights_paths: dict[str, Path] | None = None
    role_model_names: dict[str, str] | None = None
    orchestrator_policy: str | None = None
    # Phase 12 fields — all default to None for backward compatibility
    optimizer_state: dict | None = None
    role_optimizer_states: dict[str, dict] | None = None
    torch_rng_state: bytes | None = None  # pickled torch RNG state
    torch_cuda_rng_state: bytes | None = None  # pickled torch CUDA RNG state
    wandb_run_id: str | None = None


@dataclass
class RunLockHandle:
    lock_paths: list[Path] = field(default_factory=list)

    def release(self) -> None:
        for lock_path in reversed(self.lock_paths):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("Failed to remove run lock %s", lock_path)
        self.lock_paths.clear()


def _lock_root_for_targets(target_paths: list[Path]) -> Path:
    import os

    common = Path(os.path.commonpath([str(path) for path in target_paths]))
    return common / ".run_locks"


def _lock_file_name(target_path: Path) -> str:
    import hashlib

    digest = hashlib.sha256(str(target_path).encode("utf-8")).hexdigest()[:16]
    return f"{digest}.lock"


def _pid_is_running(pid: int) -> bool:
    import os

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_output_lock(*target_paths: Path) -> RunLockHandle:
    """Acquire per-target path locks for training outputs.

    Lock files are keyed by absolute target path so overlapping checkpoint,
    metrics, or JSONL destinations conflict even if the broader config differs.
    Stale locks from dead processes are removed automatically.
    """

    import atexit
    import os
    import time

    normalized_targets = [
        Path(path).resolve(strict=False)
        for path in target_paths
    ]
    lock_root = _lock_root_for_targets(normalized_targets)
    lock_root.mkdir(parents=True, exist_ok=True)

    handle = RunLockHandle()
    pid = os.getpid()
    for target_path in normalized_targets:
        lock_path = lock_root / _lock_file_name(target_path)
        payload = {
            "pid": pid,
            "target_path": str(target_path),
            "created_at": time.time(),
        }
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                try:
                    existing_pid = int(existing.get("pid", -1))
                except (TypeError, ValueError):
                    existing_pid = -1
                if _pid_is_running(existing_pid):
                    handle.release()
                    raise RuntimeError(
                        f"Output path is already locked by pid {existing_pid}: {target_path}"
                    )
                logger.warning(
                    "Removing stale run lock %s for dead pid %s",
                    lock_path,
                    existing_pid,
                )
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    continue
                continue

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                try:
                    Path(lock_path).unlink()
                except OSError:
                    pass
                handle.release()
                raise
            handle.lock_paths.append(lock_path)
            break

    atexit.register(handle.release)
    return handle


def save_checkpoint(
    root: Path,
    bundle: CheckpointBundle,
    extras: dict | None = None,
) -> Path:
    """Save a checkpoint bundle to ``root/ckpt_<step>/``.

    NOTE: This no longer updates ``root/latest/``.  Call
    :func:`atomic_replace_latest` separately after confirming all artifacts
    (adapter weights, optimizer state, RNG pickles) are durable.
    """
    ckpt_dir = root / f"ckpt_{bundle.step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "step": bundle.step,
        "wall_seconds_total": bundle.wall_seconds_total,
        "curriculum_snapshot": bundle.curriculum_snapshot,
        "normalizer_snapshot": bundle.normalizer_snapshot,
        "model_name": bundle.model_name,
        "config_hash": bundle.config_hash,
        "lora_weights_path": str(bundle.lora_weights_path),
    }
    if bundle.role_lora_weights_paths is not None:
        meta["role_lora_weights_paths"] = {
            role: str(path) for role, path in bundle.role_lora_weights_paths.items()
        }
    if bundle.role_model_names is not None:
        meta["role_model_names"] = dict(bundle.role_model_names)
    if bundle.orchestrator_policy is not None:
        meta["orchestrator_policy"] = bundle.orchestrator_policy
    if bundle.wandb_run_id is not None:
        meta["wandb_run_id"] = bundle.wandb_run_id
    if extras:
        meta["extras"] = extras

    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(ckpt_dir / "rollout_rng_state.pkl", "wb") as f:
        pickle.dump(bundle.rollout_rng_state, f)

    # Phase 12: optimizer state
    if bundle.optimizer_state is not None:
        try:
            import torch

            torch.save(bundle.optimizer_state, ckpt_dir / "optimizer_state.pt")
        except ImportError:
            logger.debug(
                "torch not importable \u2014 falling back to pickle for optimizer_state"
            )
            with open(ckpt_dir / "optimizer_state.pkl", "wb") as f:
                pickle.dump(bundle.optimizer_state, f)
    if bundle.role_optimizer_states is not None:
        try:
            import torch

            torch.save(bundle.role_optimizer_states, ckpt_dir / "role_optimizer_states.pt")
        except ImportError:
            logger.debug(
                "torch not importable — falling back to pickle for role_optimizer_states"
            )
            with open(ckpt_dir / "role_optimizer_states.pkl", "wb") as f:
                pickle.dump(bundle.role_optimizer_states, f)

    # Phase 12: torch RNG state
    if bundle.torch_rng_state is not None:
        with open(ckpt_dir / "torch_rng_state.pkl", "wb") as f:
            pickle.dump(bundle.torch_rng_state, f)

    # Phase 12: torch CUDA RNG state
    if bundle.torch_cuda_rng_state is not None:
        with open(ckpt_dir / "torch_cuda_rng_state.pkl", "wb") as f:
            pickle.dump(bundle.torch_cuda_rng_state, f)

    return ckpt_dir


def atomic_replace_latest(root: Path, ckpt_dir: Path) -> None:
    """Atomically publish *ckpt_dir* as ``root/latest/``.

    Writes to ``latest.tmp/`` first, then renames.  Best-effort atomicity
    (POSIX rename is atomic; Windows is best-effort).
    """
    latest_dir = root / "latest"
    tmp_dir = root / "latest.tmp"

    # Clean up any leftover temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        import os

        shutil.copytree(ckpt_dir, tmp_dir)

        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)

        try:
            os.replace(str(tmp_dir), str(latest_dir))
            return
        except OSError:
            if latest_dir.exists():
                shutil.rmtree(latest_dir, ignore_errors=True)
            try:
                shutil.copytree(tmp_dir, latest_dir)
            except Exception:
                if latest_dir.exists():
                    shutil.rmtree(latest_dir, ignore_errors=True)
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_checkpoint(root: Path) -> CheckpointBundle | None:
    """Load the latest checkpoint from ``root/latest/``.

    Backward-compatible: missing Phase-12 fields default to ``None`` with a
    one-time warning per field.
    """
    latest_dir = root / "latest"

    # If latest/ is missing or broken, try to find the highest valid ckpt_N/
    if not latest_dir.exists() or not (latest_dir / "meta.json").exists():
        found = _find_highest_valid_ckpt(root)
        if found is not None:
            logger.warning(
                "latest/ is missing or corrupt; falling back to %s", found.name
            )
            latest_dir = found
        else:
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

    # Phase 12: optimizer state (backward compat)
    optimizer_state: dict | None = None
    opt_path = latest_dir / "optimizer_state.pt"
    opt_pickle_path = latest_dir / "optimizer_state.pkl"
    if opt_path.exists():
        try:
            import torch

            optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=False)
        except ImportError:
            logger.warning(
                "torch not importable — optimizer_state.pt ignored"
            )
    elif opt_pickle_path.exists():
        with open(opt_pickle_path, "rb") as f:
            optimizer_state = pickle.load(f)
    else:
        logger.debug("optimizer_state.pt not found in checkpoint; starting fresh")

    role_optimizer_states: dict[str, dict] | None = None
    role_opt_path = latest_dir / "role_optimizer_states.pt"
    role_opt_pickle_path = latest_dir / "role_optimizer_states.pkl"
    if role_opt_path.exists():
        try:
            import torch

            role_optimizer_states = torch.load(role_opt_path, map_location="cpu", weights_only=False)
        except ImportError:
            logger.warning(
                "torch not importable — role_optimizer_states.pt ignored"
            )
    elif role_opt_pickle_path.exists():
        with open(role_opt_pickle_path, "rb") as f:
            role_optimizer_states = pickle.load(f)

    # Phase 12: torch RNG state
    torch_rng_state: bytes | None = None
    torch_rng_path = latest_dir / "torch_rng_state.pkl"
    if torch_rng_path.exists():
        with open(torch_rng_path, "rb") as f:
            torch_rng_state = pickle.load(f)
    else:
        logger.debug("torch_rng_state.pkl not found in checkpoint")

    # Phase 12: torch CUDA RNG state
    torch_cuda_rng_state: bytes | None = None
    cuda_rng_path = latest_dir / "torch_cuda_rng_state.pkl"
    if cuda_rng_path.exists():
        with open(cuda_rng_path, "rb") as f:
            torch_cuda_rng_state = pickle.load(f)
    else:
        logger.debug("torch_cuda_rng_state.pkl not found in checkpoint")

    # Phase 12: wandb run id
    wandb_run_id: str | None = meta.get("wandb_run_id", None)
    if "wandb_run_id" not in meta:
        logger.debug("wandb_run_id not found in checkpoint meta.json")

    return CheckpointBundle(
        step=meta["step"],
        wall_seconds_total=meta["wall_seconds_total"],
        curriculum_snapshot=meta["curriculum_snapshot"],
        normalizer_snapshot=meta["normalizer_snapshot"],
        rollout_rng_state=rng_state,
        lora_weights_path=Path(meta["lora_weights_path"]),
        model_name=meta["model_name"],
        config_hash=meta["config_hash"],
        role_lora_weights_paths={
            role: Path(path)
            for role, path in meta.get("role_lora_weights_paths", {}).items()
        }
        or None,
        role_model_names=meta.get("role_model_names"),
        orchestrator_policy=meta.get("orchestrator_policy"),
        optimizer_state=optimizer_state,
        role_optimizer_states=role_optimizer_states,
        torch_rng_state=torch_rng_state,
        torch_cuda_rng_state=torch_cuda_rng_state,
        wandb_run_id=wandb_run_id,
    )


def _find_highest_valid_ckpt(root: Path) -> Path | None:
    """Walk *root* for the highest-numbered ``ckpt_N/`` that has a valid
    ``meta.json`` and an existing ``lora_weights_path`` directory."""
    candidates: list[tuple[int, Path]] = []
    if not root.exists():
        return None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"ckpt_(\d+)", p.name)
        if not m:
            continue
        meta_path = p / "meta.json"
        if not meta_path.exists():
            continue
        # Check that the adapter directory referenced in meta.json exists
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            lora_path = Path(meta.get("lora_weights_path", ""))
            if lora_path.exists():
                candidates.append((int(m.group(1)), p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def rotate_checkpoints(root: Path, keep_last_n: int) -> None:
    """Remove oldest checkpoints, keeping only the *keep_last_n* highest-numbered."""
    if not root.exists():
        return

    ckpt_dirs: list[tuple[int, Path]] = []
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
