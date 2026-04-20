"""Offline trajectory-to-asset bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    matplotlib = None
    plt = None
    FuncAnimation = None
    PillowWriter = None


def _read_trajectory(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "frames" in payload:
        return list(payload["frames"])
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported trajectory payload")


def _normalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_id": int(frame.get("round_id", 0)),
        "per_floor_civilians": frame.get("per_floor_civilians") or {},
        "per_floor_hazard_severity": frame.get("per_floor_hazard_severity") or {},
        "directive_feed": frame.get("directive_feed") or [],
        "override_feed": frame.get("override_feed") or [],
        "reward_ticker": frame.get("reward_ticker") or {},
        "done": bool(frame.get("done", False)),
        "score_snapshot": frame.get("score_snapshot") or {},
    }


def build_unity_asset_descriptor(
    trajectory_path: Path,
    asset_config_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    frames = [_normalize_frame(frame) for frame in _read_trajectory(trajectory_path)]
    asset_config = {}
    if asset_config_path is not None and asset_config_path.exists():
        asset_config = json.loads(asset_config_path.read_text(encoding="utf-8"))
    trajectory_id = trajectory_path.stem.replace(".unity", "")
    payload = {
        "trajectory_id": trajectory_id,
        "asset_config": asset_config,
        "frames": frames,
    }
    destination = output_path or trajectory_path.with_name(f"{trajectory_id}.unity.json")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def build_headless_render(
    trajectory_path: Path,
    output_path: Path,
    fps: int = 4,
) -> Path:
    if plt is None or FuncAnimation is None or PillowWriter is None:
        raise RuntimeError("matplotlib is required for build_headless_render")
    frames = [_normalize_frame(frame) for frame in _read_trajectory(trajectory_path)]
    if not frames:
        raise ValueError("Trajectory is empty")

    fig, ax = plt.subplots(figsize=(5, 7))
    floors = [f"floor_{idx}" for idx in range(4, -1, -1)]

    def update(frame_index: int) -> None:
        ax.clear()
        frame = frames[frame_index]
        hazards = frame["per_floor_hazard_severity"]
        civilians = frame["per_floor_civilians"]
        colors = [float(hazards.get(floor, 0.0)) for floor in floors]
        ax.barh(floors, [1] * len(floors), color=plt.cm.inferno(colors), edgecolor="black")
        for idx, floor in enumerate(floors):
            ax.text(0.02, idx, f"{floor} civ={int(civilians.get(floor, 0))} haz={float(hazards.get(floor, 0.0)):.2f}", va="center", color="white")
        ax.set_xlim(0, 1)
        ax.set_title(f"Trajectory {trajectory_path.stem} round {frame['round_id']}")
        ax.set_xticks([])
        ax.set_facecolor("#0d1824")
        fig.patch.set_facecolor("#0d1824")

    animation = FuncAnimation(fig, update, frames=len(frames), interval=max(1000 // max(fps, 1), 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path
