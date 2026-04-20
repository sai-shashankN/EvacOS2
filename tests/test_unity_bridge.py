import json
from pathlib import Path

from renderer.unity_bridge import build_headless_render, build_unity_asset_descriptor


def _trajectory() -> list[dict]:
    return [
        {
            "round_id": index,
            "per_floor_civilians": {f"floor_{floor}": max(0, 5 - index) for floor in range(5)},
            "per_floor_hazard_severity": {f"floor_{floor}": round(index * 0.1, 2) for floor in range(5)},
            "directive_feed": [],
            "override_feed": [],
            "reward_ticker": {"orchestrator": index * 0.1},
            "done": index == 4,
            "score_snapshot": {"saved": index},
        }
        for index in range(5)
    ]


def test_build_unity_asset_descriptor_is_deterministic(tmp_path: Path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(_trajectory(), sort_keys=True), encoding="utf-8")
    first = build_unity_asset_descriptor(trajectory_path)
    second = build_unity_asset_descriptor(trajectory_path)
    assert first.read_bytes() == second.read_bytes()


def test_build_headless_render_writes_valid_gif(tmp_path: Path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(json.dumps(_trajectory(), sort_keys=True), encoding="utf-8")
    output = build_headless_render(trajectory_path, tmp_path / "trajectory.gif")
    assert output.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
