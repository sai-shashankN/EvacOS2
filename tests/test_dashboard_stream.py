import json
from pathlib import Path

from dashboard.log_stream import list_episodes, tail_episode


def test_list_episodes_discovers_ids(tmp_path: Path):
    (tmp_path / "episode_summary.jsonl").write_text(
        json.dumps({"episode_id": "ep-1", "seed": 42, "tier": "easy", "disaster_family": "fire", "total_steps": 1}) + "\n",
        encoding="utf-8",
    )
    episodes = list_episodes(tmp_path)
    assert [episode.episode_id for episode in episodes] == ["ep-1"]


def test_tail_episode_yields_dashboard_contract(tmp_path: Path):
    (tmp_path / "episode_summary.jsonl").write_text(
        json.dumps({"episode_id": "ep-1", "seed": 42, "tier": "easy", "disaster_family": "fire", "total_steps": 2}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "round_trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_id": "ep-1",
                    "round_id": 0,
                    "per_floor_civilians": {"floor_0": 4},
                    "per_floor_hazard_severity": {"floor_0": 0.2},
                    "directive_feed": [{"directive_type": "hold_floor"}],
                    "override_feed": [{"action_type": "override_floor_agent"}],
                    "reward_ticker": {"orchestrator": 0.1},
                    "score_snapshot": {"saved": 1},
                }),
                json.dumps({
                    "episode_id": "ep-1",
                    "round_id": 1,
                    "per_floor_civilians": {"floor_0": 3},
                    "per_floor_hazard_severity": {"floor_0": 0.4},
                    "directive_feed": [],
                    "override_feed": [],
                    "reward_ticker": {"orchestrator": 0.2},
                    "score_snapshot": {"saved": 2},
                }),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "action_trace.jsonl").write_text("", encoding="utf-8")
    rows = list(tail_episode(tmp_path, "ep-1", follow=False))
    assert len(rows) == 2
    for row in rows:
        assert {
            "episode_id",
            "round_id",
            "per_floor_civilians",
            "per_floor_hazard_severity",
            "directive_feed",
            "override_feed",
            "reward_ticker",
            "done",
            "score_snapshot",
        }.issubset(row)
