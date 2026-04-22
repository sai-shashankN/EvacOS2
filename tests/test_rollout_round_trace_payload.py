import json
import re
import shutil
import uuid
from pathlib import Path

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.schemas.multi_agent import ActionTypeMA
from training.policy_adapter import StubPolicy
from training.rollout import collect_episode


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_log_dir() -> Path:
    root = Path(".phase20_test_tmp") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _extract_episode_id(prompt: list[dict[str, str]]) -> str:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r'"episode_id"\s*:\s*"([^"]+)"', system_msg["content"])
    return match.group(1) if match else "ep_test"


def _extract_round_id(prompt: list[dict[str, str]]) -> int:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r"Round:\s*(\d+)", system_msg["content"])
    return int(match.group(1)) if match else 0


class DirectivePolicy:
    def act(self, prompt: list[dict[str, str]], agent_id: str, role: str) -> str:
        episode_id = _extract_episode_id(prompt)
        round_id = _extract_round_id(prompt)
        if role == "orchestrator" and round_id == 0:
            return json.dumps(
                {
                    "episode_id": episode_id,
                    "round_id": round_id,
                    "agent_id": "orchestrator",
                    "action_id": "orch_dir_round0",
                    "action_type": "broadcast_directive",
                    "arguments": {
                        "directive": {
                            "directive_id": "dir_round0",
                            "target": "all",
                            "directive_type": "evacuation_priority",
                            "params": {"priority": "high"},
                            "priority": "high",
                            "issued_round": round_id,
                            "ttl_rounds": 2,
                            "human_readable_note": "prioritize evacuation",
                        }
                    },
                }
            )
        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": agent_id,
                "action_id": f"{agent_id}_{round_id}",
                "action_type": "wait",
                "arguments": {},
            }
        )


def test_round_trace_emits_dashboard_payload():
    log_dir = _make_log_dir()
    try:
        env = EvacEnvironment()
        result = collect_episode(
            env,
            StubPolicy(seed=7),
            seed=11,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=2,
            jsonl_dir=log_dir,
        )

        rows = _read_jsonl(log_dir / "round_trace.jsonl")
        state = env.get_internal_state(result.episode_id)
        expected_floor_keys = {f"floor_{floor.floor_id}" for floor in state.building.floors}
        expected_agent_ids = {"orchestrator", *state.last_floor_reward_breakdowns.keys()}

        assert len(rows) == result.num_rounds == 2
        for row in rows:
            assert set(row["per_floor_civilians"]) == expected_floor_keys
            assert set(row["per_floor_hazard_severity"]) == expected_floor_keys
            assert sum(row["per_floor_civilians"].values()) > 0
            assert any(value > 0.0 for value in row["per_floor_hazard_severity"].values())
            assert set(row["reward_ticker"]) == expected_agent_ids
            assert "directive_feed" in row and isinstance(row["directive_feed"], list)
            assert "override_feed" in row and isinstance(row["override_feed"], list)
            assert row["orchestrator_action_type"] in {
                ActionTypeMA.wait.value,
                ActionTypeMA.broadcast_directive.value,
                ActionTypeMA.override_floor_agent.value,
            }
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def test_round_trace_directive_override_feeds_populate_when_emitted():
    log_dir = _make_log_dir()
    try:
        env = EvacEnvironment()
        collect_episode(
            env,
            DirectivePolicy(),
            seed=13,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=2,
            jsonl_dir=log_dir,
        )

        rows = _read_jsonl(log_dir / "round_trace.jsonl")
        assert len(rows) == 2
        assert rows[0]["directive_feed"] == [
            {
                "agent_id": "orchestrator",
                "action_id": "orch_dir_round0",
                "action_type": "broadcast_directive",
                "arguments": {
                    "directive": {
                        "directive_id": "dir_round0",
                        "target": "all",
                        "directive_type": "evacuation_priority",
                        "params": {"priority": "high"},
                        "priority": "high",
                        "issued_round": 0,
                        "ttl_rounds": 2,
                        "human_readable_note": "prioritize evacuation",
                    }
                },
                "round_id": 0,
            }
        ]
        assert rows[0]["override_feed"] == []
        assert rows[1]["directive_feed"] == []
        assert rows[1]["override_feed"] == []
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)
