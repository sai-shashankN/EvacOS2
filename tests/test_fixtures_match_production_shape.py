import json
import re
import shutil
import uuid
from pathlib import Path

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from training.rollout import collect_episode

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_KNOWN_PATHS_NOT_HIT_IN_SMOKE = {
    # Emitted in evacos_ma/env.py when rejected_actions include stairwell/elevator capacity contention.
    "coordination_bonus",  # Populated in evacos_ma/env.py:764.
    # Emitted in evacos_ma/env.py when a floor action is rejected/invalid during round execution.
    "floor_invalid_action",  # Populated in evacos_ma/env.py:757.
}


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


def _walk_reward_dicts(payload: object) -> list[dict[str, float]]:
    found: list[dict[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"last_reward_breakdown", "breakdown"} and isinstance(value, dict):
                found.append(value)
            found.extend(_walk_reward_dicts(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_walk_reward_dicts(item))
    return found


def _fixture_reward_keys() -> set[str]:
    fixture_keys: set[str] = set()
    for fixture_name in (
        "orchestrator_observation.golden.json",
        "floor_observation.golden.json",
        "step_result.golden.json",
    ):
        payload = json.loads((FIXTURES_DIR / fixture_name).read_text(encoding="utf-8"))
        for reward_dict in _walk_reward_dicts(payload):
            for key, value in reward_dict.items():
                if isinstance(value, (int, float)) and value != 0.0:
                    fixture_keys.add(key)
    return fixture_keys


def _extract_episode_id(prompt: list[dict[str, str]]) -> str:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r'"episode_id"\s*:\s*"([^"]+)"', system_msg["content"])
    return match.group(1) if match else "ep_test"


def _extract_round_id(prompt: list[dict[str, str]]) -> int:
    system_msg = next(msg for msg in prompt if msg["role"] == "system")
    match = re.search(r"Round:\s*(\d+)", system_msg["content"])
    return int(match.group(1)) if match else 0


class RewardCoveragePolicy:
    """Drive a 3-round smoke through directive, scout, prediction, and invalid-action paths.

    The smoke intentionally covers the fixture-backed non-zero keys that are easy to reach
    in a short rollout. `coordination_bonus` remains in `_KNOWN_PATHS_NOT_HIT_IN_SMOKE`
    because it requires stairwell/elevator capacity contention and is brittle to force
    deterministically in three rounds.
    """

    def act(self, prompt: list[dict[str, str]], agent_id: str, role: str) -> str:
        episode_id = _extract_episode_id(prompt)
        round_id = _extract_round_id(prompt)

        if role == "orchestrator":
            if round_id == 0:
                return json.dumps(
                    {
                        "episode_id": episode_id,
                        "round_id": round_id,
                        "agent_id": "orchestrator",
                        "action_id": "orch_directive",
                        "action_type": "broadcast_directive",
                        "arguments": {
                            "directive": {
                                "directive_id": "dir_fixture_contract",
                                "target": "floor_0",
                                "directive_type": "prioritize_room",
                                "params": {"room_id": "F0_R0"},
                                "priority": "high",
                                "issued_round": round_id,
                                "ttl_rounds": 2,
                                "human_readable_note": "prioritize room F0_R0",
                            }
                        },
                    }
                )
            return json.dumps(
                {
                    "episode_id": episode_id,
                    "round_id": round_id,
                    "agent_id": "orchestrator",
                    "action_id": f"orch_wait_{round_id}",
                    "action_type": "wait",
                    "arguments": {},
                }
            )

        floor_num = int(agent_id.split("_")[1])
        if round_id == 0 and floor_num == 0:
            payload = {
                "action_type": "scout",
                "arguments": {"target_room_id": "F0_R0"},
            }
        elif round_id == 0 and floor_num == 1:
            payload = {
                "action_type": "predict_state",
                "arguments": {
                    "belief": {
                        "belief_id": "belief_fixture_contract",
                        "predictor_agent_id": agent_id,
                        "target_entity_ids": ["F1_R0"],
                        "horizon": 1,
                        "prediction_payload": {"expected_civilians_in_room": 1},
                        "confidence": 0.8,
                        "justification": "fixture smoke",
                        "created_round": round_id,
                        "resolved_round_or_null": None,
                    }
                },
            }
        elif round_id == 0 and floor_num == 2:
            payload = {
                "action_type": "open_exit",
                "arguments": {"exit_id": "missing_exit"},
            }
        else:
            payload = {
                "action_type": "route_within_floor",
                "arguments": {
                    "from_room_id": f"F{floor_num}_R0",
                    "to_room_id": f"F{floor_num}_R1",
                },
            }

        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": agent_id,
                "action_id": f"{agent_id}_{round_id}",
                **payload,
            }
        )


def test_fixture_reward_keys_are_production_emitted():
    fixture_keys = _fixture_reward_keys()

    log_dir = _make_log_dir()
    try:
        env = EvacEnvironment()
        collect_episode(
            env,
            RewardCoveragePolicy(),
            seed=17,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=3,
            jsonl_dir=log_dir,
        )
        production_keys: set[str] = set()
        for row in _read_jsonl(log_dir / "reward_trace.jsonl"):
            breakdown = row.get("breakdown", {})
            if not isinstance(breakdown, dict):
                continue
            for key, value in breakdown.items():
                if isinstance(value, (int, float)) and value != 0.0:
                    production_keys.add(key)
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)

    missing = sorted(fixture_keys - (production_keys | _KNOWN_PATHS_NOT_HIT_IN_SMOKE))
    assert not missing, (
        "Fixture reward keys not emitted by production smoke rollout: "
        f"{missing}. Inspect evacos_ma/env.py reward_pipeline wiring or evacos_ma/reward_pipeline.py."
    )
