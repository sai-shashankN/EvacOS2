"""Tests for policy adapter: StubPolicy and parse_completion_to_action.

Heavy-dep-free.
"""

import json

import pytest

from training.policy_adapter import StubPolicy, parse_completion_to_action


class TestStubPolicy:
    def test_stub_emits_valid_json_orchestrator_and_floor(self):
        """StubPolicy emits valid-JSON completions for orchestrator + floor_0_agent across 5 seeds."""
        for seed in range(5):
            policy = StubPolicy(seed=seed)
            # Orchestrator prompt (minimal)
            orch_prompt = [
                {"role": "system", "content": "Round: 0\nepisode_id: ep1"},
                {"role": "user", "content": "Floor summaries: []\nBeliefs: total=0"},
            ]
            orch_completion = policy.act(orch_prompt, "orchestrator", "orchestrator")
            orch_obj = json.loads(orch_completion)
            assert "action_type" in orch_obj
            assert orch_obj["agent_id"] == "orchestrator"

            # Floor prompt (minimal)
            floor_prompt = [
                {"role": "system", "content": "Floor: floor_0\nRound: 0\nepisode_id: ep1"},
                {"role": "user", "content": 'Rooms: []\nExits: [{"exit_id":"E1","blocked":false}]'},
            ]
            floor_completion = policy.act(floor_prompt, "floor_0_agent", "floor_agent")
            floor_obj = json.loads(floor_completion)
            assert "action_type" in floor_obj
            assert floor_obj["agent_id"] == "floor_0_agent"


class TestParseCompletionToAction:
    def test_invalid_json_returns_none_invalid_json(self):
        """parse_completion_to_action returns (None, 'invalid_json') on garbage text."""
        action, reason = parse_completion_to_action("garbage text!!!", "orchestrator", "orchestrator")
        assert action is None
        assert reason == "invalid_json"

    def test_role_forbidden_floor_broadcast_directive(self):
        """parse_completion_to_action returns (None, 'role_forbidden') when a floor agent
        emits broadcast_directive."""
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "broadcast_directive",
            "arguments": {
                "directive": {
                    "directive_id": "d1",
                    "target": "floor_0",
                    "directive_type": "evacuation_priority",
                    "params": {},
                }
            },
        })
        action, reason = parse_completion_to_action(completion, "floor_0_agent", "floor_agent")
        assert action is None
        assert reason == "role_forbidden"
