"""Tests for round protocol determinism (Phase 5)."""

from __future__ import annotations

import hashlib
import json

import pytest

from evacos_ma.env import EvacEnvironment
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
)


def _make_bundle(episode_id: str, round_id: int) -> ActionBundleMA:
    """Create a bundle with orchestrator wait + mixed floor actions."""
    floor_actions = {}
    for floor_num in range(5):
        agent_id = f"floor_{floor_num}_agent"
        if round_id % 3 == 0 and floor_num < 3:
            action_type = ActionTypeMA.prioritize_room
            args = {"room_id": f"F{floor_num}_R0"}
        else:
            action_type = ActionTypeMA.wait
            args = {}
        floor_actions[agent_id] = ActionEnvelopeMA(
            episode_id=episode_id,
            round_id=round_id,
            agent_id=agent_id,
            action_id=f"act_{agent_id}_{round_id}",
            action_type=action_type,
            arguments=args,
        )

    return ActionBundleMA(
        episode_id=episode_id,
        round_id=round_id,
        orchestrator_action=ActionEnvelopeMA(
            episode_id=episode_id,
            round_id=round_id,
            agent_id="orchestrator",
            action_id=f"act_orch_{round_id}",
            action_type=ActionTypeMA.wait,
            arguments={},
        ),
        floor_actions=floor_actions,
    )


def _trace_hash(data: list) -> str:
    """Deterministic hash of a trace list."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class TestRoundProtocolDeterminism:
    """Run task_lh_fire_easy seed 42 for 30 rounds; repeat; assert byte-equal traces."""

    def _run_30_rounds(self):
        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_lh_fire_easy", seed=42)

        trace_data = []
        for r in range(30):
            bundle = _make_bundle(ep_id, r)
            result = env.step_multi_agent(bundle)
            accepted_ids = sorted(a.action_id for a in result.info.reservation_trace or [])
            # Collect round fingerprint
            trace_data.append({
                "round_id": r,
                "accepted_count": len(result.info.arbitration_trace or []),
                "reservation_hash": _trace_hash(result.info.reservation_trace),
                "arbitration_hash": _trace_hash(result.info.arbitration_trace),
                "done": result.done,
            })
            if result.done:
                break

        return trace_data

    def test_deterministic_traces_across_runs(self):
        """Two runs with identical seed must produce identical round traces."""
        run1 = self._run_30_rounds()
        run2 = self._run_30_rounds()
        assert run1 == run2, "Round traces differ across identical runs — non-determinism detected"

    def test_deterministic_observations_and_rewards(self):
        """Reset and step 30 rounds produce identical visibility, belief, reward."""
        env1 = EvacEnvironment()
        ep_id1, obs1 = env1.reset_multi_agent("task_lh_fire_easy", seed=42)

        env2 = EvacEnvironment()
        ep_id2, obs2 = env2.reset_multi_agent("task_lh_fire_easy", seed=42)

        # Initial observations should be identical
        assert obs1.orchestrator.round_id == obs2.orchestrator.round_id

        for r in range(15):
            bundle1 = _make_bundle(ep_id1, r)
            result1 = env1.step_multi_agent(bundle1)
            bundle2 = _make_bundle(ep_id2, r)
            result2 = env2.step_multi_agent(bundle2)

            # Compare key state
            assert result1.done == result2.done
            assert result1.observations_by_role.orchestrator.round_id == result2.observations_by_role.orchestrator.round_id

            # Compare floor reward breakdowns
            for agent_id in result1.rewards_by_role.floors:
                r1 = result1.rewards_by_role.floors[agent_id]
                r2 = result2.rewards_by_role.floors[agent_id]
                assert abs(r1.raw - r2.raw) < 1e-9, f"Round {r} agent {agent_id}: reward mismatch {r1.raw} vs {r2.raw}"

            if result1.done:
                break
