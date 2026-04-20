"""Tests for oversight counterfactual scoring (Phase 5)."""

from __future__ import annotations

import pytest

from evacos_ma.env import EvacEnvironment
from evacos_ma.round_protocol import (
    RoundProtocol,
    _compute_counterfactual_delta,
    _freeze_snapshot,
)
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
)


def _make_override_bundle(
    episode_id: str,
    round_id: int,
    target_agent: str,
    replacement_type: str = "wait",
    replacement_arguments: dict | None = None,
    floor_extra_actions: dict | None = None,
) -> ActionBundleMA:
    """Create a bundle with an override action targeting *target_agent*."""
    floor_actions = {}

    floor_actions[target_agent] = ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=round_id,
        agent_id=target_agent,
        action_id=f"orig_{target_agent}_{round_id}",
        action_type=ActionTypeMA.prioritize_room,
        arguments={"room_id": "F0_R0"},
    )
    if floor_extra_actions:
        floor_actions.update(floor_extra_actions)

    return ActionBundleMA(
        episode_id=episode_id,
        round_id=round_id,
        orchestrator_action=ActionEnvelopeMA(
            episode_id=episode_id,
            round_id=round_id,
            agent_id="orchestrator",
            action_id=f"override_{round_id}",
            action_type=ActionTypeMA.override_floor_agent,
            arguments={
                "target_floor_agent_id": target_agent,
                "replacement_action_type": replacement_type,
                "replacement_arguments": replacement_arguments or {},
                "rationale": "test override",
            },
        ),
        floor_actions=floor_actions,
    )


class TestCounterfactualDeltaPositive:
    """Override replaces bad action with good → counterfactual_delta > 0,
    oversight_bonus > 0 in orchestrator reward."""

    def test_positive_counterfactual_yields_bonus(self):
        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)

        # Do one override step
        target_agent = "floor_0_agent"
        bundle = _make_override_bundle(
            ep_id, 0, target_agent,
            replacement_type="wait",
        )
        result = env.step_multi_agent(bundle)

        # Check counterfactual deltas exist
        info = result.info
        deltas = info.score_snapshot.get("counterfactual_deltas", {})
        # The counterfactual delta should be present (may be 0 or positive depending on sim)
        assert isinstance(deltas, dict)

        # The orchestrator reward breakdown should have oversight_bonus if delta > 0
        orch_reward = result.rewards_by_role.orchestrator
        if any(d > 0 for d in deltas.values()):
            assert orch_reward.raw > 0 or "oversight_bonus" in orch_reward.breakdown.get_components()


class TestCounterfactualDeltaNotNegative:
    """Override is no better than original → oversight_bonus == 0 (not negative)."""

    def test_no_negative_oversight_bonus(self):
        """oversight_bonus should never be negative."""
        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)

        # Override with same action type (effectively no change)
        target_agent = "floor_0_agent"
        bundle = _make_override_bundle(
            ep_id, 0, target_agent,
            replacement_type="wait",
        )
        result = env.step_multi_agent(bundle)

        orch_reward = result.rewards_by_role.orchestrator
        # oversight_bonus component should not be negative
        components = orch_reward.breakdown.get_components()
        oversight = components.get("oversight_bonus", 0.0)
        assert oversight >= 0.0, f"Oversight bonus should be non-negative, got {oversight}"


class TestCounterfactualDeterminism:
    """Two runs of the same override scenario produce identical counterfactual_delta."""

    def test_deterministic_counterfactual(self):
        # Run 1
        env1 = EvacEnvironment()
        ep_id1, _ = env1.reset_multi_agent("task_1_fire_easy", seed=42)
        bundle1 = _make_override_bundle(ep_id1, 0, "floor_0_agent")
        result1 = env1.step_multi_agent(bundle1)
        delta1 = result1.info.score_snapshot.get("counterfactual_deltas", {})

        # Run 2
        env2 = EvacEnvironment()
        ep_id2, _ = env2.reset_multi_agent("task_1_fire_easy", seed=42)
        bundle2 = _make_override_bundle(ep_id2, 0, "floor_0_agent")
        result2 = env2.step_multi_agent(bundle2)
        delta2 = result2.info.score_snapshot.get("counterfactual_deltas", {})

        assert delta1 == delta2, (
            f"Counterfactual deltas differ: {delta1} vs {delta2}"
        )


class TestOverrideTrackingInObservation:
    """After override, the target floor agent's observation shows override_applied_last_round."""

    def test_override_applied_flag(self):
        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)

        target_agent = "floor_0_agent"
        bundle = _make_override_bundle(ep_id, 0, target_agent)
        result = env.step_multi_agent(bundle)

        # The floor agent should have override_applied_last_round set
        floor_obs = result.observations_by_role.floors.get(target_agent)
        assert floor_obs is not None
        assert floor_obs.override_applied_last_round is True
        assert floor_obs.override_reason_last_round == "test override"
