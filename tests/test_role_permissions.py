"""Tests for role permission enforcement (Phase 5)."""

from __future__ import annotations

import pytest

from evacos_ma.permissions import (
    FLOOR_AGENT_ACTIONS,
    ORCHESTRATOR_ACTIONS,
    ValidationResult,
    action_mask_for_role,
    validate_action_for_role,
)
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentRole,
)

# ---------------------------------------------------------------------------
# 1. Coverage: every ActionTypeMA in at least one role set; wait in both.
# ---------------------------------------------------------------------------


class TestActionCoverage:
    def test_wait_in_both_sets(self):
        assert ActionTypeMA.wait in FLOOR_AGENT_ACTIONS
        assert ActionTypeMA.wait in ORCHESTRATOR_ACTIONS

    def test_every_action_type_in_at_least_one_set(self):
        all_classified = FLOOR_AGENT_ACTIONS | ORCHESTRATOR_ACTIONS
        for at in ActionTypeMA:
            assert at in all_classified, f"{at} is not in any role set"

    def test_floor_and_orchestrator_sets_intersect_only_at_wait(self):
        intersection = FLOOR_AGENT_ACTIONS & ORCHESTRATOR_ACTIONS
        assert intersection == {ActionTypeMA.wait}


# ---------------------------------------------------------------------------
# 2. Role-forbidden rejections
# ---------------------------------------------------------------------------


def _make_envelope(action_type: ActionTypeMA, agent_id: str = "test_agent") -> ActionEnvelopeMA:
    return ActionEnvelopeMA(
        episode_id="ep_test",
        round_id=0,
        agent_id=agent_id,
        action_id=f"act_{action_type.value}",
        action_type=action_type,
        arguments={},
    )


class TestRoleForbidden:
    def test_floor_agent_submitting_override_is_rejected(self):
        action = _make_envelope(ActionTypeMA.override_floor_agent)
        result = validate_action_for_role(action, AgentRole.floor_agent)
        assert not result.valid
        assert result.reason == "role_forbidden"

    def test_floor_agent_submitting_broadcast_directive_is_rejected(self):
        action = _make_envelope(ActionTypeMA.broadcast_directive)
        result = validate_action_for_role(action, AgentRole.floor_agent)
        assert not result.valid
        assert result.reason == "role_forbidden"

    def test_orchestrator_submitting_route_within_floor_is_rejected(self):
        action = _make_envelope(ActionTypeMA.route_within_floor)
        result = validate_action_for_role(action, AgentRole.orchestrator)
        assert not result.valid
        assert result.reason == "role_forbidden"

    def test_orchestrator_submitting_scout_is_rejected(self):
        action = _make_envelope(ActionTypeMA.scout)
        result = validate_action_for_role(action, AgentRole.orchestrator)
        assert not result.valid
        assert result.reason == "role_forbidden"

    def test_floor_agent_submitting_route_within_floor_is_accepted(self):
        action = _make_envelope(ActionTypeMA.route_within_floor)
        result = validate_action_for_role(action, AgentRole.floor_agent)
        assert result.valid
        assert result.reason == "ok"

    def test_orchestrator_submitting_override_is_accepted(self):
        action = _make_envelope(ActionTypeMA.override_floor_agent)
        result = validate_action_for_role(action, AgentRole.orchestrator)
        assert result.valid
        assert result.reason == "ok"

    def test_wait_accepted_for_both_roles(self):
        action = _make_envelope(ActionTypeMA.wait)
        assert validate_action_for_role(action, AgentRole.floor_agent).valid
        assert validate_action_for_role(action, AgentRole.orchestrator).valid


# ---------------------------------------------------------------------------
# 3. action_mask correctness
# ---------------------------------------------------------------------------


class TestActionMask:
    def test_floor_agent_mask_contains_exactly_floor_actions(self):
        mask = action_mask_for_role(AgentRole.floor_agent)
        expected = sorted(a.value for a in FLOOR_AGENT_ACTIONS)
        assert sorted(mask) == expected

    def test_orchestrator_mask_contains_exactly_orchestrator_actions(self):
        mask = action_mask_for_role(AgentRole.orchestrator)
        expected = sorted(a.value for a in ORCHESTRATOR_ACTIONS)
        assert sorted(mask) == expected

    def test_floor_mask_does_not_contain_override(self):
        mask = action_mask_for_role(AgentRole.floor_agent)
        assert ActionTypeMA.override_floor_agent.value not in mask

    def test_orchestrator_mask_does_not_contain_scout(self):
        mask = action_mask_for_role(AgentRole.orchestrator)
        assert ActionTypeMA.scout.value not in mask

    def test_both_masks_contain_wait(self):
        assert ActionTypeMA.wait.value in action_mask_for_role(AgentRole.floor_agent)
        assert ActionTypeMA.wait.value in action_mask_for_role(AgentRole.orchestrator)


# ---------------------------------------------------------------------------
# 4. Integration: action_mask in built observation
# ---------------------------------------------------------------------------


class TestActionMaskInObservation:
    def test_floor_agent_observation_action_mask_non_empty(self):
        from evacos_ma.env import EvacEnvironment

        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)
        for agent_id, floor_obs in obs.floors.items():
            assert len(floor_obs.action_mask) > 0, f"Floor {agent_id} has empty action_mask"
            # Must not contain orchestrator-only actions
            assert "override_floor_agent" not in floor_obs.action_mask
            assert "broadcast_directive" not in floor_obs.action_mask

    def test_orchestrator_observation_action_mask_non_empty(self):
        from evacos_ma.env import EvacEnvironment

        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)
        orch_mask = obs.orchestrator.action_mask
        assert len(orch_mask) > 0
        # Must not contain floor-only actions
        assert "route_within_floor" not in orch_mask
        assert "scout" not in orch_mask
        # Must contain orchestrator actions
        assert "override_floor_agent" in orch_mask
        assert "broadcast_directive" in orch_mask
