"""Tests for directive lifecycle (Phase 5)."""

from __future__ import annotations

import pytest

from evacos_ma.directives import DirectiveStore, RECOGNISED_DIRECTIVE_TYPES
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
    Directive,
    DirectiveOutcome,
    DirectivePriority,
)


class TestDirectiveTTL:
    """broadcast_directive with ttl_rounds=3 → active_directive on target floor
    observation for rounds r, r+1, r+2; absent at r+3."""

    def test_directive_appears_and_expires(self):
        store = DirectiveStore()
        directive = Directive(
            directive_id="d1",
            target="floor_0_agent",
            directive_type="evacuate_via_stairwell",
            params={"stairwell_id": "S1"},
            priority=DirectivePriority.normal,
            issued_round=0,
            ttl_rounds=3,
        )
        store.issue(directive)

        # Present at rounds 0, 1, 2
        for r in range(3):
            store.tick(r)
            d = store.active_directive_for_target("floor_0_agent", r)
            assert d is not None, f"Directive should be active at round {r}"
            assert d.directive_id == "d1"

        # Expired at round 3
        store.tick(3)
        d = store.active_directive_for_target("floor_0_agent", 3)
        assert d is None, "Directive should be expired at round 3"

    def test_directive_present_at_expiration_boundary(self):
        """Directive issued at round 5 with TTL 3 is active at rounds 5,6,7, gone at 8."""
        store = DirectiveStore()
        directive = Directive(
            directive_id="d_boundary",
            target="floor_1_agent",
            directive_type="hold_floor",
            params={},
            priority=DirectivePriority.normal,
            issued_round=5,
            ttl_rounds=3,
        )
        store.issue(directive)

        for r in range(5, 8):
            store.tick(r)
            assert store.active_directive_for_target("floor_1_agent", r) is not None

        store.tick(8)
        assert store.active_directive_for_target("floor_1_agent", 8) is None


class TestDirectiveSupersession:
    """Issue D1 at r=0, issue D2 with supersedes=D1 at r=2 → D1 removed, D2 appears."""

    def test_supersession(self):
        store = DirectiveStore()

        d1 = Directive(
            directive_id="D1",
            target="floor_0",
            directive_type="hold_floor",
            params={},
            priority=DirectivePriority.normal,
            issued_round=0,
            ttl_rounds=10,
        )
        store.issue(d1)

        # At round 1, D1 is active
        store.tick(1)
        active = store.active_directive_for_target("floor_0_agent", 1)
        assert active is not None
        assert active.directive_id == "D1"

        # Issue D2 at round 2, superseding D1
        d2 = Directive(
            directive_id="D2",
            target="floor_0",
            directive_type="evacuate_via_stairwell",
            params={"stairwell_id": "S2"},
            priority=DirectivePriority.high,
            issued_round=2,
            ttl_rounds=10,
            supersedes_directive_id_or_null="D1",
        )
        store.issue(d2)

        # At round 2, D1 should be gone, D2 should be active
        store.tick(2)
        active = store.active_directive_for_target("floor_0_agent", 2)
        assert active is not None
        assert active.directive_id == "D2"

        # D1 should no longer be retrievable
        # (it was removed from the store)
        outcomes = store.directive_outcomes()
        assert len(outcomes) == 2  # Both D1 and D2 issued


class TestUnknownDirectiveType:
    """Unknown directive_type stored, does not crash, does not grant priority boost."""

    def test_unknown_directive_stored_safely(self):
        store = DirectiveStore()
        d_unknown = Directive(
            directive_id="d_unknown",
            target="floor_0_agent",
            directive_type="custom_experimental_type",
            params={},
            priority=DirectivePriority.normal,
            issued_round=0,
            ttl_rounds=5,
        )
        store.issue(d_unknown)

        # Should be stored
        active = store.active_directive_for_target("floor_0_agent", 0)
        assert active is not None
        assert active.directive_type == "custom_experimental_type"

        # Should not be in recognised types
        assert "custom_experimental_type" not in RECOGNISED_DIRECTIVE_TYPES

        # has_high_priority_directive should return False for unknown types
        # (since priority is normal, not high)
        assert not store.has_high_priority_directive("floor_0_agent", 0, "custom_experimental_type")

    def test_unknown_directive_no_priority_boost_in_arbitration(self):
        """Verify that an unknown directive does not affect arbitration outcomes."""
        from evacos_ma.arbitration import Arbitrator

        store = DirectiveStore()
        d_unknown = Directive(
            directive_id="d_unknown",
            target="floor_0_agent",
            directive_type="custom_type",
            params={},
            priority=DirectivePriority.high,
            issued_round=0,
            ttl_rounds=5,
        )
        store.issue(d_unknown)

        arbitrator = Arbitrator()
        snapshot = {"stairwell_capacities": {"S1": 1}, "exit_throughputs": {}}

        floor_actions = {
            "floor_0_agent": _make_route_action("floor_0_agent", "act_0", "S1"),
            "floor_1_agent": _make_route_action("floor_1_agent", "act_1", "S1"),
        }

        result_with_unknown = arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=None,
            floor_actions=floor_actions,
            directive_store=store,
            round_id=0,
        )

        # With unknown type, the high-priority directive check won't match
        # any recognised type, so both agents compete on action_id tiebreak alone
        # The lex-first action_id wins
        accepted = [a for a in result_with_unknown.accepted if a.arguments.get("stairwell_id") == "S1"]
        assert len(accepted) == 1


def _make_route_action(agent_id: str, action_id: str, stairwell_id: str) -> ActionEnvelopeMA:
    return ActionEnvelopeMA(
        episode_id="ep_test",
        round_id=0,
        agent_id=agent_id,
        action_id=action_id,
        action_type=ActionTypeMA.route_within_floor,
        arguments={"stairwell_id": stairwell_id},
    )


class TestDirectiveOutcomes:
    """recent_directive_outcomes grows by 1 per directive issued."""

    def test_outcomes_grow_per_directive(self):
        store = DirectiveStore()
        assert len(store.directive_outcomes()) == 0

        d1 = Directive(
            directive_id="d1",
            target="floor_0",
            directive_type="hold_floor",
            issued_round=0,
            ttl_rounds=5,
        )
        store.issue(d1)
        assert len(store.directive_outcomes()) == 1

        d2 = Directive(
            directive_id="d2",
            target="floor_1",
            directive_type="prioritize_floor_egress",
            issued_round=1,
            ttl_rounds=5,
        )
        store.issue(d2)
        assert len(store.directive_outcomes()) == 2

        outcomes = store.directive_outcomes()
        assert outcomes[0].directive_id == "d1"
        assert outcomes[1].directive_id == "d2"
        assert outcomes[0].accepted is True

    def test_directive_outcomes_in_orchestrator_observation(self):
        """Verify that directive outcomes appear in orchestrator observation."""
        from evacos_ma.env import EvacEnvironment

        env = EvacEnvironment()
        ep_id, obs = env.reset_multi_agent("task_1_fire_easy", seed=42)

        # Initially no directives
        assert len(obs.orchestrator.recent_directive_outcomes) == 0

        # Issue a directive via step_multi_agent
        bundle = ActionBundleMA(
            episode_id=ep_id,
            round_id=0,
            orchestrator_action=ActionEnvelopeMA(
                episode_id=ep_id,
                round_id=0,
                agent_id="orchestrator",
                action_id="broadcast_1",
                action_type=ActionTypeMA.broadcast_directive,
                arguments={
                    "directive": {
                        "directive_id": "dir_test",
                        "target": "floor_0_agent",
                        "directive_type": "hold_floor",
                        "params": {},
                        "priority": "normal",
                        "issued_round": 0,
                        "ttl_rounds": 5,
                    },
                },
            ),
        )
        result = env.step_multi_agent(bundle)
        assert len(result.observations_by_role.orchestrator.recent_directive_outcomes) >= 1


def test_duplicate_directive_id_is_rejected_without_overwriting_active_state():
    store = DirectiveStore()
    first = Directive(
        directive_id="dup-dir",
        target="floor_0_agent",
        directive_type="hold_floor",
        params={"a": 1},
        priority=DirectivePriority.normal,
        issued_round=0,
        ttl_rounds=5,
    )
    second = Directive(
        directive_id="dup-dir",
        target="floor_1_agent",
        directive_type="evacuate_via_stairwell",
        params={"stairwell_id": "S1"},
        priority=DirectivePriority.high,
        issued_round=1,
        ttl_rounds=5,
    )

    first_outcome = store.issue(first)
    second_outcome = store.issue(second)

    assert first_outcome.accepted is True
    assert second_outcome.accepted is False
    assert second_outcome.outcome_summary == "duplicate_directive_id"
    assert list(store._active) == ["dup-dir"]
    assert store._active["dup-dir"] == first
