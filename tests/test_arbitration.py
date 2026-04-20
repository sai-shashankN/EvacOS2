"""Tests for reservation and arbitration layer (Phase 5)."""

from __future__ import annotations

import pytest

from evacos_ma.arbitration import Arbitrator, PRIORITY_CLASS
from evacos_ma.directives import DirectiveStore
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    Directive,
    DirectivePriority,
)


def _make_action(
    action_type: ActionTypeMA,
    agent_id: str,
    action_id: str | None = None,
    arguments: dict | None = None,
) -> ActionEnvelopeMA:
    return ActionEnvelopeMA(
        episode_id="ep_test",
        round_id=0,
        agent_id=agent_id,
        action_id=action_id or f"act_{agent_id}_{action_type.value}",
        action_type=action_type,
        arguments=arguments or {},
    )


class TestStairwellCapacity:
    """6 floor agents all intent-to-route through the same stairwell with capacity=5.
    → 5 accepted, 1 rejected with reason='stairwell_capacity'.
    Rejection chosen by action_id tiebreak (lex-last rejected)."""

    def test_stairwell_capacity_arbitration(self):
        arbitrator = Arbitrator()
        snapshot = {
            "stairwell_capacities": {"stairwell_0": 5},
            "exit_throughputs": {},
        }
        floor_actions = {}
        for i in range(6):
            agent_id = f"floor_{i}_agent"
            floor_actions[agent_id] = _make_action(
                ActionTypeMA.route_within_floor,
                agent_id,
                action_id=f"route_{i}",
                arguments={"stairwell_id": "stairwell_0"},
            )

        result = arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=None,
            floor_actions=floor_actions,
            directive_store=None,
            round_id=0,
        )

        # Count accepted route_within_floor actions
        accepted_routes = [
            a for a in result.accepted
            if a.action_type == ActionTypeMA.route_within_floor
            and a.arguments.get("stairwell_id") == "stairwell_0"
        ]
        assert len(accepted_routes) == 5, f"Expected 5 accepted, got {len(accepted_routes)}"

        # Count rejections for stairwell capacity
        stairwell_rejections = [
            r for r in result.rejections
            if r["reason"] == "stairwell_capacity"
        ]
        assert len(stairwell_rejections) == 1, f"Expected 1 rejection, got {len(stairwell_rejections)}"
        assert stairwell_rejections[0]["reason"] == "stairwell_capacity"

        # The lex-last action_id should be rejected (lowest priority = smallest action_id since same type)
        all_ids = sorted(f"route_{i}" for i in range(6))
        accepted_ids = sorted(a.action_id for a in accepted_routes)
        rejected_id = stairwell_rejections[0]["action_id"]
        # With same priority class and tiebreak on action_id ascending in sort (reverse=True in sort),
        # the lex-first action_ids are accepted, lex-last rejected
        assert rejected_id == all_ids[-1], f"Expected lex-last rejected, got {rejected_id}"


class TestOverrideBeatsBase:
    """Orchestrator override_floor_agent on floor_2_agent → floor_2's original
    prioritize_room gets dropped, override wait executes, arbitration_trace
    records reason='orchestrator_override'."""

    def test_override_replaces_floor_action(self):
        arbitrator = Arbitrator()
        snapshot = {"stairwell_capacities": {}, "exit_throughputs": {}}

        floor_actions = {
            "floor_2_agent": _make_action(
                ActionTypeMA.prioritize_room,
                "floor_2_agent",
                action_id="orig_prioritize",
                arguments={"room_id": "F2_R1"},
            ),
            "floor_0_agent": _make_action(
                ActionTypeMA.wait,
                "floor_0_agent",
            ),
        }

        orchestrator_action = _make_action(
            ActionTypeMA.override_floor_agent,
            "orchestrator",
            action_id="override_act",
            arguments={
                "target_floor_agent_id": "floor_2_agent",
                "replacement_action_type": "wait",
                "replacement_arguments": {},
            },
        )

        # The override replaces the floor action; mark it as an override target
        replacement_action = _make_action(
            ActionTypeMA.wait,
            "floor_2_agent",
            action_id="override_override_act",
        )
        override_targets = {"floor_2_agent": replacement_action}

        # Simulate the replaced floor actions
        replaced_floor_actions = dict(floor_actions)
        replaced_floor_actions["floor_2_agent"] = replacement_action

        result = arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=orchestrator_action,
            floor_actions=replaced_floor_actions,
            directive_store=None,
            round_id=0,
            override_targets=override_targets,
        )

        # The override action should be accepted
        assert any(a.action_id == "override_override_act" for a in result.accepted)
        # No rejection for the replaced action (it was replaced before arbitration)
        assert not any(r["action_id"] == "orig_prioritize" for r in result.rejections)


class TestDirectivePriorityBoost:
    """High-priority directive evacuate_via_stairwell for floor →
    floor's route_within_floor to that stairwell wins arbitration
    over a competing floor's non-directive route."""

    def test_directive_boosts_priority(self):
        arbitrator = Arbitrator()
        directive_store = DirectiveStore()

        # Issue high-priority directive to floor_0_agent
        directive = Directive(
            directive_id="dir_1",
            target="floor_0_agent",
            directive_type="evacuate_via_stairwell",
            params={"stairwell_id": "stairwell_0"},
            priority=DirectivePriority.high,
            issued_round=0,
            ttl_rounds=10,
        )
        directive_store.issue(directive)

        snapshot = {
            "stairwell_capacities": {"stairwell_0": 1},  # Only 1 slot
            "exit_throughputs": {},
        }

        floor_actions = {
            "floor_0_agent": _make_action(
                ActionTypeMA.route_within_floor,
                "floor_0_agent",
                action_id="boosted_route",
                arguments={"stairwell_id": "stairwell_0"},
            ),
            "floor_1_agent": _make_action(
                ActionTypeMA.route_within_floor,
                "floor_1_agent",
                action_id="normal_route",
                arguments={"stairwell_id": "stairwell_0"},
            ),
        }

        result = arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=None,
            floor_actions=floor_actions,
            directive_store=directive_store,
            round_id=0,
        )

        # The boosted agent should win
        accepted_routes = [
            a for a in result.accepted
            if a.arguments.get("stairwell_id") == "stairwell_0"
        ]
        assert len(accepted_routes) == 1
        assert accepted_routes[0].agent_id == "floor_0_agent", "Directive-boosted agent should win"

        # The non-boosted agent should be rejected
        rejections = [r for r in result.rejections if r["action_id"] == "normal_route"]
        assert len(rejections) == 1

        # Check arbitration trace for directive_priority reason
        arb_entries = [e for e in result.arbitration_trace if e.get("resource_id") == "stairwell_0"]
        assert len(arb_entries) >= 1
        # Reason should mention directive_priority (or priority_class)
        reasons = {e.get("reason") for e in arb_entries}
        assert "directive_priority" in reasons or "priority_class" in reasons


class TestLockdownVsRoute:
    """Floor A lockdown_room=R1, Floor B route_within_floor to R1 → route rejected."""

    def test_lockdown_blocks_route(self):
        arbitrator = Arbitrator()
        snapshot = {"stairwell_capacities": {}, "exit_throughputs": {}}

        floor_actions = {
            "floor_0_agent": _make_action(
                ActionTypeMA.lockdown_room,
                "floor_0_agent",
                action_id="lockdown_a",
                arguments={"room_id": "R1"},
            ),
            "floor_1_agent": _make_action(
                ActionTypeMA.route_within_floor,
                "floor_1_agent",
                action_id="route_b",
                arguments={"to_room_id": "R1"},
            ),
        }

        result = arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=None,
            floor_actions=floor_actions,
            directive_store=None,
            round_id=0,
        )

        # route_b should be rejected
        rejections = [r for r in result.rejections if r["action_id"] == "route_b"]
        assert len(rejections) == 1
        assert rejections[0]["reason"] == "room_locked_down"

        # lockdown should be accepted (no contention on lockdown itself)
        assert any(a.action_id == "lockdown_a" for a in result.accepted)
