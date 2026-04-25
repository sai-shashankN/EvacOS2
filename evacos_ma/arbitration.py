"""Reservation and arbitration layer for shared resources.

Phase 5: resolves contention on stairwells, elevators, exits, and room
lockdown vs routing conflicts using the BLUEPRINT priority ordering.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentId,
    Directive,
    DirectivePriority,
)
from evacos_ma.directives import DirectiveStore


# ---------------------------------------------------------------------------
# Priority class table (documented in FORK_NOTES)
# ---------------------------------------------------------------------------

PRIORITY_CLASS: dict[ActionTypeMA, int] = {
    ActionTypeMA.route_between_floors: 10,
    ActionTypeMA.call_elevator: 9,
    ActionTypeMA.evacuate_floor_priority: 8,
    ActionTypeMA.route_within_floor: 5,
    ActionTypeMA.open_exit: 4,
    ActionTypeMA.prioritize_room: 3,
    ActionTypeMA.lockdown_room: 3,
    ActionTypeMA.scout: 2,
    ActionTypeMA.predict_state: 1,
    ActionTypeMA.handoff_to_orchestrator: 1,
    ActionTypeMA.wait: 0,
    ActionTypeMA.broadcast_directive: 0,
    ActionTypeMA.override_floor_agent: 0,
    ActionTypeMA.request_explanation: 0,
}


def _route_target_if_known(action: ActionEnvelopeMA, explicit_key: str, known: dict[str, Any]) -> Optional[str]:
    explicit = action.arguments.get(explicit_key)
    if explicit:
        return str(explicit)
    to_room = action.arguments.get("to_room_id")
    return str(to_room) if to_room in known else None


# ---------------------------------------------------------------------------
# Data structures returned by arbitration
# ---------------------------------------------------------------------------

class ArbitrationResult:
    """Holds accepted actions, rejections, reservation trace, and arbitration trace."""

    __slots__ = (
        "accepted",
        "rejections",
        "reservation_trace",
        "arbitration_trace",
    )

    def __init__(
        self,
        accepted: list[ActionEnvelopeMA],
        rejections: list[dict[str, Any]],
        reservation_trace: list[dict[str, Any]],
        arbitration_trace: list[dict[str, Any]],
    ) -> None:
        self.accepted = accepted
        self.rejections = rejections
        self.reservation_trace = reservation_trace
        self.arbitration_trace = arbitration_trace


# ---------------------------------------------------------------------------
# Arbitrator
# ---------------------------------------------------------------------------

class Arbitrator:
    """Stateless arbitrator - takes a frozen snapshot + actions, returns results."""

    def arbitrate(
        self,
        snapshot: dict[str, Any],
        orchestrator_action: Optional[ActionEnvelopeMA],
        floor_actions: dict[AgentId, ActionEnvelopeMA],
        directive_store: Optional[DirectiveStore],
        round_id: int,
        override_targets: Optional[dict[str, ActionEnvelopeMA]] = None,
    ) -> ArbitrationResult:
        """Run the full arbitration pass."""
        override_targets = override_targets or {}
        all_actions: list[ActionEnvelopeMA] = list(floor_actions.values())
        if orchestrator_action is not None:
            all_actions.append(orchestrator_action)

        accepted: list[ActionEnvelopeMA] = []
        rejections: list[dict[str, Any]] = []
        reservation_trace: list[dict[str, Any]] = []
        arbitration_trace: list[dict[str, Any]] = []
        decided_ids: set[str] = set()

        # --- Phase 1: room lockdown conflicts ---
        lockdown_rooms: set[str] = set()
        for action in all_actions:
            if action.action_type == ActionTypeMA.lockdown_room:
                lockdown_rooms.add(action.arguments.get("room_id", ""))

        for action in all_actions:
            if action.action_type != ActionTypeMA.route_within_floor:
                continue
            to_room = action.arguments.get("to_room_id", "")
            if to_room not in lockdown_rooms:
                continue
            rejections.append(
                {
                    "agent_id": action.agent_id,
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "reason": "room_locked_down",
                    "resource_id": to_room,
                }
            )
            arbitration_trace.append(
                {
                    "round_id": round_id,
                    "resource_id": to_room,
                    "contenders": [action.action_id],
                    "winner_action_id": None,
                    "reason": "room_locked_down",
                }
            )
            decided_ids.add(action.action_id)

        # --- Phase 2: stairwell capacity ---
        stairwell_intents: dict[str, list[ActionEnvelopeMA]] = {}
        stairwell_capacities: dict[str, int] = snapshot.get("stairwell_capacities", {})
        self._current_stairwell_capacities = stairwell_capacities
        for action in all_actions:
            if action.action_id in decided_ids:
                continue
            stairwell_id = self._stairwell_resource(action)
            if stairwell_id:
                stairwell_intents.setdefault(stairwell_id, []).append(action)

        for stairwell_id, contenders in stairwell_intents.items():
            capacity = stairwell_capacities.get(stairwell_id, 5)
            sorted_contenders = self._sort_by_priority(
                contenders, directive_store, round_id, override_targets,
            )
            winners = sorted_contenders[:capacity]
            losers = sorted_contenders[capacity:]
            accepted.extend(winners)
            decided_ids.update(action.action_id for action in contenders)
            for loser in losers:
                rejections.append(
                    {
                        "agent_id": loser.agent_id,
                        "action_id": loser.action_id,
                        "action_type": loser.action_type.value,
                        "reason": "stairwell_capacity",
                        "resource_id": stairwell_id,
                    }
                )
            if losers:
                reason = self._arbitration_reason(
                    losers[0], winners[0], directive_store, override_targets, round_id,
                )
                arbitration_trace.append(
                    {
                        "round_id": round_id,
                        "resource_id": stairwell_id,
                        "contenders": sorted(action.action_id for action in contenders),
                        "winner_action_id": winners[0].action_id,
                        "reason": reason,
                    }
                )
            reservation_trace.append(
                {
                    "round_id": round_id,
                    "resource_id": stairwell_id,
                    "claim_count": len(contenders),
                    "capacity": capacity,
                    "accepted_action_ids": sorted(action.action_id for action in winners),
                    "rejected_action_ids": sorted(action.action_id for action in losers),
                }
            )

        # --- Phase 2b: elevator reservation ---
        elevator_intents: dict[str, list[ActionEnvelopeMA]] = {}
        for action in all_actions:
            if action.action_id in decided_ids:
                continue
            if action.action_type == ActionTypeMA.call_elevator:
                elevator_id = action.arguments.get("elevator_id", "elevator_main")
                elevator_intents.setdefault(elevator_id, []).append(action)

        for elevator_id, contenders in elevator_intents.items():
            sorted_contenders = self._sort_by_priority(
                contenders, directive_store, round_id, override_targets,
            )
            winners = sorted_contenders[:1]
            losers = sorted_contenders[1:]
            accepted.extend(winners)
            decided_ids.update(action.action_id for action in contenders)
            for loser in losers:
                rejections.append(
                    {
                        "agent_id": loser.agent_id,
                        "action_id": loser.action_id,
                        "action_type": loser.action_type.value,
                        "reason": "elevator_capacity",
                        "resource_id": elevator_id,
                    }
                )
            if losers:
                arbitration_trace.append(
                    {
                        "round_id": round_id,
                        "resource_id": elevator_id,
                        "contenders": sorted(action.action_id for action in contenders),
                        "winner_action_id": winners[0].action_id,
                        "reason": self._arbitration_reason(
                            losers[0], winners[0], directive_store, override_targets, round_id,
                        ),
                    }
                )

        # --- Phase 2c: exit throughput ---
        exit_intents: dict[str, list[ActionEnvelopeMA]] = {}
        exit_throughput: dict[str, int] = snapshot.get("exit_throughputs", {})
        self._current_exit_throughput = exit_throughput
        for action in all_actions:
            if action.action_id in decided_ids:
                continue
            if action.action_type in (ActionTypeMA.open_exit, ActionTypeMA.route_within_floor):
                exit_id = self._exit_resource(action)
                if exit_id:
                    exit_intents.setdefault(exit_id, []).append(action)

        for exit_id, contenders in exit_intents.items():
            throughput = exit_throughput.get(exit_id, 10)
            sorted_contenders = self._sort_by_priority(
                contenders, directive_store, round_id, override_targets,
            )
            winners = sorted_contenders[:throughput]
            losers = sorted_contenders[throughput:]
            accepted.extend(winners)
            decided_ids.update(action.action_id for action in contenders)
            for loser in losers:
                rejections.append(
                    {
                        "agent_id": loser.agent_id,
                        "action_id": loser.action_id,
                        "action_type": loser.action_type.value,
                        "reason": "exit_throughput",
                        "resource_id": exit_id,
                    }
                )
            if losers:
                arbitration_trace.append(
                    {
                        "round_id": round_id,
                        "resource_id": exit_id,
                        "contenders": sorted(action.action_id for action in contenders),
                        "winner_action_id": winners[0].action_id,
                        "reason": self._arbitration_reason(
                            losers[0], winners[0], directive_store, override_targets, round_id,
                        ),
                    }
                )

        # --- Phase 3: accept everything else (no contention) ---
        for action in all_actions:
            if action.action_id in decided_ids:
                continue
            accepted.append(action)
            decided_ids.add(action.action_id)

        return ArbitrationResult(
            accepted=accepted,
            rejections=rejections,
            reservation_trace=reservation_trace,
            arbitration_trace=arbitration_trace,
        )

    # ------------------------------------------------------------------
    # Priority sorting
    # ------------------------------------------------------------------

    def _sort_by_priority(
        self,
        contenders: list[ActionEnvelopeMA],
        directive_store: Optional[DirectiveStore],
        round_id: int,
        override_targets: dict[str, ActionEnvelopeMA],
    ) -> list[ActionEnvelopeMA]:
        """Sort contenders by arbitration priority (highest first)."""
        override_action_ids = {action.action_id for action in override_targets.values()}

        def _key(action: ActionEnvelopeMA) -> tuple[int, int, int, str]:
            override_tier = 1 if (
                action.action_id in override_action_ids
                or action.action_type == ActionTypeMA.override_floor_agent
            ) else 0
            directive_boost = 1 if (
                directive_store is not None
                and self._has_matching_high_priority_directive(action, directive_store, round_id)
            ) else 0
            priority_class = PRIORITY_CLASS.get(action.action_type, 0)
            return (-override_tier, -directive_boost, -priority_class, action.action_id)

        return sorted(contenders, key=_key)

    def _arbitration_reason(
        self,
        loser: ActionEnvelopeMA,
        winner: ActionEnvelopeMA,
        directive_store: Optional[DirectiveStore],
        override_targets: dict[str, ActionEnvelopeMA],
        round_id: int,
    ) -> str:
        """Determine the reason the loser lost."""
        override_action_ids = {action.action_id for action in override_targets.values()}
        if winner.action_type == ActionTypeMA.override_floor_agent or winner.action_id in override_action_ids:
            return "orchestrator_override"
        if directive_store is not None and self._has_matching_high_priority_directive(
            winner, directive_store, round_id,
        ):
            return "directive_priority"
        winner_pc = PRIORITY_CLASS.get(winner.action_type, 0)
        loser_pc = PRIORITY_CLASS.get(loser.action_type, 0)
        if winner_pc != loser_pc:
            return "priority_class"
        return "action_id_tiebreak"

    def _has_matching_high_priority_directive(
        self,
        action: ActionEnvelopeMA,
        directive_store: DirectiveStore,
        round_id: int,
    ) -> bool:
        for directive in directive_store.matching_directives(action.agent_id, round_id):
            if directive.priority != DirectivePriority.high:
                continue
            if self._directive_matches_action(directive, action):
                return True
        return False

    def _directive_matches_action(
        self,
        directive: Directive,
        action: ActionEnvelopeMA,
    ) -> bool:
        if directive.directive_type == "evacuate_via_stairwell":
            stairwell_id = directive.params.get("stairwell_id")
            return (
                stairwell_id is not None
                and action.action_type in {ActionTypeMA.route_within_floor, ActionTypeMA.route_between_floors}
                and action.arguments.get("stairwell_id") == stairwell_id
            )
        if directive.directive_type == "prioritize_floor_egress":
            return action.action_type in {
                ActionTypeMA.route_within_floor,
                ActionTypeMA.route_between_floors,
                ActionTypeMA.open_exit,
            }
        if directive.directive_type == "allow_elevator_use":
            return action.action_type in {
                ActionTypeMA.call_elevator,
                ActionTypeMA.route_between_floors,
            }
        return False

    # ------------------------------------------------------------------
    # Resource identification helpers
    # ------------------------------------------------------------------

    def _stairwell_resource(self, action: ActionEnvelopeMA) -> Optional[str]:
        """Extract stairwell resource id from an action, if it uses one."""
        if action.action_type == ActionTypeMA.route_within_floor:
            return _route_target_if_known(
                action,
                "stairwell_id",
                getattr(self, "_current_stairwell_capacities", {}),
            )
        if action.action_type == ActionTypeMA.route_between_floors:
            return action.arguments.get("stairwell_id")
        if action.action_type == ActionTypeMA.call_elevator:
            return None  # elevators are separate
        return None

    def _exit_resource(self, action: ActionEnvelopeMA) -> Optional[str]:
        """Extract exit resource id from an action, if it uses one."""
        if action.action_type == ActionTypeMA.open_exit:
            return action.arguments.get("exit_id")
        if action.action_type == ActionTypeMA.route_within_floor:
            return _route_target_if_known(
                action,
                "exit_id",
                getattr(self, "_current_exit_throughput", {}),
            )
        return None
