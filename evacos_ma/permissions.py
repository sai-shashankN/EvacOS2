"""Role permission enforcement for multi-agent actions.

Phase 5: validates that submitted actions are permitted for the agent's role.
"""

from __future__ import annotations

from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentRole,
)


class ValidationResult:
    """Lightweight validation result — avoids a Pydantic model to stay flexible."""

    __slots__ = ("valid", "reason", "action_id")

    def __init__(self, valid: bool, reason: str, action_id: str) -> None:
        self.valid = valid
        self.reason = reason
        self.action_id = action_id

    def __bool__(self) -> bool:
        return self.valid


FLOOR_AGENT_ACTIONS: frozenset[ActionTypeMA] = frozenset({
    ActionTypeMA.route_within_floor,
    ActionTypeMA.prioritize_room,
    ActionTypeMA.open_exit,
    ActionTypeMA.lockdown_room,
    ActionTypeMA.scout,
    ActionTypeMA.predict_state,
    ActionTypeMA.handoff_to_orchestrator,
    ActionTypeMA.wait,
})

ORCHESTRATOR_ACTIONS: frozenset[ActionTypeMA] = frozenset({
    ActionTypeMA.route_between_floors,
    ActionTypeMA.call_elevator,
    ActionTypeMA.evacuate_floor_priority,
    ActionTypeMA.broadcast_directive,
    ActionTypeMA.override_floor_agent,
    ActionTypeMA.request_explanation,
    ActionTypeMA.wait,
})


def validate_action_for_role(
    action: ActionEnvelopeMA,
    role: AgentRole,
) -> ValidationResult:
    """Return ValidationResult indicating if *action* is permitted for *role*."""
    allowed = FLOOR_AGENT_ACTIONS if role == AgentRole.floor_agent else ORCHESTRATOR_ACTIONS
    if action.action_type in allowed:
        return ValidationResult(valid=True, reason="ok", action_id=action.action_id)
    return ValidationResult(valid=False, reason="role_forbidden", action_id=action.action_id)


def action_mask_for_role(role: AgentRole) -> list[str]:
    """Return sorted list of ActionTypeMA values permitted for *role*."""
    allowed = FLOOR_AGENT_ACTIONS if role == AgentRole.floor_agent else ORCHESTRATOR_ACTIONS
    return sorted(a.value for a in allowed)
