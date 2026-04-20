"""ActionType reconciliation module.

Phase 3 decision: `ActionType` in `evacos_ma/models.py` is the canonical
single-agent enum used by Phase 1 env code. `ActionTypeMA` in
`evacos_ma/schemas/multi_agent.py` is the canonical forward-looking multi-agent
enum for Phase 5+.

Mapping of variants:
  - route_civilians (ActionType): floor-only single-agent. Replaced by
    route_within_floor in ActionTypeMA for multi-agent.
  - evacuate_floor (ActionType): floor-only single-agent. Replaced by
    evacuate_floor_priority in ActionTypeMA (orchestrator-only).
  - prioritize_room (ActionType): both single-agent and multi-agent. Exists in
    both enums.
  - block_route (ActionType): floor-only single-agent. No direct MA equivalent.
  - call_elevator (ActionType): floor-only single-agent. In MA, call_elevator
    is orchestrator-only.
  - open_exit (ActionType): both. Exists in both enums.
  - lockdown_room (ActionType): both. Exists in both enums.
  - request_render (ActionType): legacy single-agent only (rendering). Not in MA.
  - wait (ActionType): both. Exists in both enums.

The single-agent env (Phase 1) continues to use ActionType from models.py.
Multi-agent code (Phase 5+) will use ActionTypeMA. No drift can occur because
Phase 1 is frozen.
"""

from __future__ import annotations

# Re-export both enums for convenient access
from evacos_ma.models import ActionType  # noqa: F401
from evacos_ma.schemas.multi_agent import ActionTypeMA  # noqa: F401

__all__ = ["ActionType", "ActionTypeMA"]
