"""Directive lifecycle management for the multi-agent protocol.

Phase 5: storage, expiration, supersession, and priority matching for
orchestrator directives targeting floor agents.
"""

from __future__ import annotations

from typing import Any, Optional

from evacos_ma.schemas.multi_agent import (
    AgentId,
    Directive,
    DirectiveId,
    DirectiveOutcome,
    DirectivePriority,
)


# Recognised directive vocabulary this phase.
RECOGNISED_DIRECTIVE_TYPES: frozenset[str] = frozenset({
    "evacuate_via_stairwell",
    "prioritize_floor_egress",
    "hold_floor",
    "allow_elevator_use",
})


class DirectiveStore:
    """Per-episode directive registry with expiration and supersession."""

    def __init__(self) -> None:
        self._active: dict[DirectiveId, Directive] = {}
        self._outcomes: list[DirectiveOutcome] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def issue(self, directive: Directive) -> DirectiveOutcome:
        """Register a new directive.  Handles supersession."""
        # Supersession
        if directive.supersedes_directive_id_or_null is not None:
            old_id = directive.supersedes_directive_id_or_null
            if old_id in self._active:
                del self._active[old_id]

        self._active[directive.directive_id] = directive
        outcome = DirectiveOutcome(
            directive_id=directive.directive_id,
            target_floor_id=_extract_floor_id(directive.target),
            directive_type=directive.directive_type,
            accepted=True,
            outcome_summary="issued",
        )
        self._outcomes.append(outcome)
        return outcome

    def tick(self, current_round: int) -> None:
        """Prune expired directives."""
        expired = [
            did
            for did, d in self._active.items()
            if current_round >= d.issued_round + d.ttl_rounds
        ]
        for did in expired:
            del self._active[did]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def active_directive_for_target(
        self,
        target_id: str,
        current_round: int,
    ) -> Optional[Directive]:
        """Return the highest-priority live directive for *target_id*.

        *target_id* may be a floor agent id, floor public id, or ``"all"``.
        Priority ordering: ``high > normal > low``; ties broken by newest
        ``issued_round``.
        """
        candidates = self._matching_directives(target_id, current_round)
        if not candidates:
            return None
        return max(candidates, key=_directive_priority_key)

    def has_high_priority_directive(
        self,
        target_id: str,
        current_round: int,
        directive_type: str,
    ) -> bool:
        """Return True if there is a live high-priority directive of *directive_type* for *target_id*."""
        candidates = self._matching_directives(target_id, current_round)
        return any(
            d.priority == DirectivePriority.high
            and d.directive_type == directive_type
            for d in candidates
        )

    def directive_outcomes(self) -> list[DirectiveOutcome]:
        """Return all directive outcomes issued this episode."""
        return list(self._outcomes)

    def matching_directives(
        self,
        target_id: str,
        current_round: int,
    ) -> list[Directive]:
        """Return all live directives matching *target_id*."""
        return list(self._matching_directives(target_id, current_round))

    def _matching_directives(
        self,
        target_id: str,
        current_round: int,
    ) -> list[Directive]:
        results: list[Directive] = []
        for d in self._active.values():
            if current_round >= d.issued_round + d.ttl_rounds:
                continue
            if _directive_matches(d, target_id):
                results.append(d)
        return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _directive_matches(d: Directive, target_id: str) -> bool:
    """Check if directive *d* targets *target_id*."""
    if d.target == "all":
        return True
    if d.target == target_id:
        return True
    # Allow matching floor id from agent id: floor_0_agent -> floor_0
    if target_id.startswith("floor_") and "_agent" in target_id:
        floor_id = target_id.rsplit("_agent", 1)[0]
        if d.target == floor_id:
            return True
    # Also allow matching agent id from floor id
    if d.target.startswith("floor_") and "_agent" not in d.target:
        floor_id = d.target
        agent_id = floor_id + "_agent"
        if target_id == agent_id:
            return True
    return False


_PRIORITY_ORDER = {
    DirectivePriority.high: 3,
    DirectivePriority.normal: 2,
    DirectivePriority.low: 1,
}


def _directive_priority_key(d: Directive) -> tuple[int, int]:
    return (_PRIORITY_ORDER.get(d.priority, 0), d.issued_round)


def _extract_floor_id(target: str) -> str:
    """Best-effort floor id extraction from directive target."""
    if target.startswith("floor_"):
        return target.rsplit("_agent", 1)[0]
    return target
