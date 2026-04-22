"""Prompt builders for floor-agent and orchestrator roles.

Each builder consumes the frozen observation envelopes and produces
OpenAI-chat-format messages.  Heavy-dependency-free.
"""

from __future__ import annotations

import json
from typing import Any

from evacos_ma.schemas.multi_agent import (
    FloorAgentObservationMA,
    OrchestratorObservationMA,
)

# Version must match the module-level constant in training/__init__
PROMPT_TEMPLATE_VERSION = "2026.04.20"


# ---------------------------------------------------------------------------
# Floor-agent prompt
# ---------------------------------------------------------------------------


def build_floor_prompt(
    obs: FloorAgentObservationMA,
    *,
    version: str = PROMPT_TEMPLATE_VERSION,
) -> list[dict[str, str]]:
    """Build a chat-format prompt for a floor agent."""
    system_msg = (
        "You are a floor evacuation agent in a multi-story building emergency.\n"
        "Your role: evacuate civilians from your assigned floor safely.\n"
        f"Episode: {obs.episode_id}\n"
        f"Agent ID: {obs.agent_id}\n"
        f"Floor: {obs.floor_id}\n"
        f"Disaster: {obs.disaster_family}\n"
        f"Round: {obs.round_id}\n"
        f"Step: {obs.step}/{obs.max_steps}\n"
    )

    # Action mask
    system_msg += f"\nAllowed actions: {json.dumps(obs.action_mask)}\n"

    # Observation summary
    rooms_str = json.dumps(
        [
            {"room_id": r.room_id, "mobile": r.occupancy_mobile, "injured": r.occupancy_injured,
             "severity": r.hazard_severity, "smoke": r.smoke_level, "accessible": r.accessible}
            for r in (obs.visible_rooms or [])
        ]
    )
    exits_str = json.dumps(
        [{"exit_id": e.exit_id, "blocked": e.blocked, "requires_open": e.requires_open_action}
         for e in (obs.exits_on_floor or [])]
    )
    civilians_str = json.dumps(
        [{"group_id": c.civilian_group_id, "room": c.location_room_id,
          "count": c.count, "status": c.status, "mobility": c.mobility_profile}
         for c in (obs.visible_civilian_groups or [])]
    )
    hazards_str = json.dumps(
        [{"hazard_id": h.hazard_id, "type": h.hazard_type, "severity": h.severity,
          "room": h.room_id}
         for h in (obs.local_hazards or [])]
    )

    user_parts: list[str] = [
        f"Rooms: {rooms_str}",
        f"Exits: {exits_str}",
        f"Civilians: {civilians_str}",
        f"Hazards: {hazards_str}",
    ]

    if obs.active_directive is not None:
        d = obs.active_directive
        user_parts.append(
            f"Active directive: {d.directive_type} -> {d.target} "
            f"(priority={d.priority.value}, note={d.human_readable_note})"
        )

    if obs.override_applied_last_round:
        user_parts.append(f"Override applied last round: {obs.override_reason_last_round}")

    user_msg = "\n".join(user_parts)

    response_format = (
        "\n### Response format\n"
        "Respond with a SINGLE JSON object matching ActionEnvelopeMA. No prose, no code fences.\n"
        "Required keys: episode_id, round_id, agent_id, action_id (any unique string), "
        "action_type (one of the allowed actions), arguments (dict), "
        "rationale (optional string), client_metadata (optional dict).\n"
        f"Prompt template version: {version}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg + response_format},
    ]


# ---------------------------------------------------------------------------
# Orchestrator prompt
# ---------------------------------------------------------------------------


def build_orchestrator_prompt(
    obs: OrchestratorObservationMA,
    *,
    version: str = PROMPT_TEMPLATE_VERSION,
) -> list[dict[str, str]]:
    """Build a chat-format prompt for the orchestrator."""
    system_msg = (
        "You are the ORCHESTRATOR agent coordinating evacuation across all floors.\n"
        "Your role: prioritize floors, manage inter-floor routing, issue directives, "
        "and override floor agents when necessary.\n"
        f"Episode: {obs.episode_id}\n"
        "Agent ID: orchestrator\n"
        f"Disaster: {obs.disaster_family}\n"
        f"Round: {obs.round_id}\n"
        f"Step: {obs.step}/{obs.max_steps}\n"
    )

    system_msg += f"\nAllowed actions: {json.dumps(obs.action_mask)}\n"

    # Floor summaries
    summaries_str = json.dumps(
        [
            {"floor_id": s.floor_id, "civilians": s.known_civilian_count,
             "unknown_rooms": s.unknown_room_count, "hazard": s.hazard_severity,
             "queue_pressure": s.queue_pressure, "exit_cap": s.exit_capacity_remaining}
            for s in (obs.floor_summaries or [])
        ]
    )

    user_parts: list[str] = [f"Floor summaries: {summaries_str}"]

    # Belief rollup
    br = obs.belief_rollup
    user_parts.append(
        f"Beliefs: total={br.total_beliefs}, avg_conf={br.avg_confidence:.2f}, "
        f"resolved={br.resolved_count}, pending={br.pending_count}"
    )

    # Recent floor actions
    if obs.recent_floor_actions:
        actions_str = json.dumps(
            [{"agent": a.agent_id, "type": a.action_type, "round": a.round_id}
             for a in obs.recent_floor_actions[:10]]
        )
        user_parts.append(f"Recent floor actions: {actions_str}")

    # Escalations
    if obs.unresolved_escalations:
        esc_str = json.dumps(
            [{"agent": e.agent_id, "floor": e.floor_id, "cat": e.category, "urgency": e.urgency}
             for e in obs.unresolved_escalations]
        )
        user_parts.append(f"Unresolved escalations: {esc_str}")

    # Directive outcomes
    if obs.recent_directive_outcomes:
        do_str = json.dumps(
            [{"id": d.directive_id, "floor": d.target_floor_id,
              "type": d.directive_type, "accepted": d.accepted}
             for d in obs.recent_directive_outcomes[:5]]
        )
        user_parts.append(f"Recent directive outcomes: {do_str}")

    user_msg = "\n".join(user_parts)

    response_format = (
        "\n### Response format\n"
        "Respond with a SINGLE JSON object matching ActionEnvelopeMA. No prose, no code fences.\n"
        "Required keys: episode_id, round_id, agent_id, action_id (any unique string), "
        "action_type (one of the allowed actions), arguments (dict), "
        "rationale (optional string), client_metadata (optional dict).\n"
        f"Prompt template version: {version}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg + response_format},
    ]
