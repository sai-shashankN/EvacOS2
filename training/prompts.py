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
PROMPT_TEMPLATE_VERSION = "2026.04.29.1"


# ---------------------------------------------------------------------------
# Floor-agent prompt
# ---------------------------------------------------------------------------


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _floor_action_examples(obs: FloorAgentObservationMA) -> str:
    """Build few-shot actions from the current observation IDs.

    Static IDs in examples are surprisingly dangerous here: small models copy
    them literally. Keep examples grounded in the exact IDs visible now.
    """

    allowed = {str(getattr(item, "value", item)) for item in (obs.action_mask or [])}
    visible_rooms = [room.room_id for room in (obs.visible_rooms or [])]
    civilian_rooms = [
        group.location_room_id
        for group in (obs.visible_civilian_groups or [])
        if int(getattr(group, "count", 0) or 0) > 0 and group.location_room_id
    ]
    from_room = civilian_rooms[0] if civilian_rooms else (visible_rooms[0] if visible_rooms else None)
    scout_room = visible_rooms[-1] if visible_rooms else from_room

    exits = list(obs.exits_on_floor or [])
    preferred_exit = next((exit_view for exit_view in exits if not exit_view.blocked), None)
    preferred_exit = preferred_exit or (exits[0] if exits else None)
    stairwells = list(obs.stairwell_entries or [])
    preferred_stair = next((stair for stair in stairwells if not stair.blocked), None)
    preferred_stair = preferred_stair or (stairwells[0] if stairwells else None)

    base = {
        "episode_id": obs.episode_id,
        "round_id": obs.round_id,
        "agent_id": obs.agent_id,
    }
    examples: list[dict[str, Any]] = []
    if "route_within_floor" in allowed and from_room and preferred_exit is not None:
        examples.append(
            {
                **base,
                "action_id": "route_to_exit",
                "action_type": "route_within_floor",
                "arguments": {"from_room_id": from_room, "exit_id": preferred_exit.exit_id},
            }
        )
    if "route_within_floor" in allowed and from_room and preferred_stair is not None:
        examples.append(
            {
                **base,
                "action_id": "route_to_stair",
                "action_type": "route_within_floor",
                "arguments": {"from_room_id": from_room, "stairwell_id": preferred_stair.stairwell_id},
            }
        )
    if (
        "open_exit" in allowed
        and preferred_exit is not None
        and bool(getattr(preferred_exit, "requires_open_action", False))
    ):
        examples.append(
            {
                **base,
                "action_id": "open_exit",
                "action_type": "open_exit",
                "arguments": {"exit_id": preferred_exit.exit_id},
            }
        )
    if "scout" in allowed and scout_room:
        examples.append(
            {
                **base,
                "action_id": "scout_room",
                "action_type": "scout",
                "arguments": {"target_room_id": scout_room},
            }
        )
    if not examples:
        examples.append(
            {
                **base,
                "action_id": "wait_safe",
                "action_type": "wait",
                "arguments": {},
            }
        )
    return " ".join(_compact_json(example) for example in examples)


def _floor_route_argument_menu(obs: FloorAgentObservationMA) -> str:
    """Return copyable route argument bundles grounded in visible IDs."""

    allowed = {str(getattr(item, "value", item)) for item in (obs.action_mask or [])}
    if "route_within_floor" not in allowed:
        return "[]"

    visible_rooms = [room.room_id for room in (obs.visible_rooms or [])]
    civilian_rooms = [
        group.location_room_id
        for group in (obs.visible_civilian_groups or [])
        if int(getattr(group, "count", 0) or 0) > 0 and group.location_room_id
    ]
    source_rooms = civilian_rooms or visible_rooms
    bundles: list[dict[str, str]] = []

    for from_room in source_rooms[:3]:
        for exit_view in (obs.exits_on_floor or []):
            if not bool(getattr(exit_view, "blocked", False)):
                bundles.append({"from_room_id": from_room, "exit_id": exit_view.exit_id})
        for stair in (obs.stairwell_entries or []):
            if not bool(getattr(stair, "blocked", False)):
                bundles.append({"from_room_id": from_room, "stairwell_id": stair.stairwell_id})

    if not bundles and len(visible_rooms) > 1:
        for from_room in source_rooms[:2]:
            for to_room in visible_rooms[:4]:
                if to_room != from_room:
                    bundles.append({"from_room_id": from_room, "to_room_id": to_room})

    return _compact_json({"route_within_floor_arguments": bundles[:12]})


def _orchestrator_priority_argument_menu(obs: OrchestratorObservationMA) -> str:
    """Return a copyable priority-floor argument bundle from visible summaries."""

    summaries = list(obs.floor_summaries or [])
    ordered = sorted(
        summaries,
        key=lambda summary: (
            float(getattr(summary, "hazard_severity", 0.0) or 0.0),
            float(getattr(summary, "queue_pressure", 0.0) or 0.0),
            int(getattr(summary, "known_civilian_count", 0) or 0),
        ),
        reverse=True,
    )
    floor_ids = [summary.floor_id for summary in ordered if summary.floor_id]
    if not floor_ids:
        floor_ids = [summary.floor_id for summary in summaries if summary.floor_id]
    return _compact_json({"ordered_floor_ids": floor_ids[:5]})


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
    stairwells_str = json.dumps(
        [
            {
                "stairwell_id": stair.stairwell_id,
                "blocked": stair.blocked,
                "capacity": stair.capacity_per_step,
            }
            for stair in (obs.stairwell_entries or [])
        ]
    )
    corridors_str = json.dumps(
        [
            {
                "corridor_id": corridor.corridor_id,
                "from_node_id": corridor.from_node_id,
                "to_node_id": corridor.to_node_id,
                "hazard_severity": corridor.hazard_severity,
                "passable": corridor.passable,
            }
            for corridor in (obs.visible_corridors or [])
        ]
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
        f"Stairwells: {stairwells_str}",
        f"Corridors: {corridors_str}",
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

    user_parts.append(
        "Decision policy: if civilians are visible and an allowed action can evacuate them, "
        "open a route, gather missing information, or escalate a concrete blocker, choose that "
        "active action instead of wait."
    )
    user_parts.append(
        "Use wait only when no safe/useful action is available, the action mask blocks the better "
        "options, or the situation genuinely requires holding."
    )
    user_parts.append(
        "For route_within_floor, use to_room_id only for actual rooms. If the target is an exit "
        "or stairwell, put it in exit_id or stairwell_id so the action changes evacuation state."
    )
    user_parts.append(
        "Valid route_within_floor argument bundles you may copy exactly: "
        + _floor_route_argument_menu(obs)
    )
    user_parts.append(
        "When civilians and a usable exit or stairwell are visible, prefer route_within_floor "
        "with exit_id or stairwell_id. Use room-to-room routes only as a temporary safe-room move."
    )
    user_parts.append(
        "For gas or flood, avoid routes through Corridors where passable=false or hazard_severity is high."
    )
    user_parts.append(
        "Do not use open_exit unless an exit has requires_open=true. If requires_open=false, "
        "route civilians to that exit with route_within_floor instead."
    )
    user_parts.append(
        "Copy IDs exactly from Rooms, Exits, Stairwells, and Civilians. Never invent IDs, "
        "and never use floor_id, agent_id, any_room, all_rooms, none, or undefined as a room/exit target."
    )
    user_parts.append(
        "Examples using current observation IDs when allowed by the action mask: "
        + _floor_action_examples(obs)
    )

    user_msg = "\n".join(user_parts)

    response_format = (
        "\n### Response format\n"
        "Respond with a SINGLE compact JSON object matching ActionEnvelopeMA. "
        "No prose, no markdown, no code fences, and keep it on one line.\n"
        "Required keys: episode_id, round_id, agent_id, action_id (any unique string), "
        "action_type (one of the allowed actions), arguments (dict).\n"
        "Never output the literal placeholder action_type=\"action_type\"; replace it with an exact "
        "allowed action such as route_within_floor or wait.\n"
        "round_id must be the integer shown above, not a string or composite id.\n"
        "client_metadata must be omitted or {}, never null.\n"
        "Omit optional keys during training; especially omit rationale to keep the JSON short.\n"
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
    user_parts.append(
        "Copyable evacuate_floor_priority arguments: "
        + _orchestrator_priority_argument_menu(obs)
    )
    user_parts.append(
        "Orchestrator argument schemas: "
        'evacuate_floor_priority -> {"ordered_floor_ids":["floor_1","floor_3"]}; '
        'broadcast_directive -> {"directive":{"directive_id":"dir_5","target":"floor_1",'
        '"directive_type":"prioritize_evacuation","params":{},"priority":"high",'
        f'"issued_round":{obs.round_id},"ttl_rounds":5,"human_readable_note":"..."}}; '
        'override_floor_agent -> {"target_floor_agent_id":"floor_1_agent",'
        '"replacement_action_type":"wait","replacement_arguments":{}}; '
        'request_explanation -> {"target_floor_agent_id":"floor_1_agent","question":"..."}; '
        "wait -> {}."
    )
    user_parts.append(
        "For evacuate_floor_priority, never use priority_floor, floor, target_floor, "
        "or a single string. The only valid key is ordered_floor_ids and its value "
        "must be a list of visible floor_id strings."
    )
    user_parts.append(
        "For evacuate_floor_priority, the entire arguments object must be exactly "
        '{"ordered_floor_ids":[...]}. Do not wrap it inside an action-specific '
        "or nested object."
    )
    user_parts.append(
        "Keep JSON identifiers short. If unsure, omit optional ids; the parser can "
        "auto-fill missing episode_id, agent_id, round_id, and action_id."
    )

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
        "Respond with a SINGLE compact JSON object matching ActionEnvelopeMA. "
        "No prose, no markdown, no code fences, and keep it on one line.\n"
        "Required keys: episode_id, round_id, agent_id, action_id (any unique string), "
        "action_type (one of the allowed actions), arguments (dict).\n"
        "Never output the literal placeholder action_type=\"action_type\"; replace it with an exact "
        "allowed action from the action mask.\n"
        "round_id must be the integer shown above, not a string or composite id.\n"
        "client_metadata must be omitted or {}, never null.\n"
        "Omit optional keys during training; especially omit rationale to keep the JSON short.\n"
        "Do not repeat long ids or continue generating after the closing brace.\n"
        f"Prompt template version: {version}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg + response_format},
    ]
