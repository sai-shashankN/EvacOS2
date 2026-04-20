from __future__ import annotations

from typing import Any, Callable

from evacos_ma.models import ActionType, EpisodeStateInternal, HazardType, IncidentOutcomes


SCORE_EPSILON = 1e-3


def _clamp(
    value: float,
    min_val: float = SCORE_EPSILON,
    max_val: float = 1.0 - SCORE_EPSILON,
) -> float:
    return max(min_val, min(max_val, value))


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _bounded_ratio(value: float) -> float:
    return _clamp(value)


def _iter_rooms(episode: EpisodeStateInternal):
    for floor in episode.building.floors:
        for room in floor.rooms:
            yield room


def _room_lookup(episode: EpisodeStateInternal) -> dict[str, Any]:
    return {room.room_id: room for room in _iter_rooms(episode)}


def _stairwell_lookup(episode: EpisodeStateInternal) -> dict[str, Any]:
    stairwells: dict[str, Any] = {}
    for floor in episode.building.floors:
        for stairwell in floor.stairwells:
            stairwells.setdefault(stairwell.stairwell_id, stairwell)
    return stairwells


def _stairwell_pair_lookup(episode: EpisodeStateInternal) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for stairwell_id, stairwell in _stairwell_lookup(episode).items():
        ordered_floors = sorted(stairwell.floor_ids)
        for lower_floor, upper_floor in zip(ordered_floors, ordered_floors[1:], strict=False):
            lower_room = stairwell.entry_room_ids[lower_floor]
            upper_room = stairwell.entry_room_ids[upper_floor]
            lookup[(lower_room, upper_room)] = stairwell_id
            lookup[(upper_room, lower_room)] = stairwell_id
    return lookup


def _action_records(
    episode: EpisodeStateInternal,
    *,
    action_type: ActionType | None = None,
    valid_only: bool = False,
) -> list[Any]:
    records = episode.action_history
    if action_type is not None:
        records = [record for record in records if record.action_type == action_type]
    if valid_only:
        records = [record for record in records if record.valid]
    return records


def _route_actions(episode: EpisodeStateInternal) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in _action_records(
        episode,
        action_type=ActionType.route_civilians,
        valid_only=True,
    ):
        raw = dict(record.raw_action)
        raw["_step"] = record.step
        actions.append(raw)
    return actions


def _occupancy_total(data: dict[str, Any] | None) -> int:
    if not data:
        return 0
    return int(data.get("mobile", 0)) + int(data.get("injured", 0)) + int(
        data.get("mobility_impaired", 0)
    )


def _action_target(record: Any) -> Any:
    raw = record.raw_action
    for key in (
        "to_node_id",
        "room_id",
        "edge_id",
        "exit_id",
        "elevator_id",
        "floor_id",
    ):
        if key in raw:
            return raw[key]
    return None


def _count_unnecessary_reroutes(episode: EpisodeStateInternal) -> int:
    active_sources: dict[str, int] = {}
    unnecessary = 0
    for action in _route_actions(episode):
        source = str(action.get("from_node_id", ""))
        step = int(action["_step"])
        if source in active_sources and step <= active_sources[source]:
            unnecessary += 1

        occupancy = action.get("occupancy", {})
        active_duration = 1
        if int(occupancy.get("injured", 0)) > 0:
            active_duration = 2
        active_sources[source] = step + active_duration
    return unnecessary


def _has_repeated_action_3_times(episode: EpisodeStateInternal) -> bool:
    history = episode.action_history
    for index in range(len(history) - 2):
        first, second, third = history[index : index + 3]
        if (
            first.action_type == second.action_type == third.action_type
            and _action_target(first) == _action_target(second) == _action_target(third)
        ):
            return True
    return False


def _incident_outcome_report(episode: EpisodeStateInternal) -> IncidentOutcomes:
    resolved_total = episode.civilians_saved.total + episode.civilians_lost.total
    if episode.resolved_incident_outcomes.total == resolved_total:
        report = episode.resolved_incident_outcomes.model_copy(deep=True)
    else:
        report = IncidentOutcomes(
            safe=episode.civilians_saved.total,
            deaths=episode.civilians_lost.total,
        )

    unresolved = max(0, episode.total_civilians.total - report.total)
    if unresolved > 0:
        report.severe_injury += unresolved
    return report


def _agent_anticipated_collapse(episode: EpisodeStateInternal) -> bool:
    stairwells = _stairwell_lookup(episode)
    blocked_stairwell_ids = {
        stairwell_id
        for stairwell_id, stairwell in stairwells.items()
        if stairwell.blocked
    }
    if not blocked_stairwell_ids:
        return False

    for record in _action_records(
        episode,
        action_type=ActionType.block_route,
        valid_only=True,
    ):
        if record.step < 15 and record.raw_action.get("edge_id") in blocked_stairwell_ids:
            return True

    stairwell_pairs = _stairwell_pair_lookup(episode)
    for action in _route_actions(episode):
        if int(action["_step"]) >= 15:
            continue
        source = str(action.get("from_node_id", ""))
        target = str(action.get("to_node_id", ""))
        stairwell_id = stairwell_pairs.get((source, target))
        if stairwell_id is not None and stairwell_id not in blocked_stairwell_ids:
            return True

    return False


def _grade_task_1_fire_easy(episode: EpisodeStateInternal) -> dict[str, Any]:
    total_civilians = episode.total_civilians.total
    civilians_saved = episode.civilians_saved.total
    steps_used = episode.step
    room_lookup = _room_lookup(episode)

    civilians_routed_through_fire = 0
    for action in _route_actions(episode):
        target_room = room_lookup.get(str(action.get("to_node_id", "")))
        if target_room is None:
            continue
        if (
            target_room.hazard.hazard_type == HazardType.fire
            and target_room.hazard.severity > 0.3
        ):
            civilians_routed_through_fire += 1

    base_fraction = _safe_div(civilians_saved, total_civilians)
    all_saved = civilians_saved == total_civilians and total_civilians > 0
    bonus = 0.1 if all_saved and steps_used < 20 else 0.0
    penalty = 0.05 * civilians_routed_through_fire
    score = _clamp(base_fraction + bonus - penalty)

    return {
        "score": score,
        "breakdown": {
            "base_fraction": _bounded_ratio(base_fraction),
            "bonus": bonus,
            "penalty": penalty,
            "civilians_saved": float(civilians_saved),
            "total_civilians": float(total_civilians),
            "steps_used": float(steps_used),
            "all_saved_flag": _bounded_ratio(float(all_saved)),
            "civilians_routed_through_fire": float(civilians_routed_through_fire),
        },
    }


def _grade_task_2_flood_medium(episode: EpisodeStateInternal) -> dict[str, Any]:
    total_civilians = episode.total_civilians.total
    total_injured = episode.total_civilians.injured
    civilians_saved = episode.civilians_saved.total
    injured_saved = episode.civilians_saved.injured
    steps_used = episode.step
    max_steps = episode.task.max_steps

    base_fraction = _safe_div(civilians_saved, total_civilians)
    injured_fraction = _safe_div(injured_saved, total_injured)
    weighted_fraction = (base_fraction * 0.7) + (injured_fraction * 0.3)
    efficiency_ratio = 1.0 - _safe_div(steps_used, max_steps)
    score = _clamp(weighted_fraction * (0.8 + 0.2 * efficiency_ratio))

    return {
        "score": score,
        "breakdown": {
            "base_fraction": _bounded_ratio(base_fraction),
            "injured_fraction": _bounded_ratio(injured_fraction),
            "weighted_fraction": _bounded_ratio(weighted_fraction),
            "efficiency_ratio": _bounded_ratio(efficiency_ratio),
            "steps_used": float(steps_used),
            "max_steps": float(max_steps),
            "civilians_saved": float(civilians_saved),
            "injured_saved": float(injured_saved),
            "total_civilians": float(total_civilians),
            "total_injured": float(total_injured),
        },
    }


def _grade_task_3_earthquake_hard(episode: EpisodeStateInternal) -> dict[str, Any]:
    total_civilians = episode.total_civilians.total
    total_injured = episode.total_civilians.injured
    total_impaired = episode.total_civilians.mobility_impaired

    civilians_saved = episode.civilians_saved.total
    injured_saved = episode.civilians_saved.injured
    impaired_saved = episode.civilians_saved.mobility_impaired

    base_fraction = _safe_div(civilians_saved, total_civilians)
    injured_fraction = _safe_div(injured_saved, total_injured)
    mobility_fraction = _safe_div(impaired_saved, total_impaired)
    combined_fraction = (
        (base_fraction * 0.5)
        + (injured_fraction * 0.25)
        + (mobility_fraction * 0.25)
    )

    reroutes = _count_unnecessary_reroutes(episode)
    reroute_penalty = 0.05 * reroutes
    repeated_action_loop = _has_repeated_action_3_times(episode)
    loop_penalty = 0.1 if repeated_action_loop else 0.0
    bonus = 0.15 if civilians_saved == total_civilians and total_civilians > 0 else 0.0

    score = _clamp(combined_fraction - reroute_penalty - loop_penalty + bonus)

    return {
        "score": score,
        "breakdown": {
            "base_fraction": _bounded_ratio(base_fraction),
            "injured_fraction": _bounded_ratio(injured_fraction),
            "mobility_fraction": _bounded_ratio(mobility_fraction),
            "combined_fraction": _bounded_ratio(combined_fraction),
            "reroute_penalty": reroute_penalty,
            "loop_penalty": loop_penalty,
            "bonus": bonus,
            "unnecessary_reroutes": float(reroutes),
            "repeated_action_loop_flag": _bounded_ratio(float(repeated_action_loop)),
            "civilians_saved": float(civilians_saved),
            "injured_saved": float(injured_saved),
            "mobility_saved": float(impaired_saved),
            "total_civilians": float(total_civilians),
            "total_injured": float(total_injured),
            "total_mobility_impaired": float(total_impaired),
        },
    }


def _grade_task_5_multi_cascade(episode: EpisodeStateInternal) -> dict[str, Any]:
    total_civilians = episode.total_civilians.total
    civilians_saved = episode.civilians_saved.total
    base_fraction = _safe_div(civilians_saved, total_civilians)

    panicked_room_ids = {
        room_id for room_id, timer in episode.panic_timers.items() if timer >= 3
    }
    room_lookup = _room_lookup(episode)
    panicked_civilians = sum(
        room_lookup[room_id].occupancy.total
        for room_id in panicked_room_ids
        if room_id in room_lookup
    )
    panicked_saved = min(civilians_saved, panicked_civilians)
    panic_handled_ratio = (
        _safe_div(panicked_saved, panicked_civilians) if panicked_civilians > 0 else 1.0
    )

    anticipated_collapse = _agent_anticipated_collapse(episode)
    anticipation = 0.1 if anticipated_collapse else 0.0
    score = _clamp((base_fraction * 0.7) + (panic_handled_ratio * 0.2) + anticipation)

    return {
        "score": score,
        "breakdown": {
            "base_fraction": _bounded_ratio(base_fraction),
            "panic_handled_ratio": _bounded_ratio(panic_handled_ratio),
            "anticipation": anticipation,
            "anticipated_collapse_flag": _bounded_ratio(float(anticipated_collapse)),
            "panicked_civilians": float(panicked_civilians),
            "panicked_saved": float(panicked_saved),
            "civilians_saved": float(civilians_saved),
            "total_civilians": float(total_civilians),
        },
    }


GRADERS: dict[str, Callable[[EpisodeStateInternal], dict[str, Any]]] = {
    "task_1_fire_easy": _grade_task_1_fire_easy,
    "task_2_flood_medium": _grade_task_2_flood_medium,
    "task_3_earthquake_hard": _grade_task_3_earthquake_hard,
    "task_4_cascade_hard": _grade_task_5_multi_cascade,
    # Long-horizon graders reuse existing logic
    "task_lh_fire_easy": _grade_task_1_fire_easy,
    "task_lh_flood_medium": _grade_task_2_flood_medium,
    "task_lh_cascade_hard": _grade_task_5_multi_cascade,
    "task_lh_cascade_brutal": _grade_task_5_multi_cascade,
}


def grade_episode(episode: EpisodeStateInternal) -> dict[str, Any]:
    """Grade a completed episode. Returns {'score': float, 'breakdown': dict}."""
    task_id = episode.task.task_id
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader for task: {task_id}")
    graded = grader(episode)
    graded["score"] = _clamp(float(graded["score"]))
    report = _incident_outcome_report(episode)
    graded.setdefault("breakdown", {})
    graded["breakdown"].update(
        {
            "incident_safe": float(report.safe),
            "incident_mild_injury": float(report.mild_injury),
            "incident_severe_injury": float(report.severe_injury),
            "incident_deaths": float(report.deaths),
        }
    )
    return graded
