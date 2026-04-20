from __future__ import annotations

import random
from collections import deque

from pydantic import BaseModel, ConfigDict, Field

from evacos_ma.models import Building
from evacos_ma.schemas.multi_agent import CorridorView, RoomView


class VisibilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_visibility_radius: int = 2
    max_age_before_hidden: int = 8
    sensor_quality_hazard_slope: float = 0.6
    sensor_quality_min: float = 0.2
    occupancy_jitter_at_min: int = 2
    severity_jitter_at_min: float = 0.15
    unseen_room_initial_age: int = 1


class FloorVisibilityState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_observed_round_by_room: dict[str, int] = Field(default_factory=dict)
    scouted_rooms_this_round: set[str] = Field(default_factory=set)


def _parse_floor_number(floor_id: str | int) -> int:
    if isinstance(floor_id, int):
        return floor_id
    if floor_id.startswith("floor_"):
        return int(floor_id.split("_", maxsplit=1)[1])
    if floor_id.startswith("F") and floor_id[1:].isdigit():
        return int(floor_id[1:])
    return int(floor_id)


def _room_lookup(building: Building) -> dict[str, object]:
    return {
        room.room_id: room
        for floor in building.floors
        for room in floor.rooms
    }


def _rooms_on_floor(building: Building, floor_num: int) -> list[object]:
    return next(
        floor.rooms
        for floor in building.floors
        if floor.floor_id == floor_num
    )


def _corridors_on_floor(building: Building, floor_num: int) -> list[object]:
    return next(
        floor.corridors
        for floor in building.floors
        if floor.floor_id == floor_num
    )


def _graph_neighborhood(
    building: Building,
    start_room_ids: set[str],
    radius: int,
    floor_num: int,
) -> set[str]:
    room_lookup = _room_lookup(building)
    visited = set(start_room_ids)
    queue: deque[tuple[str, int]] = deque((room_id, 0) for room_id in sorted(start_room_ids))
    while queue:
        room_id, distance = queue.popleft()
        if distance >= radius:
            continue
        room = room_lookup[room_id]
        for adjacent_id in room.adjacent_node_ids:
            if adjacent_id not in room_lookup:
                continue
            adjacent_room = room_lookup[adjacent_id]
            if adjacent_room.floor_id != floor_num or adjacent_id in visited:
                continue
            visited.add(adjacent_id)
            queue.append((adjacent_id, distance + 1))
    return visited


def _occupancy_jitter_bound(sensor_quality: float, config: VisibilityConfig) -> int:
    if sensor_quality >= 1.0:
        return 0
    denominator = max(1e-9, 1.0 - config.sensor_quality_min)
    scale = max(0.0, min(1.0, (1.0 - sensor_quality) / denominator))
    return round(config.occupancy_jitter_at_min * scale)


def _severity_jitter_bound(sensor_quality: float, config: VisibilityConfig) -> float:
    if sensor_quality >= 1.0:
        return 0.0
    denominator = max(1e-9, 1.0 - config.sensor_quality_min)
    scale = max(0.0, min(1.0, (1.0 - sensor_quality) / denominator))
    return config.severity_jitter_at_min * scale


def _clamp_int(value: int, low: int = 0) -> int:
    return max(low, value)


def _clamp_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_floor_observation(
    *,
    floor_id: str | int,
    building: Building,
    hazard_engine,
    vis_state: FloorVisibilityState,
    current_round: int,
    rng_seed: int,
    config: VisibilityConfig,
) -> tuple[list[RoomView], list[CorridorView], dict[str, int], float]:
    del hazard_engine  # room hazard state is already mirrored onto the building model

    floor_num = _parse_floor_number(floor_id)
    floor_rooms = _rooms_on_floor(building, floor_num)
    room_lookup = _room_lookup(building)
    known_group_rooms = {
        room_id
        for room_id, last_round in vis_state.last_observed_round_by_room.items()
        if room_id in room_lookup
        and room_lookup[room_id].floor_id == floor_num
        and room_lookup[room_id].occupancy.total > 0
        and current_round - last_round <= config.max_age_before_hidden
    }
    anchor_rooms = set(known_group_rooms)
    anchor_rooms.update(
        room_id for room_id in vis_state.scouted_rooms_this_round
        if room_id in room_lookup and room_lookup[room_id].floor_id == floor_num
    )

    fresh_room_ids = _graph_neighborhood(
        building,
        anchor_rooms,
        config.base_visibility_radius,
        floor_num,
    ) if anchor_rooms else set()

    visibility_age_by_room: dict[str, int] = {}
    visible_room_ids: list[str] = []

    for room in sorted(floor_rooms, key=lambda item: item.room_id):
        if room.room_id in fresh_room_ids:
            vis_state.last_observed_round_by_room[room.room_id] = current_round
            visibility_age_by_room[room.room_id] = 0
            visible_room_ids.append(room.room_id)
            continue

        seen_before = room.room_id in vis_state.last_observed_round_by_room
        if seen_before:
            raw_age = current_round - vis_state.last_observed_round_by_room[room.room_id]
        else:
            raw_age = current_round + config.unseen_room_initial_age
        capped_age = min(config.max_age_before_hidden, max(0, raw_age))
        visibility_age_by_room[room.room_id] = capped_age
        if seen_before and raw_age <= config.max_age_before_hidden:
            visible_room_ids.append(room.room_id)

    visible_ground_truth_rooms = [room_lookup[room_id] for room_id in visible_room_ids]
    if visible_ground_truth_rooms:
        mean_hazard = sum(room.hazard.severity for room in visible_ground_truth_rooms) / len(visible_ground_truth_rooms)
        sensor_quality = max(
            config.sensor_quality_min,
            1.0 - config.sensor_quality_hazard_slope * mean_hazard,
        )
    else:
        sensor_quality = 1.0

    rng = random.Random(rng_seed)
    occupancy_jitter_bound = _occupancy_jitter_bound(sensor_quality, config)
    severity_jitter_bound = _severity_jitter_bound(sensor_quality, config)

    visible_rooms: list[RoomView] = []
    for room_id in visible_room_ids:
        room = room_lookup[room_id]
        mobile = room.occupancy.mobile
        injured = room.occupancy.injured
        impaired = room.occupancy.mobility_impaired
        hazard_severity = room.hazard.severity
        smoke_level = room.hazard.smoke
        if sensor_quality < 1.0:
            mobile = _clamp_int(mobile + rng.randint(-occupancy_jitter_bound, occupancy_jitter_bound))
            injured = _clamp_int(injured + rng.randint(-occupancy_jitter_bound, occupancy_jitter_bound))
            impaired = _clamp_int(impaired + rng.randint(-occupancy_jitter_bound, occupancy_jitter_bound))
            hazard_severity = _clamp_float(hazard_severity + rng.uniform(-severity_jitter_bound, severity_jitter_bound))
            smoke_level = _clamp_float(smoke_level + rng.uniform(-severity_jitter_bound, severity_jitter_bound))

        visible_rooms.append(
            RoomView(
                room_id=room.room_id,
                floor_id=f"floor_{room.floor_id}",
                occupancy_mobile=mobile,
                occupancy_injured=injured,
                occupancy_mobility_impaired=impaired,
                hazard_severity=hazard_severity,
                smoke_level=smoke_level,
                accessible=room.accessible,
                passable=room.hazard.passable,
            )
        )

    visible_room_set = set(visible_room_ids)
    visible_corridors = [
        CorridorView(
            corridor_id=corridor.corridor_id,
            from_node_id=corridor.from_node_id,
            to_node_id=corridor.to_node_id,
            hazard_severity=corridor.hazard.severity,
            passable=corridor.hazard.passable,
        )
        for corridor in sorted(_corridors_on_floor(building, floor_num), key=lambda item: item.corridor_id)
        if corridor.from_node_id in visible_room_set and corridor.to_node_id in visible_room_set
    ]
    return visible_rooms, visible_corridors, visibility_age_by_room, sensor_quality
