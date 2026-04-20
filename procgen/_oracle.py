"""Capacity-aware oracle simulator for feasibility validation."""

from __future__ import annotations

from collections import defaultdict, deque

from evacos_ma.models import (
    Building,
    Elevator,
    Exit,
    Occupancy,
    Room,
    ScheduledEvent,
    Stairwell,
)


def build_adjacency(building: Building) -> dict[str, set[str]]:
    """Return adjacency map from graph edges plus exit links."""
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in building.graph_edges:
        adj[edge.from_id].add(edge.to_id)
    for floor in building.floors:
        for exit_obj in floor.exits:
            if not exit_obj.blocked:
                adj[exit_obj.adjacent_room_id].add(exit_obj.exit_id)
                adj[exit_obj.exit_id].add(exit_obj.adjacent_room_id)
    return adj


def _collect_exits(building: Building) -> dict[str, Exit]:
    exits: dict[str, Exit] = {}
    for floor in building.floors:
        for exit_obj in floor.exits:
            exits.setdefault(exit_obj.exit_id, exit_obj)
    return exits


def _collect_stairwells(building: Building) -> dict[str, Stairwell]:
    stairwells: dict[str, Stairwell] = {}
    for floor in building.floors:
        for stairwell in floor.stairwells:
            stairwells.setdefault(stairwell.stairwell_id, stairwell)
    return stairwells


def _collect_elevators(building: Building) -> dict[str, Elevator]:
    elevators: dict[str, Elevator] = {}
    for floor in building.floors:
        for elevator in floor.elevators:
            elevators.setdefault(elevator.elevator_id, elevator)
    return elevators


def _collect_rooms(building: Building) -> dict[str, Room]:
    return {room.room_id: room for floor in building.floors for room in floor.rooms}


def _room_floor_lookup(building: Building) -> dict[str, int]:
    return {room.room_id: room.floor_id for floor in building.floors for room in floor.rooms}


def _build_edge_metadata(
    building: Building,
) -> tuple[dict[str, set[str]], dict[tuple[str, str], tuple[str, str | None]]]:
    adj = build_adjacency(building)
    metadata: dict[tuple[str, str], tuple[str, str | None]] = {}
    for edge in building.graph_edges:
        metadata[(edge.from_id, edge.to_id)] = (edge.edge_type, None)

    for stairwell in _collect_stairwells(building).values():
        if stairwell.blocked:
            continue
        ordered = sorted(stairwell.floor_ids)
        for lo, hi in zip(ordered, ordered[1:]):
            lo_room = stairwell.entry_room_ids[lo]
            hi_room = stairwell.entry_room_ids[hi]
            metadata[(lo_room, hi_room)] = ("stairwell", stairwell.stairwell_id)
            metadata[(hi_room, lo_room)] = ("stairwell", stairwell.stairwell_id)

    for elevator in _collect_elevators(building).values():
        if not elevator.operational:
            continue
        rooms_sorted = sorted(
            (room.floor_id, room.room_id)
            for floor in building.floors
            for room in floor.rooms
            if elevator.elevator_id in room.adjacent_node_ids
        )
        for (_, lo_room), (_, hi_room) in zip(rooms_sorted, rooms_sorted[1:]):
            metadata[(lo_room, hi_room)] = ("elevator", elevator.elevator_id)
            metadata[(hi_room, lo_room)] = ("elevator", elevator.elevator_id)

    for floor in building.floors:
        for exit_obj in floor.exits:
            if not exit_obj.blocked:
                metadata[(exit_obj.adjacent_room_id, exit_obj.exit_id)] = ("corridor", None)
                metadata[(exit_obj.exit_id, exit_obj.adjacent_room_id)] = ("corridor", None)

    return adj, metadata


def shortest_path_to_exit(building: Building, start_room_id: str) -> int | None:
    adj = build_adjacency(building)
    exit_ids = {exit_id for exit_id, exit_obj in _collect_exits(building).items() if not exit_obj.blocked}
    queue: deque[tuple[str, int]] = deque([(start_room_id, 0)])
    visited = {start_room_id}
    while queue:
        node, dist = queue.popleft()
        for neighbor in sorted(adj.get(node, [])):
            if neighbor in exit_ids:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None


def min_path_per_floor(building: Building) -> dict[int, int]:
    result: dict[int, int] = {}
    for floor in building.floors:
        best: int | None = None
        for room in floor.rooms:
            distance = shortest_path_to_exit(building, room.room_id)
            if distance is not None and (best is None or distance < best):
                best = distance
        result[floor.floor_id] = best if best is not None else -1
    return result


def _cohort_count(occupancy: Occupancy, cohort: str) -> int:
    return int(getattr(occupancy, cohort))


def _add_to_occupancy(occupancy: Occupancy, cohort: str, count: int) -> None:
    setattr(occupancy, cohort, _cohort_count(occupancy, cohort) + count)


def _subtract_from_occupancy(occupancy: Occupancy, cohort: str, count: int) -> None:
    setattr(occupancy, cohort, max(0, _cohort_count(occupancy, cohort) - count))


def _bfs_next_hop(
    adj: dict[str, set[str]],
    edge_meta: dict[tuple[str, str], tuple[str, str | None]],
    start: str,
    targets: set[str],
    blocked_rooms: set[str],
    cohort: str,
) -> tuple[str, str, str | None] | None:
    if start in targets:
        return start, "corridor", None
    queue: deque[tuple[str, tuple[str, str, str | None] | None]] = deque([(start, None)])
    visited = {start}
    while queue:
        node, first_edge = queue.popleft()
        for neighbor in sorted(adj.get(node, [])):
            if neighbor in visited:
                continue
            if neighbor in blocked_rooms and neighbor not in targets:
                continue
            path_kind, resource_id = edge_meta.get((node, neighbor), ("corridor", None))
            if cohort == "mobility_impaired" and path_kind == "stairwell":
                continue
            next_first = first_edge or (neighbor, path_kind, resource_id)
            if neighbor in targets:
                return next_first
            visited.add(neighbor)
            queue.append((neighbor, next_first))
    return None


def run_oracle(building: Building, scheduled_events: list[ScheduledEvent]) -> float:
    rooms = _collect_rooms(building)
    exits = _collect_exits(building)
    stairwells = _collect_stairwells(building)
    elevators = _collect_elevators(building)
    room_floors = _room_floor_lookup(building)
    adj, edge_meta = _build_edge_metadata(building)
    exit_ids = {exit_id for exit_id, exit_obj in exits.items() if not exit_obj.blocked}

    total_pop = sum(room.occupancy.total for room in rooms.values())
    if total_pop == 0:
        return 1.0

    room_occ = {
        room_id: room.occupancy.model_copy(deep=True)
        for room_id, room in rooms.items()
    }
    blocked_rooms = {
        room_id
        for room_id, room in rooms.items()
        if not room.hazard.passable or not room.accessible
    }
    blocked_stairwells = {
        stairwell_id
        for stairwell_id, stairwell in stairwells.items()
        if stairwell.blocked
    }
    down_elevators = {
        elevator_id
        for elevator_id, elevator in elevators.items()
        if not elevator.operational
    }

    trigger_map: dict[int, list[ScheduledEvent]] = defaultdict(list)
    for event in scheduled_events:
        trigger_map[event.trigger_step].append(event)

    max_trigger = max((event.trigger_step for event in scheduled_events), default=0)
    horizon = min(max_trigger + 10, 100)
    cohorts = ("mobile", "injured", "mobility_impaired")
    in_transit: list[dict[str, int | str]] = []
    saved = 0

    for step in range(1, horizon + 1):
        remaining_transit: list[dict[str, int | str]] = []
        for transit in in_transit:
            if int(transit["arrival_step"]) > step:
                remaining_transit.append(transit)
                continue
            target_id = str(transit["target_id"])
            cohort = str(transit["cohort"])
            count = int(transit["count"])
            if target_id in exit_ids:
                saved += count
            else:
                _add_to_occupancy(room_occ[target_id], cohort, count)
        in_transit = remaining_transit

        for event in trigger_map.get(step, []):
            target_id = str(event.target_id)
            severity = float(event.payload.get("severity", 0.5))
            if target_id in rooms and severity >= 0.3:
                blocked_rooms.add(target_id)
            if target_id in stairwells and event.event_type.value == "stairwell_collapse":
                blocked_stairwells.add(target_id)
            if target_id in elevators and event.event_type.value == "power_outage":
                down_elevators.add(target_id)

        stair_budget = {
            stairwell_id: stairwell.capacity_per_step
            for stairwell_id, stairwell in stairwells.items()
            if stairwell_id not in blocked_stairwells
        }
        elev_budget = {
            elevator_id: elevator.capacity
            for elevator_id, elevator in elevators.items()
            if elevator_id not in down_elevators
        }
        exit_budget = {exit_id: 10 for exit_id in exit_ids}

        for room_id in sorted(room_occ):
            if room_id in blocked_rooms:
                continue
            for cohort in cohorts:
                count = _cohort_count(room_occ[room_id], cohort)
                if count <= 0:
                    continue

                next_hop = _bfs_next_hop(adj, edge_meta, room_id, exit_ids, blocked_rooms, cohort)
                if next_hop is None:
                    continue

                target_id, path_kind, resource_id = next_hop
                movable = count
                if path_kind == "stairwell":
                    if resource_id not in stair_budget:
                        continue
                    movable = min(movable, stair_budget[resource_id])
                    stair_budget[resource_id] -= movable
                elif path_kind == "elevator":
                    if resource_id not in elev_budget:
                        continue
                    movable = min(movable, elev_budget[resource_id])
                    elev_budget[resource_id] -= movable

                if target_id in exit_ids:
                    movable = min(movable, exit_budget.get(target_id, 0))
                    exit_budget[target_id] -= movable

                if movable <= 0:
                    continue

                travel_time = 2 if cohort == "injured" else 1
                if path_kind == "elevator" and resource_id is not None:
                    elevator = elevators[resource_id]
                    source_floor = room_floors[room_id]
                    target_floor = room_floors.get(target_id, source_floor)
                    floor_distance = abs(target_floor - source_floor)
                    elevator_time = max(1, floor_distance) * elevator.travel_time_per_floor
                    travel_time = max(travel_time, elevator_time)

                _subtract_from_occupancy(room_occ[room_id], cohort, movable)
                in_transit.append(
                    {
                        "arrival_step": step + travel_time,
                        "target_id": target_id,
                        "cohort": cohort,
                        "count": movable,
                    }
                )

    for transit in in_transit:
        if str(transit["target_id"]) in exit_ids and int(transit["arrival_step"]) <= horizon:
            saved += int(transit["count"])

    return saved / total_pop if total_pop > 0 else 1.0
