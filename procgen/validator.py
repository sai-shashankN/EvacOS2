"""Feasibility validator for procedurally generated instances."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from evacos_ma.models import Building, Elevator, Exit, Room, ScheduledEvent, Stairwell

from procgen._oracle import build_adjacency, min_path_per_floor, run_oracle
from procgen.generator import GeneratedInstance, Tier, generate_instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    oracle_save_rate: float
    min_path_length_per_floor: dict[int, int]
    earliest_blockage_round: int | None
    reasons: list[str]


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


def _room_has_t0_path(adj: dict[str, set[str]], exit_ids: set[str], room_id: str) -> bool:
    visited = {room_id}
    queue = deque([room_id])
    while queue:
        node = queue.popleft()
        if node in exit_ids:
            return True
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _check_connectivity(building: Building) -> list[str]:
    reasons: list[str] = []
    adj = build_adjacency(building)
    exit_ids = {exit_id for exit_id, exit_obj in _collect_exits(building).items() if not exit_obj.blocked}
    for floor in building.floors:
        for room in floor.rooms:
            if not _room_has_t0_path(adj, exit_ids, room.room_id):
                reasons.append(
                    f"Room {room.room_id} on floor {floor.floor_id} has no path to exit at t=0"
                )
    return reasons


def _check_inter_floor_connectivity(building: Building) -> list[str]:
    reasons: list[str] = []
    rooms = _collect_rooms(building)
    sink_floors = {
        exit_obj.floor_id
        for exit_obj in _collect_exits(building).values()
        if not exit_obj.blocked
    }
    floor_graph: dict[int, set[int]] = defaultdict(set)

    for stairwell in _collect_stairwells(building).values():
        if stairwell.blocked:
            continue
        ordered = sorted(stairwell.floor_ids)
        for lo, hi in zip(ordered, ordered[1:]):
            lo_room = rooms.get(stairwell.entry_room_ids[lo])
            hi_room = rooms.get(stairwell.entry_room_ids[hi])
            if lo_room is None or hi_room is None:
                continue
            if not (lo_room.accessible and lo_room.hazard.passable and hi_room.accessible and hi_room.hazard.passable):
                continue
            floor_graph[lo].add(hi)
            floor_graph[hi].add(lo)

    for elevator in _collect_elevators(building).values():
        if not elevator.operational:
            continue
        rooms_by_floor = {
            room.floor_id: room
            for floor in building.floors
            for room in floor.rooms
            if elevator.elevator_id in room.adjacent_node_ids
        }
        ordered = sorted(rooms_by_floor)
        for lo, hi in zip(ordered, ordered[1:]):
            lo_room = rooms_by_floor[lo]
            hi_room = rooms_by_floor[hi]
            if not (lo_room.accessible and lo_room.hazard.passable and hi_room.accessible and hi_room.hazard.passable):
                continue
            floor_graph[lo].add(hi)
            floor_graph[hi].add(lo)

    for floor in building.floors:
        floor_graph.setdefault(floor.floor_id, set())

    for floor in sorted(floor_graph):
        visited = {floor}
        queue = deque([floor])
        connected = floor in sink_floors
        while queue and not connected:
            node = queue.popleft()
            if node in sink_floors:
                connected = True
                break
            for neighbor in sorted(floor_graph[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if not connected:
            reasons.append(
                f"Floor {floor} is disconnected from all safe sink floors in the inter-floor graph at t=0"
            )

    return reasons


def _find_earliest_blockage(building: Building, scheduled_events: list[ScheduledEvent]) -> int | None:
    if not scheduled_events:
        return None

    adj = build_adjacency(building)
    exit_ids = {exit_id for exit_id, exit_obj in _collect_exits(building).items() if not exit_obj.blocked}
    rooms = _collect_rooms(building)
    blocked_rooms = {
        room_id
        for room_id, room in rooms.items()
        if not room.hazard.passable or not room.accessible
    }
    trigger_map: dict[int, list[ScheduledEvent]] = defaultdict(list)
    for event in scheduled_events:
        trigger_map[event.trigger_step].append(event)

    max_step = max(event.trigger_step for event in scheduled_events)
    for step in range(1, max_step + 1):
        for event in trigger_map.get(step, []):
            if event.target_id in rooms and float(event.payload.get("severity", 0.5)) >= 0.3:
                blocked_rooms.add(event.target_id)

        for floor in building.floors:
            floor_reachable = False
            for room in floor.rooms:
                if room.room_id in blocked_rooms:
                    continue
                visited = {room.room_id}
                queue = deque([room.room_id])
                while queue:
                    node = queue.popleft()
                    if node in exit_ids:
                        floor_reachable = True
                        break
                    for neighbor in adj.get(node, set()):
                        if neighbor in blocked_rooms or neighbor in visited:
                            continue
                        visited.add(neighbor)
                        queue.append(neighbor)
                if floor_reachable:
                    break
            if not floor_reachable and floor.rooms:
                return step

    return None


def validate(instance: GeneratedInstance) -> ValidationReport:
    building = instance.building
    events = instance.scheduled_events
    reasons: list[str] = []

    reasons.extend(_check_connectivity(building))
    reasons.extend(_check_inter_floor_connectivity(building))

    oracle_rate = run_oracle(building, events)
    if oracle_rate < 0.60:
        reasons.append(f"Oracle save rate {oracle_rate:.2%} is below 60% threshold")

    earliest_blockage_round = _find_earliest_blockage(building, events)
    paths = min_path_per_floor(building)

    return ValidationReport(
        valid=not reasons,
        oracle_save_rate=oracle_rate,
        min_path_length_per_floor=paths,
        earliest_blockage_round=earliest_blockage_round,
        reasons=reasons,
    )


def mark_seed_invalid(seed: int, tier: Tier, disaster_family: str) -> None:
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "invalid_seeds.jsonl"
    record = {"seed": seed, "tier": tier, "disaster_family": str(disaster_family)}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def regenerate_until_valid(
    seed: int,
    tier: Tier,
    disaster_family: str,
    max_attempts: int = 20,
) -> tuple[GeneratedInstance, ValidationReport] | None:
    family = disaster_family if isinstance(disaster_family, str) else disaster_family.value

    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        instance = generate_instance(attempt_seed, tier, family)
        report = validate(instance)
        if report.valid:
            return instance, report
        logger.info(
            "Attempt %d seed=%d invalid: %s",
            attempt + 1,
            attempt_seed,
            "; ".join(report.reasons),
        )

    mark_seed_invalid(seed, tier, family)
    return None
