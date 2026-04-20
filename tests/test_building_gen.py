from __future__ import annotations

from collections import defaultdict, deque

import pytest

from evacos_ma.building_gen import generate_building
from evacos_ma.models import Building, ExitType, HazardType, Room
from evacos_ma.task_registry import TASKS


ROOM_BOUNDS = {
    "small_3floor": (4, 5),
    "medium_5floor": (5, 6),
    "complex_5floor": (6, 8),
    "medium_4floor": (5, 6),
    "complex_5floor_full": (7, 8),
    "lh_fire_easy_5floor": (5, 6),
    "lh_flood_medium_6floor": (6, 7),
    "lh_cascade_hard_6floor": (7, 8),
    "lh_cascade_brutal_7floor": (8, 10),
}

STAIRWELL_COUNTS = {
    "task_1_fire_easy": 2,
    "task_2_flood_medium": 3,
    "task_3_earthquake_hard": 4,
    "task_4_cascade_hard": 4,
    "task_lh_fire_easy": 3,
    "task_lh_flood_medium": 4,
    "task_lh_cascade_hard": 4,
    "task_lh_cascade_brutal": 5,
}

TOTAL_CIVILIANS = {
    "task_1_fire_easy": 15,
    "task_2_flood_medium": 30,
    "task_3_earthquake_hard": 50,
    "task_4_cascade_hard": 60,
    "task_lh_fire_easy": 35,
    "task_lh_flood_medium": 55,
    "task_lh_cascade_hard": 72,
    "task_lh_cascade_brutal": 110,
}

EXPECTED_EXIT_TYPES = {
    "task_1_fire_easy": {ExitType.ground},
    "task_2_flood_medium": {ExitType.rooftop},
    "task_3_earthquake_hard": {
        ExitType.ground,
        ExitType.rooftop,
        ExitType.emergency_window,
    },
    "task_4_cascade_hard": {
        ExitType.ground,
        ExitType.rooftop,
        ExitType.emergency_window,
    },
    "task_lh_fire_easy": {ExitType.ground},
    "task_lh_flood_medium": {ExitType.ground, ExitType.rooftop},
    "task_lh_cascade_hard": {
        ExitType.ground,
        ExitType.rooftop,
        ExitType.emergency_window,
    },
    "task_lh_cascade_brutal": {
        ExitType.ground,
        ExitType.rooftop,
        ExitType.emergency_window,
    },
}


def _all_rooms(building: Building) -> list[Room]:
    return [room for floor in building.floors for room in floor.rooms]


def _all_exits(building: Building):
    seen: set[str] = set()
    exits = []
    for floor in building.floors:
        for exit_obj in floor.exits:
            if exit_obj.exit_id not in seen:
                seen.add(exit_obj.exit_id)
                exits.append(exit_obj)
    return exits


def _all_stairwells(building: Building):
    seen: dict[str, object] = {}
    for floor in building.floors:
        for stairwell in floor.stairwells:
            seen.setdefault(stairwell.stairwell_id, stairwell)
    return [seen[key] for key in sorted(seen)]


def _all_elevators(building: Building):
    seen: dict[str, object] = {}
    for floor in building.floors:
        for elevator in floor.elevators:
            seen.setdefault(elevator.elevator_id, elevator)
    return [seen[key] for key in sorted(seen)]


def _build_reachability(building: Building) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in building.graph_edges:
        adjacency[edge.from_id].add(edge.to_id)
    for exit_obj in _all_exits(building):
        adjacency[exit_obj.adjacent_room_id].add(exit_obj.exit_id)
        adjacency[exit_obj.exit_id].add(exit_obj.adjacent_room_id)
    return adjacency


def _reachable_exit(room_id: str, building: Building) -> bool:
    adjacency = _build_reachability(building)
    exit_ids = {exit_obj.exit_id for exit_obj in _all_exits(building) if not exit_obj.blocked}
    queue = deque([room_id])
    visited = {room_id}
    while queue:
        current = queue.popleft()
        if current in exit_ids:
            return True
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def test_deterministic_generation() -> None:
    building_a = generate_building("task_1_fire_easy", 42)
    building_b = generate_building("task_1_fire_easy", 42)
    assert building_a.model_dump() == building_b.model_dump()


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_all_tasks_generate(task_id: str) -> None:
    building = generate_building(task_id, 42)
    assert building.building_id == f"{task_id}_42"
    assert building.floors


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_room_counts(task_id: str) -> None:
    building = generate_building(task_id, 42)
    minimum, maximum = ROOM_BOUNDS[TASKS[task_id].building_profile]
    for floor in building.floors:
        assert minimum <= len(floor.rooms) <= maximum


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_no_overlapping_rooms(task_id: str) -> None:
    building = generate_building(task_id, 42)
    for floor in building.floors:
        for index, left_room in enumerate(floor.rooms):
            left_rect = left_room.geometry
            assert left_rect.x >= 0
            assert left_rect.y >= 0
            assert left_rect.w > 0
            assert left_rect.h > 0
            assert left_rect.x + left_rect.w <= floor.width
            assert left_rect.y + left_rect.h <= floor.height
            for right_room in floor.rooms[index + 1 :]:
                right_rect = right_room.geometry
                overlaps = not (
                    left_rect.x + left_rect.w <= right_rect.x
                    or right_rect.x + right_rect.w <= left_rect.x
                    or left_rect.y + left_rect.h <= right_rect.y
                    or right_rect.y + right_rect.h <= left_rect.y
                )
                assert not overlaps


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_connectivity(task_id: str) -> None:
    building = generate_building(task_id, 42)
    for room in _all_rooms(building):
        assert _reachable_exit(room.room_id, building)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_civilian_counts(task_id: str) -> None:
    building = generate_building(task_id, 42)
    total = sum(room.occupancy.total for room in _all_rooms(building))
    assert total == TOTAL_CIVILIANS[task_id]


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_stairwell_counts(task_id: str) -> None:
    building = generate_building(task_id, 42)
    stairwells = _all_stairwells(building)
    assert len(stairwells) == STAIRWELL_COUNTS[task_id]
    for stairwell in stairwells:
        assert len(stairwell.entry_room_ids) == len(building.floors)


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_active_profiles_generate_no_elevators(task_id: str) -> None:
    building = generate_building(task_id, 42)
    assert _all_elevators(building) == []


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_exit_types(task_id: str) -> None:
    building = generate_building(task_id, 42)
    exit_types = {exit_obj.exit_type for exit_obj in _all_exits(building)}
    assert exit_types == EXPECTED_EXIT_TYPES[task_id]


def test_injured_placement() -> None:
    fire = generate_building("task_1_fire_easy", 42)
    assert sum(room.occupancy.injured for room in _all_rooms(fire)) == 0

    flood = generate_building("task_2_flood_medium", 42)
    flood_injured_floors = {
        room.floor_id for room in _all_rooms(flood) if room.occupancy.injured > 0
    }
    assert sum(room.occupancy.injured for room in _all_rooms(flood)) == 4
    assert flood_injured_floors <= {0, 1, 2}

    earthquake = generate_building("task_3_earthquake_hard", 42)
    impaired_rooms = [room for room in _all_rooms(earthquake) if room.occupancy.mobility_impaired > 0]
    exit_adjacent_room_ids = {exit_obj.adjacent_room_id for exit_obj in _all_exits(earthquake)}
    assert sum(room.occupancy.injured for room in _all_rooms(earthquake)) == 10
    assert sum(room.occupancy.mobility_impaired for room in _all_rooms(earthquake)) == 3
    assert impaired_rooms
    assert all(room.floor_id in {2, 4} for room in impaired_rooms)
    assert all(room.room_id in exit_adjacent_room_ids for room in impaired_rooms)

    cascade = generate_building("task_4_cascade_hard", 42)
    assert sum(room.occupancy.injured for room in _all_rooms(cascade)) == 8


def test_disaster_zones() -> None:
    building = generate_building("task_3_earthquake_hard", 42)
    total_rooms = len(_all_rooms(building))
    ratio = len(building.disaster_zones) / total_rooms
    assert 0.15 <= ratio <= 0.25
    assert any(
        room.hazard.hazard_type == HazardType.structural
        for room in _all_rooms(building)
    )
