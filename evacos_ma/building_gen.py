from __future__ import annotations

import math
import random
from collections import defaultdict, deque

from evacos_ma.models import (
    Building,
    Corridor,
    EdgeRef,
    Elevator,
    Exit,
    ExitType,
    Floor,
    HazardState,
    HazardType,
    Rect,
    Room,
    Stairwell,
)
from evacos_ma.task_registry import get_task


FLOOR_WIDTH = 800
FLOOR_HEIGHT = 400

PROFILE_CONFIGS: dict[str, dict[str, object]] = {
    "small_3floor": {
        "floors": 3,
        "room_range": (4, 5),
        "stairwells": 2,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
        ],
        "elevators": 0,
        "population": {"mobile": 15, "injured": 0, "impaired": 0},
        "weight_lower": False,
    },
    "medium_5floor": {
        "floors": 5,
        "room_range": (5, 6),
        "stairwells": 3,
        "blocked_stairwells": [0],
        "exits": [
            {"floor_id": 4, "exit_type": ExitType.rooftop, "side": "center"},
        ],
        "elevators": 0,
        "population": {"mobile": 26, "injured": 4, "impaired": 0},
        "weight_lower": True,
    },
    "complex_5floor": {
        "floors": 5,
        "room_range": (6, 8),
        "stairwells": 4,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
            {"floor_id": 4, "exit_type": ExitType.rooftop, "side": "center"},
            {
                "floor_id": 2,
                "exit_type": ExitType.emergency_window,
                "side": "right",
            },
        ],
        "elevators": 0,
        "population": {"mobile": 37, "injured": 10, "impaired": 3},
        "weight_lower": False,
        "collapse_ratio": 0.2,
    },
    "medium_4floor": {
        "floors": 4,
        "room_range": (5, 6),
        "stairwells": 3,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
        ],
        "elevators": 0,
        "population": {"mobile": 25, "injured": 0, "impaired": 0},
        "weight_lower": False,
    },
    "complex_5floor_full": {
        "floors": 5,
        "room_range": (7, 8),
        "stairwells": 4,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "center"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
            {"floor_id": 4, "exit_type": ExitType.rooftop, "side": "center"},
            {
                "floor_id": 2,
                "exit_type": ExitType.emergency_window,
                "side": "right",
                "requires_open_action": True,
            },
        ],
        "elevators": 0,
        "population": {"mobile": 48, "injured": 8, "impaired": 4},
        "weight_lower": False,
    },
    # Long-horizon building profiles
    "lh_fire_easy_5floor": {
        "floors": 5,
        "room_range": (5, 6),
        "stairwells": 3,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
        ],
        "elevators": 0,
        "population": {"mobile": 30, "injured": 5, "impaired": 0},
        "weight_lower": False,
    },
    "lh_flood_medium_6floor": {
        "floors": 6,
        "room_range": (6, 7),
        "stairwells": 4,
        "blocked_stairwells": [0],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
            {"floor_id": 5, "exit_type": ExitType.rooftop, "side": "center"},
        ],
        "elevators": 0,
        "population": {"mobile": 45, "injured": 8, "impaired": 2},
        "weight_lower": True,
    },
    "lh_cascade_hard_6floor": {
        "floors": 6,
        "room_range": (7, 8),
        "stairwells": 4,
        "blocked_stairwells": [],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
            {"floor_id": 5, "exit_type": ExitType.rooftop, "side": "center"},
            {
                "floor_id": 3,
                "exit_type": ExitType.emergency_window,
                "side": "right",
                "requires_open_action": True,
            },
        ],
        "elevators": 0,
        "population": {"mobile": 55, "injured": 12, "impaired": 5},
        "weight_lower": False,
    },
    "lh_cascade_brutal_7floor": {
        "floors": 7,
        "room_range": (8, 10),
        "stairwells": 5,
        "blocked_stairwells": [0],
        "exits": [
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "left"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "center"},
            {"floor_id": 0, "exit_type": ExitType.ground, "side": "right"},
            {"floor_id": 6, "exit_type": ExitType.rooftop, "side": "center"},
            {
                "floor_id": 3,
                "exit_type": ExitType.emergency_window,
                "side": "right",
                "requires_open_action": True,
            },
            {
                "floor_id": 5,
                "exit_type": ExitType.emergency_window,
                "side": "left",
                "requires_open_action": True,
            },
        ],
        "elevators": 0,
        "population": {"mobile": 80, "injured": 20, "impaired": 10},
        "weight_lower": False,
    },
}

ROOM_TYPES = ("office", "office", "lab", "hall", "utility", "shelter")


def _generate_floor(
    rng: random.Random,
    floor_id: int,
    num_rooms: int,
    floor_width: int,
    floor_height: int,
) -> Floor:
    """Generate a single floor with rooms and corridors."""
    num_rows = 1 if num_rooms <= 4 else 2
    num_cols = math.ceil(num_rooms / num_rows)
    padding_x = 20
    padding_y = 20
    gap_x = 10
    gap_y = 10

    room_width = (floor_width - (2 * padding_x) - ((num_cols - 1) * gap_x)) // num_cols
    room_height = (floor_height - (2 * padding_y) - ((num_rows - 1) * gap_y)) // num_rows

    rooms: list[Room] = []
    for room_index in range(num_rooms):
        row = room_index // num_cols
        col = room_index % num_cols
        x = padding_x + col * (room_width + gap_x)
        y = padding_y + row * (room_height + gap_y)
        room = Room(
            room_id=f"F{floor_id}_R{room_index}",
            floor_id=floor_id,
            room_type=ROOM_TYPES[rng.randrange(len(ROOM_TYPES))],
            geometry=Rect(x=x, y=y, w=room_width, h=room_height),
        )
        rooms.append(room)

    corridors: list[Corridor] = []
    corridor_index = 0
    adjacency_pairs: list[tuple[int, int]] = []
    for room_index in range(num_rooms):
        row = room_index // num_cols
        col = room_index % num_cols
        right_index = room_index + 1
        down_index = room_index + num_cols
        if col < num_cols - 1 and right_index < num_rooms and (right_index // num_cols) == row:
            adjacency_pairs.append((room_index, right_index))
        if down_index < num_rooms:
            adjacency_pairs.append((room_index, down_index))

    for left_index, right_index in adjacency_pairs:
        left_room = rooms[left_index]
        right_room = rooms[right_index]
        corridors.append(
            Corridor(
                corridor_id=f"F{floor_id}_C{corridor_index}",
                from_node_id=left_room.room_id,
                to_node_id=right_room.room_id,
            )
        )
        left_room.adjacent_node_ids.append(right_room.room_id)
        right_room.adjacent_node_ids.append(left_room.room_id)
        corridor_index += 1

    return Floor(
        floor_id=floor_id,
        width=floor_width,
        height=floor_height,
        rooms=rooms,
        corridors=corridors,
    )


def _select_room_by_position(floor: Floor, position: str) -> Room:
    ordered_rooms = sorted(
        floor.rooms,
        key=lambda room: (room.geometry.x + (room.geometry.w // 2), room.geometry.y, room.room_id),
    )
    if position == "left":
        return ordered_rooms[0]
    if position == "right":
        return ordered_rooms[-1]
    if position == "center":
        return ordered_rooms[len(ordered_rooms) // 2]
    raise ValueError(f"Unknown room position: {position}")


def _select_stairwell_entry_room(floor: Floor, stair_index: int, num_stairwells: int) -> Room:
    ordered_rooms = sorted(
        floor.rooms,
        key=lambda room: (room.geometry.x + (room.geometry.w // 2), room.geometry.y, room.room_id),
    )
    n = len(ordered_rooms)
    if num_stairwells == 1:
        candidate_indices = [n // 2]
    elif num_stairwells == 2:
        candidate_indices = [0, n - 1]
    elif num_stairwells == 3:
        candidate_indices = [0, n // 2, n - 1]
    else:
        # Spread candidates evenly across rooms
        candidate_indices = [
            min(n - 1, int(i * (n - 1) / (num_stairwells - 1)))
            for i in range(num_stairwells)
        ]

    used: set[int] = set()
    resolved: list[int] = []
    for candidate in candidate_indices:
        while candidate in used and candidate < len(ordered_rooms) - 1:
            candidate += 1
        while candidate in used and candidate > 0:
            candidate -= 1
        used.add(candidate)
        resolved.append(candidate)
    return ordered_rooms[resolved[stair_index]]


def _place_stairwells(
    rng: random.Random,
    floors: list[Floor],
    num_stairwells: int,
    blocked_indices: list[int] = [],
) -> list[Stairwell]:
    """Place stairwells connecting floors, optionally pre-blocking some."""
    del rng
    stairwells: list[Stairwell] = []
    floor_ids = [floor.floor_id for floor in floors]
    for stair_index in range(num_stairwells):
        entry_room_ids = {
            floor.floor_id: _select_stairwell_entry_room(floor, stair_index, num_stairwells).room_id
            for floor in floors
        }
        stairwell = Stairwell(
            stairwell_id=f"SW{stair_index}",
            floor_ids=floor_ids,
            blocked=stair_index in blocked_indices,
            entry_room_ids=entry_room_ids,
        )
        stairwells.append(stairwell)
        for floor in floors:
            floor.stairwells.append(stairwell.model_copy(deep=True))
            room = next(room for room in floor.rooms if room.room_id == entry_room_ids[floor.floor_id])
            room.adjacent_node_ids.append(stairwell.stairwell_id)
    return stairwells


def _place_exits(rng: random.Random, floors: list[Floor], exit_specs: list[dict]) -> list[Exit]:
    """Place exits on specified floors adjacent to edge rooms."""
    del rng
    exits: list[Exit] = []
    for exit_index, spec in enumerate(exit_specs):
        floor = floors[int(spec["floor_id"])]
        room = _select_room_by_position(floor, str(spec.get("side", "center")))
        exit_obj = Exit(
            exit_id=f"EX{exit_index}",
            floor_id=floor.floor_id,
            exit_type=spec["exit_type"],
            adjacent_room_id=room.room_id,
            requires_open_action=bool(spec.get("requires_open_action", False)),
        )
        exits.append(exit_obj)
        floor.exits.append(exit_obj)
        room.adjacent_node_ids.append(exit_obj.exit_id)
    return exits


def _elevator_room_index(floor: Floor, elevator_id: str) -> int:
    numeric_id = int(elevator_id.removeprefix("EL"))
    if numeric_id == 0:
        return len(floor.rooms) // 2
    return min(len(floor.rooms) - 1, max(1, (3 * len(floor.rooms)) // 4))


def _place_elevator(rng: random.Random, floors: list[Floor], elevator_id: str) -> Elevator:
    """Place an elevator connecting all floors."""
    del rng
    elevator = Elevator(
        elevator_id=elevator_id,
        floor_ids=[floor.floor_id for floor in floors],
    )
    for floor in floors:
        room = floor.rooms[_elevator_room_index(floor, elevator_id)]
        room.adjacent_node_ids.append(elevator_id)
        floor.elevators.append(elevator.model_copy(deep=True))
    return elevator


def _allocate_counts(total: int, weights: list[int]) -> list[int]:
    assigned = [0] * len(weights)
    if total <= 0:
        return assigned
    total_weight = sum(weights)
    for index, weight in enumerate(weights):
        assigned[index] = (total * weight) // total_weight
    remainder = total - sum(assigned)
    for index in sorted(range(len(weights)), key=lambda idx: (-weights[idx], idx)):
        if remainder == 0:
            break
        assigned[index] += 1
        remainder -= 1
    return assigned


def _choose_room_for_people(
    rng: random.Random,
    rooms: list[Room],
    preferred_ids: set[str] | None = None,
) -> Room:
    preferred_ids = preferred_ids or set()
    eligible_rooms = [room for room in rooms if not preferred_ids or room.room_id in preferred_ids]
    if not eligible_rooms:
        eligible_rooms = rooms
    return rng.choice(eligible_rooms)


def _distribute_population_to_rooms(
    rng: random.Random,
    rooms: list[Room],
    counts: list[int],
    cohort: str,
    preferred_ids: set[str] | None = None,
) -> None:
    for _ in counts:
        room = _choose_room_for_people(rng, rooms, preferred_ids)
        if cohort == "mobile":
            room.occupancy.mobile += 1
        elif cohort == "injured":
            room.occupancy.injured += 1
        else:
            room.occupancy.mobility_impaired += 1


def _distribute_civilians(
    rng: random.Random,
    building: Building,
    total_mobile: int,
    total_injured: int,
    total_impaired: int,
    weight_lower: bool = False,
) -> None:
    """Distribute civilians across rooms. If weight_lower, put more on lower floors."""
    floors = sorted(building.floors, key=lambda floor: floor.floor_id)
    floor_weights = (
        list(range(len(floors), 0, -1))
        if weight_lower
        else [1 for _ in floors]
    )
    mobile_per_floor = _allocate_counts(total_mobile, floor_weights)
    injured_floor_weights = floor_weights if weight_lower else [1 for _ in floors]
    impaired_floor_weights = [1 for _ in floors]
    exits = _collect_exits(building)
    exit_floors = {exit_obj.floor_id for exit_obj in exits}
    refuge_floors = {floor_id for floor_id in exit_floors if floor_id > 0}

    if total_impaired > 0 and (refuge_floors or exit_floors):
        preferred_floors = refuge_floors or exit_floors
        impaired_floor_weights = [
            1 if floor.floor_id in preferred_floors else 0 for floor in floors
        ]
        if sum(impaired_floor_weights) == 0:
            impaired_floor_weights = [1 for _ in floors]

    injured_per_floor = _allocate_counts(total_injured, injured_floor_weights)
    impaired_per_floor = _allocate_counts(total_impaired, impaired_floor_weights)

    exit_adjacent_room_ids = {exit_obj.adjacent_room_id for exit_obj in exits}

    for floor, mobile_count, injured_count, impaired_count in zip(
        floors,
        mobile_per_floor,
        injured_per_floor,
        impaired_per_floor,
        strict=True,
    ):
        _distribute_population_to_rooms(
            rng,
            floor.rooms,
            [1] * mobile_count,
            "mobile",
        )
        _distribute_population_to_rooms(
            rng,
            floor.rooms,
            [1] * injured_count,
            "injured",
        )
        preferred_impaired_rooms = {
            room.room_id for room in floor.rooms if room.room_id in exit_adjacent_room_ids
        }
        _distribute_population_to_rooms(
            rng,
            floor.rooms,
            [1] * impaired_count,
            "impaired",
            preferred_ids=preferred_impaired_rooms,
        )


def _collect_unique_stairwells(building: Building) -> list[Stairwell]:
    stairwell_map: dict[str, Stairwell] = {}
    for floor in building.floors:
        for stairwell in floor.stairwells:
            stairwell_map.setdefault(stairwell.stairwell_id, stairwell)
    return [stairwell_map[key] for key in sorted(stairwell_map)]


def _collect_unique_elevators(building: Building) -> list[Elevator]:
    elevator_map: dict[str, Elevator] = {}
    for floor in building.floors:
        for elevator in floor.elevators:
            elevator_map.setdefault(elevator.elevator_id, elevator)
    return [elevator_map[key] for key in sorted(elevator_map)]


def _collect_exits(building: Building) -> list[Exit]:
    exits: list[Exit] = []
    seen: set[str] = set()
    for floor in building.floors:
        for exit_obj in floor.exits:
            if exit_obj.exit_id not in seen:
                seen.add(exit_obj.exit_id)
                exits.append(exit_obj)
    return exits


def _build_graph_edges(building: Building) -> list[EdgeRef]:
    """Extract all graph edges from corridors, stairwells, and elevators."""
    edges: list[EdgeRef] = []
    seen: set[tuple[str, str, str]] = set()

    def add_edge(from_id: str, to_id: str, edge_type: str) -> None:
        key = (from_id, to_id, edge_type)
        if key not in seen:
            seen.add(key)
            edges.append(EdgeRef(from_id=from_id, to_id=to_id, edge_type=edge_type))

    for floor in building.floors:
        for corridor in floor.corridors:
            add_edge(corridor.from_node_id, corridor.to_node_id, "corridor")
            add_edge(corridor.to_node_id, corridor.from_node_id, "corridor")

    for stairwell in _collect_unique_stairwells(building):
        if stairwell.blocked:
            continue
        ordered_floor_ids = sorted(stairwell.floor_ids)
        for lower_floor_id, upper_floor_id in zip(ordered_floor_ids, ordered_floor_ids[1:], strict=False):
            lower_room_id = stairwell.entry_room_ids[lower_floor_id]
            upper_room_id = stairwell.entry_room_ids[upper_floor_id]
            add_edge(lower_room_id, upper_room_id, "stairwell")
            add_edge(upper_room_id, lower_room_id, "stairwell")

    for elevator in _collect_unique_elevators(building):
        if not elevator.operational:
            continue
        elevator_rooms = sorted(
            (
                room.floor_id,
                room.room_id,
            )
            for floor in building.floors
            for room in floor.rooms
            if elevator.elevator_id in room.adjacent_node_ids
        )
        for (_, lower_room_id), (_, upper_room_id) in zip(elevator_rooms, elevator_rooms[1:], strict=False):
            add_edge(lower_room_id, upper_room_id, "elevator")
            add_edge(upper_room_id, lower_room_id, "elevator")

    return edges


def _validate_connectivity(building: Building) -> bool:
    """BFS from each occupied room to verify at least one exit is reachable."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in building.graph_edges:
        adjacency[edge.from_id].add(edge.to_id)

    exit_nodes = {exit_obj.exit_id for exit_obj in _collect_exits(building) if not exit_obj.blocked}
    for exit_obj in _collect_exits(building):
        if exit_obj.blocked:
            continue
        adjacency[exit_obj.adjacent_room_id].add(exit_obj.exit_id)
        adjacency[exit_obj.exit_id].add(exit_obj.adjacent_room_id)

    room_ids = [room.room_id for floor in building.floors for room in floor.rooms]
    for room_id in room_ids:
        queue = deque([room_id])
        visited = {room_id}
        found_exit = False
        while queue:
            current = queue.popleft()
            if current in exit_nodes:
                found_exit = True
                break
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if not found_exit:
            return False
    return True


def _mark_room_hazard(room: Room, hazard_type: HazardType, severity: float) -> None:
    room.hazard = HazardState(
        hazard_type=hazard_type,
        severity=severity,
        smoke=severity if hazard_type == HazardType.fire else 0.0,
        water_level=severity if hazard_type == HazardType.flood else 0.0,
        structural_integrity=0.7 if hazard_type == HazardType.structural else 1.0,
    )


def _set_disaster_profile(
    rng: random.Random,
    task_id: str,
    building: Building,
    profile: dict[str, object],
) -> None:
    floors = {floor.floor_id: floor for floor in building.floors}

    if task_id == "task_1_fire_easy":
        room = rng.choice(floors[2].rooms)
        _mark_room_hazard(room, HazardType.fire, 0.3)
    elif task_id == "task_2_flood_medium":
        room = rng.choice(floors[0].rooms)
        _mark_room_hazard(room, HazardType.flood, 0.2)
    elif task_id == "task_3_earthquake_hard":
        room = rng.choice(floors[0].rooms)
        room.hazard = HazardState(
            hazard_type=HazardType.structural,
            severity=0.25,
            smoke=0.0,
            water_level=0.0,
            structural_integrity=0.75,
            passable=True,
        )
        all_rooms = [room for floor in building.floors for room in floor.rooms]
        collapse_ratio = float(profile.get("collapse_ratio", 0.2))
        collapse_count = max(1, round(len(all_rooms) * collapse_ratio))
        building.disaster_zones = sorted(
            room.room_id for room in rng.sample(all_rooms, collapse_count)
        )
    elif task_id == "task_4_cascade_hard":
        room = rng.choice(floors[1].rooms)
        _mark_room_hazard(room, HazardType.fire, 0.35)
    # Long-horizon disaster profiles
    elif task_id == "task_lh_fire_easy":
        room = rng.choice(floors[2].rooms)
        _mark_room_hazard(room, HazardType.fire, 0.25)
    elif task_id == "task_lh_flood_medium":
        room = rng.choice(floors[0].rooms)
        _mark_room_hazard(room, HazardType.flood, 0.2)
    elif task_id == "task_lh_cascade_hard":
        room = rng.choice(floors[1].rooms)
        _mark_room_hazard(room, HazardType.fire, 0.3)
    elif task_id == "task_lh_cascade_brutal":
        room = rng.choice(floors[1].rooms)
        _mark_room_hazard(room, HazardType.fire, 0.35)
        all_rooms = [r for f in building.floors for r in f.rooms]
        collapse_count = max(1, round(len(all_rooms) * 0.15))
        building.disaster_zones = sorted(
            r.room_id for r in rng.sample(all_rooms, collapse_count)
        )


def _generate_room_counts(
    rng: random.Random,
    num_floors: int,
    room_range: tuple[int, int],
) -> list[int]:
    minimum, maximum = room_range
    return [rng.randint(minimum, maximum) for _ in range(num_floors)]


def generate_building(task_id: str, seed: int) -> Building:
    """Main entry point. Dispatches to profile-specific builders."""
    task = get_task(task_id)
    profile = PROFILE_CONFIGS[task.building_profile]
    rng = random.Random(seed)

    num_floors = int(profile["floors"])
    room_range = tuple(profile["room_range"])
    room_counts = _generate_room_counts(rng, num_floors, room_range)
    floors = [
        _generate_floor(rng, floor_id, room_counts[floor_id], FLOOR_WIDTH, FLOOR_HEIGHT)
        for floor_id in range(num_floors)
    ]

    _place_stairwells(
        rng,
        floors,
        int(profile["stairwells"]),
        list(profile.get("blocked_stairwells", [])),
    )
    _place_exits(rng, floors, list(profile["exits"]))
    for elevator_index in range(int(profile["elevators"])):
        _place_elevator(rng, floors, f"EL{elevator_index}")

    building = Building(
        building_id=f"{task_id}_{seed}",
        seed=seed,
        floors=floors,
    )
    building.graph_edges = _build_graph_edges(building)

    population = dict(profile["population"])
    _distribute_civilians(
        rng,
        building,
        int(population["mobile"]),
        int(population["injured"]),
        int(population["impaired"]),
        weight_lower=bool(profile.get("weight_lower", False)),
    )
    _set_disaster_profile(rng, task_id, building, profile)

    if not _validate_connectivity(building):
        raise ValueError(f"Generated building for {task_id} is not fully connected")

    for floor in building.floors:
        room_neighbor_counts = {
            room.room_id: sum(
                1 for adjacent_id in room.adjacent_node_ids if adjacent_id.startswith(f"F{floor.floor_id}_R")
            )
            for room in floor.rooms
        }
        if any(count < 1 for count in room_neighbor_counts.values()):
            raise ValueError(f"Generated floor {floor.floor_id} has isolated room geometry")

    return building
