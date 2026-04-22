"""Seeded procedural generator for 5-floor Building instances.

Produces deterministically-constructed buildings aligned to a tier x disaster_family
grid, with staged cascade schedules for hazard progression.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict, deque
from typing import Literal

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
    EventType,
    Occupancy,
    Rect,
    Room,
    ScheduledEvent,
    Stairwell,
)

GENERATOR_CONFIG_VERSION = "2026.04.22"
Tier = Literal["easy", "medium", "hard", "brutal"]

# ---------------------------------------------------------------------------
# Tier knob table
# ---------------------------------------------------------------------------
TIER_KNOBS: dict[str, dict] = {
    "easy": {
        "rooms_per_floor": (6, 8),
        "stairwells": 3,
        "elevators": 1,
        "exits": 3,
        "redundancy": 1.5,
        "impaired_frac": 0.05,
        "cascade_aggression": 0.25,
    },
    "medium": {
        "rooms_per_floor": (5, 7),
        "stairwells": 2,
        "elevators": 1,
        "exits": 2,
        "redundancy": 1.2,
        "impaired_frac": 0.15,
        "cascade_aggression": 0.50,
    },
    "hard": {
        "rooms_per_floor": (4, 6),
        "stairwells": 2,
        "elevators": 0,
        "exits": 2,
        "redundancy": 1.0,
        "impaired_frac": 0.25,
        "cascade_aggression": 0.75,
    },
    "brutal": {
        "rooms_per_floor": (3, 5),
        "stairwells": 1,
        "elevators": 0,
        "exits": 1,
        "redundancy": 0.8,
        "impaired_frac": 0.35,
        "cascade_aggression": 1.00,
    },
}

FLOOR_WIDTH = 800
FLOOR_HEIGHT = 400
NUM_FLOORS = 5

ROOM_TYPES = ("office", "office", "lab", "hall", "utility", "shelter")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
import dataclasses


@dataclasses.dataclass(frozen=True)
class GeneratorConfig:
    tier: Tier
    disaster_family: str  # DisasterType.value
    config_version: str = GENERATOR_CONFIG_VERSION
    min_playable_blockage_round: int = 1

    @property
    def knobs(self) -> dict:
        return TIER_KNOBS[self.tier]


@dataclasses.dataclass(frozen=True)
class GeneratedInstance:
    building: Building
    scheduled_events: list[ScheduledEvent]
    generator_config_hash: str
    config: GeneratorConfig
    seed: int


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _compute_config_hash(
    tier: str,
    disaster_family: str,
    config_version: str,
    min_playable_blockage_round: int,
) -> str:
    knobs = TIER_KNOBS[tier]
    canonical = {
        "tier": tier,
        "disaster_family": disaster_family,
        "config_version": config_version,
        "min_playable_blockage_round": min_playable_blockage_round,
        "knobs": knobs,
    }
    blob = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Floor / Room / Corridor generation
# ---------------------------------------------------------------------------
def _generate_room_counts(rng: random.Random, room_range: tuple[int, int]) -> list[int]:
    lo, hi = room_range
    return [rng.randint(lo, hi) for _ in range(NUM_FLOORS)]


def _generate_floor(
    rng: random.Random,
    floor_id: int,
    num_rooms: int,
    redundancy: float,
) -> Floor:
    num_rows = 1 if num_rooms <= 4 else 2
    num_cols = math.ceil(num_rooms / num_rows)
    padding_x = 20
    padding_y = 20
    gap_x = 10
    gap_y = 10

    room_width = (FLOOR_WIDTH - 2 * padding_x - (num_cols - 1) * gap_x) // num_cols
    room_height = (FLOOR_HEIGHT - 2 * padding_y - (num_rows - 1) * gap_y) // num_rows

    rooms: list[Room] = []
    for ri in range(num_rooms):
        row = ri // num_cols
        col = ri % num_cols
        x = padding_x + col * (room_width + gap_x)
        y = padding_y + row * (room_height + gap_y)
        room = Room(
            room_id=f"F{floor_id}_R{ri}",
            floor_id=floor_id,
            room_type=ROOM_TYPES[rng.randrange(len(ROOM_TYPES))],
            geometry=Rect(x=x, y=y, w=room_width, h=room_height),
        )
        rooms.append(room)

    corridors: list[Corridor] = []
    ci = 0
    # Grid edges (horizontal + vertical)
    adj_pairs: list[tuple[int, int]] = []
    for ri in range(num_rooms):
        row = ri // num_cols
        col = ri % num_cols
        right = ri + 1
        down = ri + num_cols
        if col < num_cols - 1 and right < num_rooms and (right // num_cols) == row:
            adj_pairs.append((ri, right))
        if down < num_rooms:
            adj_pairs.append((ri, down))

    # Extra corridor edges based on redundancy (>1.0 adds diagonal / skip connections)
    if redundancy > 1.0 and num_rooms > 3:
        extra_target = max(1, int((redundancy - 1.0) * num_rooms))
        added = 0
        for attempt in range(extra_target * 10):
            if added >= extra_target:
                break
            a = rng.randint(0, num_rooms - 1)
            b = rng.randint(0, num_rooms - 1)
            if a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair not in adj_pairs:
                adj_pairs.append(pair)
                added += 1

    for left_idx, right_idx in adj_pairs:
        left_room = rooms[left_idx]
        right_room = rooms[right_idx]
        corridors.append(
            Corridor(
                corridor_id=f"F{floor_id}_C{ci}",
                from_node_id=left_room.room_id,
                to_node_id=right_room.room_id,
            )
        )
        left_room.adjacent_node_ids.append(right_room.room_id)
        right_room.adjacent_node_ids.append(left_room.room_id)
        ci += 1

    return Floor(
        floor_id=floor_id,
        width=FLOOR_WIDTH,
        height=FLOOR_HEIGHT,
        rooms=rooms,
        corridors=corridors,
    )


# ---------------------------------------------------------------------------
# Stairwell placement
# ---------------------------------------------------------------------------
def _select_stairwell_entry_room(floor: Floor, stair_index: int, num_stairwells: int) -> Room:
    ordered = sorted(
        floor.rooms,
        key=lambda r: (r.geometry.x + r.geometry.w // 2, r.geometry.y, r.room_id),
    )
    n = len(ordered)
    if num_stairwells == 1:
        idx = n // 2
    elif num_stairwells == 2:
        idx = 0 if stair_index == 0 else n - 1
    elif num_stairwells == 3:
        idx = [0, n // 2, n - 1][stair_index]
    else:
        idx = min(n - 1, int(stair_index * (n - 1) / max(1, num_stairwells - 1)))
    return ordered[idx]


def _place_stairwells(rng: random.Random, floors: list[Floor], count: int) -> None:
    floor_ids = [f.floor_id for f in floors]
    for si in range(count):
        entry_room_ids = {
            fl.floor_id: _select_stairwell_entry_room(fl, si, count).room_id
            for fl in floors
        }
        sw = Stairwell(
            stairwell_id=f"SW{si}",
            floor_ids=floor_ids,
            blocked=False,
            entry_room_ids=entry_room_ids,
        )
        for fl in floors:
            fl.stairwells.append(sw.model_copy(deep=True))
            room = next(r for r in fl.rooms if r.room_id == entry_room_ids[fl.floor_id])
            room.adjacent_node_ids.append(sw.stairwell_id)


# ---------------------------------------------------------------------------
# Elevator placement
# ---------------------------------------------------------------------------
def _place_elevators(rng: random.Random, floors: list[Floor], count: int) -> None:
    for ei in range(count):
        elevator_id = f"EL{ei}"
        ev = Elevator(
            elevator_id=elevator_id,
            floor_ids=[f.floor_id for f in floors],
        )
        for fl in floors:
            idx = len(fl.rooms) // 2
            room = fl.rooms[idx]
            room.adjacent_node_ids.append(elevator_id)
            fl.elevators.append(ev.model_copy(deep=True))


# ---------------------------------------------------------------------------
# Exit placement
# ---------------------------------------------------------------------------
def _select_room_by_position(floor: Floor, position: str) -> Room:
    ordered = sorted(
        floor.rooms,
        key=lambda r: (r.geometry.x + r.geometry.w // 2, r.geometry.y, r.room_id),
    )
    if position == "left":
        return ordered[0]
    if position == "right":
        return ordered[-1]
    return ordered[len(ordered) // 2]


def _place_exits(rng: random.Random, floors: list[Floor], count: int, tier: str) -> None:
    # Ground floor gets most exits; if count > 2, top floor gets one
    sides = ["left", "center", "right"]
    exit_idx = 0
    # Place exits on ground floor first
    ground = floors[0]
    for i in range(min(count, 3)):
        room = _select_room_by_position(ground, sides[i % len(sides)])
        ex = Exit(
            exit_id=f"EX{exit_idx}",
            floor_id=0,
            exit_type=ExitType.ground,
            adjacent_room_id=room.room_id,
        )
        ground.exits.append(ex)
        room.adjacent_node_ids.append(ex.exit_id)
        exit_idx += 1

    if count > 3:
        # Place remaining exits on top floor
        top = floors[-1]
        for i in range(count - 3):
            room = _select_room_by_position(top, sides[i % len(sides)])
            ex = Exit(
                exit_id=f"EX{exit_idx}",
                floor_id=floors[-1].floor_id,
                exit_type=ExitType.rooftop,
                adjacent_room_id=room.room_id,
            )
            top.exits.append(ex)
            room.adjacent_node_ids.append(ex.exit_id)
            exit_idx += 1


# ---------------------------------------------------------------------------
# Graph edges builder
# ---------------------------------------------------------------------------
def _build_graph_edges(building: Building) -> list[EdgeRef]:
    edges: list[EdgeRef] = []
    seen: set[tuple[str, str, str]] = set()

    def add_edge(from_id: str, to_id: str, etype: str) -> None:
        key = (from_id, to_id, etype)
        if key not in seen:
            seen.add(key)
            edges.append(EdgeRef(from_id=from_id, to_id=to_id, edge_type=etype))

    # Corridors (bidirectional)
    for fl in building.floors:
        for c in fl.corridors:
            add_edge(c.from_node_id, c.to_node_id, "corridor")
            add_edge(c.to_node_id, c.from_node_id, "corridor")

    # Stairwells
    sw_map: dict[str, Stairwell] = {}
    for fl in building.floors:
        for sw in fl.stairwells:
            sw_map.setdefault(sw.stairwell_id, sw)
    for sw in sorted(sw_map.values(), key=lambda s: s.stairwell_id):
        if sw.blocked:
            continue
        ordered = sorted(sw.floor_ids)
        for lo, hi in zip(ordered, ordered[1:]):
            lo_room = sw.entry_room_ids[lo]
            hi_room = sw.entry_room_ids[hi]
            add_edge(lo_room, hi_room, "stairwell")
            add_edge(hi_room, lo_room, "stairwell")

    # Elevators
    ev_map: dict[str, Elevator] = {}
    for fl in building.floors:
        for ev in fl.elevators:
            ev_map.setdefault(ev.elevator_id, ev)
    for ev in sorted(ev_map.values(), key=lambda e: e.elevator_id):
        if not ev.operational:
            continue
        rooms_sorted = sorted(
            (r.floor_id, r.room_id)
            for fl in building.floors
            for r in fl.rooms
            if ev.elevator_id in r.adjacent_node_ids
        )
        for (_, lo_room), (_, hi_room) in zip(rooms_sorted, rooms_sorted[1:]):
            add_edge(lo_room, hi_room, "elevator")
            add_edge(hi_room, lo_room, "elevator")

    return edges


# ---------------------------------------------------------------------------
# Population distribution
# ---------------------------------------------------------------------------
def _allocate_counts(total: int, weights: list[int]) -> list[int]:
    assigned = [0] * len(weights)
    if total <= 0:
        return assigned
    tw = sum(weights)
    if tw == 0:
        return assigned
    for i, w in enumerate(weights):
        assigned[i] = (total * w) // tw
    remainder = total - sum(assigned)
    for i in sorted(range(len(weights)), key=lambda idx: (-weights[idx], idx)):
        if remainder == 0:
            break
        assigned[i] += 1
        remainder -= 1
    return assigned


def _distribute_civilians(
    rng: random.Random,
    building: Building,
    total_mobile: int,
    total_injured: int,
    total_impaired: int,
) -> None:
    floors = sorted(building.floors, key=lambda f: f.floor_id)
    weights = [1 for _ in floors]

    mobile_per = _allocate_counts(total_mobile, weights)
    injured_per = _allocate_counts(total_injured, weights)
    impaired_per = _allocate_counts(total_impaired, weights)

    for fl, mc, ic, imc in zip(floors, mobile_per, injured_per, impaired_per, strict=True):
        rooms = fl.rooms
        for _ in range(mc):
            r = rng.choice(rooms)
            r.occupancy.mobile += 1
        for _ in range(ic):
            r = rng.choice(rooms)
            r.occupancy.injured += 1
        for _ in range(imc):
            r = rng.choice(rooms)
            r.occupancy.mobility_impaired += 1


# ---------------------------------------------------------------------------
# Initial hazards
# ---------------------------------------------------------------------------
def _mark_room_hazard(room: Room, htype: HazardType, severity: float) -> None:
    room.hazard = HazardState(
        hazard_type=htype,
        severity=severity,
        smoke=severity if htype == HazardType.fire else 0.0,
        water_level=severity if htype == HazardType.flood else 0.0,
        structural_integrity=0.7 if htype == HazardType.structural else 1.0,
    )


_HAZARD_MAP = {
    "fire": HazardType.fire,
    "flood": HazardType.flood,
    "gas": HazardType.gas,
    "structural": HazardType.structural,
    "active_threat": HazardType.threat,
}


def _place_initial_hazard(
    rng: random.Random,
    building: Building,
    disaster_family: str,
    tier: str,
) -> None:
    floors_map = {f.floor_id: f for f in building.floors}

    if disaster_family == "multi_cascade":
        # Mix two hazard types: place one fire room and one gas room
        htypes = [HazardType.fire, HazardType.gas]
    else:
        htypes = [_HAZARD_MAP[disaster_family]]

    severity_base = {"easy": 0.2, "medium": 0.3, "hard": 0.4, "brutal": 0.5}.get(tier, 0.3)

    for htype in htypes:
        if disaster_family == "active_threat":
            # Pick interior room on an upper floor for lockdown
            target_floor = floors_map.get(rng.randint(2, 4), building.floors[-1])
        elif htype == HazardType.flood:
            target_floor = floors_map[0]
        elif htype == HazardType.gas:
            target_floor = floors_map.get(rng.randint(3, 4), building.floors[-1])
        else:
            target_floor = floors_map.get(rng.randint(1, 3), building.floors[len(building.floors) // 2])

        room = rng.choice(target_floor.rooms)
        _mark_room_hazard(room, htype, severity_base)

    if disaster_family == "structural":
        all_rooms = [r for f in building.floors for r in f.rooms]
        collapse_count = max(1, round(len(all_rooms) * 0.15))
        building.disaster_zones = sorted(
            r.room_id for r in rng.sample(all_rooms, collapse_count)
        )


# ---------------------------------------------------------------------------
# Cascade schedule builder
# ---------------------------------------------------------------------------
def _build_cascade_schedule(
    rng: random.Random,
    building: Building,
    disaster_family: str,
    aggression: float,
) -> list[ScheduledEvent]:
    """Build staged cascade events from the disaster family and aggression level."""
    events: list[ScheduledEvent] = []
    all_rooms = sorted(
        [r for f in building.floors for r in f.rooms],
        key=lambda r: r.room_id,
    )

    # Number of waves: 2-6 based on aggression
    num_waves = max(2, round(2 + 4 * aggression))
    # Interarrival: higher aggression -> earlier waves
    base_interval = max(3, round(30 * (1 - aggression) + 5))

    kind_to_event_type = {
        "fire": EventType.fire_ignition,
        "flood": EventType.flood_rise,
        "gas": EventType.gas_rupture,
        "structural": EventType.structural_collapse,
        "active_threat": EventType.threat_move,
    }

    if disaster_family == "multi_cascade":
        hazard_kinds = ["fire", "gas"]
    elif disaster_family == "active_threat":
        hazard_kinds = ["active_threat"]
    else:
        hazard_kinds = [disaster_family]

    event_counter = 0
    for wave_idx in range(num_waves):
        trigger_step = base_interval * (wave_idx + 1) + rng.randint(0, max(1, base_interval // 2))
        kind = hazard_kinds[wave_idx % len(hazard_kinds)]
        etype = kind_to_event_type.get(kind, EventType.fire_ignition)
        target_room = rng.choice(all_rooms)

        severity = round(0.2 + 0.3 * (wave_idx / max(1, num_waves - 1)), 3)

        events.append(
            ScheduledEvent(
                event_id=f"procgen_cascade_{event_counter}",
                trigger_step=trigger_step,
                event_type=etype,
                target_id=target_room.room_id,
                payload={
                    "origin_room_id": target_room.room_id,
                    "severity": severity,
                    "wave": wave_idx,
                },
            )
        )
        event_counter += 1

    return events


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------
def generate_instance(
    seed: int,
    tier: Tier,
    disaster_family: "str",
    config_version: str = GENERATOR_CONFIG_VERSION,
    min_playable_blockage_round: int = 1,
) -> GeneratedInstance:
    """Generate a deterministic Building + ScheduledEvent set from (seed, tier, disaster_family)."""
    from evacos_ma.models import DisasterType

    # Validate disaster_family
    df_enum = DisasterType(disaster_family) if not isinstance(disaster_family, DisasterType) else disaster_family
    df_str = df_enum.value

    knobs = TIER_KNOBS[tier]
    config = GeneratorConfig(
        tier=tier,
        disaster_family=df_str,
        config_version=config_version,
        min_playable_blockage_round=min_playable_blockage_round,
    )
    config_hash = _compute_config_hash(
        tier,
        df_str,
        config_version,
        min_playable_blockage_round,
    )

    rng = random.Random(seed)

    # 1. Generate floors with rooms and corridors
    room_counts = _generate_room_counts(rng, knobs["rooms_per_floor"])
    floors = [
        _generate_floor(rng, fid, room_counts[fid], knobs["redundancy"])
        for fid in range(NUM_FLOORS)
    ]

    # 2. Place stairwells
    _place_stairwells(rng, floors, knobs["stairwells"])

    # 3. Place elevators
    _place_elevators(rng, floors, knobs["elevators"])

    # 4. Place exits
    _place_exits(rng, floors, knobs["exits"], tier)

    # 5. Assemble building
    building = Building(
        building_id=f"procgen_{tier}_{df_str}_{seed}",
        seed=seed,
        floors=floors,
    )
    building.graph_edges = _build_graph_edges(building)

    # 6. Population
    total_pop = rng.randint(30, 80)
    impaired_each = round(total_pop * knobs["impaired_frac"] / 2)
    injured = impaired_each
    impaired = impaired_each
    mobile = total_pop - injured - impaired
    if mobile < 0:
        mobile = 0
        injured = min(injured, total_pop)
        impaired = total_pop - injured

    _distribute_civilians(rng, building, mobile, injured, impaired)

    # 7. Initial hazards
    _place_initial_hazard(rng, building, df_str, tier)

    # 8. Cascade schedule
    scheduled_events = _build_cascade_schedule(
        rng, building, df_str, knobs["cascade_aggression"]
    )

    return GeneratedInstance(
        building=building,
        scheduled_events=scheduled_events,
        generator_config_hash=config_hash,
        config=config,
        seed=seed,
    )
