from __future__ import annotations

import pytest

from evacos_ma.building_gen import generate_building
from evacos_ma.disaster import (
    FireSpread,
    FloodRise,
    GasLeak,
    MultiCascade,
    StructuralDamage,
)
from evacos_ma.models import EventType, HazardState, HazardType, Occupancy, ScheduledEvent


def _all_rooms(building):
    return [room for floor in building.floors for room in floor.rooms]


def _room_map(building):
    return {room.room_id: room for room in _all_rooms(building)}


def _unique_stairwells(building):
    stairwells = {}
    for floor in building.floors:
        for stairwell in floor.stairwells:
            stairwells.setdefault(stairwell.stairwell_id, stairwell)
    return stairwells


def _clear_hazards(building) -> None:
    for room in _all_rooms(building):
        room.hazard = HazardState()
    for floor in building.floors:
        for corridor in floor.corridors:
            corridor.hazard = HazardState()


def _find_stairwell_entry(building) -> tuple[str, str]:
    stairwell_id, stairwell = sorted(_unique_stairwells(building).items())[0]
    highest_floor = max(stairwell.entry_room_ids)
    return stairwell_id, stairwell.entry_room_ids[highest_floor]


def _count_submerged_floors(building) -> int:
    submerged = 0
    for floor in building.floors:
        if all(
            room.hazard.hazard_type == HazardType.flood
            and room.hazard.severity == pytest.approx(1.0)
            and not room.hazard.passable
            for room in floor.rooms
        ):
            submerged += 1
    return submerged


def _find_corridor_origin(building) -> tuple[str, str, str]:
    floor = building.floors[0]
    corridor = floor.corridors[0]
    return corridor.from_node_id, corridor.to_node_id, corridor.corridor_id


def test_fire_spread_rate() -> None:
    building = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building)
    _, origin_room_id = _find_stairwell_entry(building)
    engine = FireSpread(building, origin_room_id, seed=7)

    burning_counts = [len(engine.get_hazard_map())]
    event_counts = []
    for step in range(1, 7):
        events = engine.advance(step)
        burning_counts.append(len(engine.get_hazard_map()))
        event_counts.append(len(events))

    assert burning_counts == [1, 1, 1, 1, 2, 2, 2]
    assert event_counts == [0, 0, 0, 1, 0, 0]


def test_fire_severity_increase() -> None:
    building = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building)
    _, origin_room_id = _find_stairwell_entry(building)
    engine = FireSpread(building, origin_room_id, seed=7)

    assert building.floors[2].rooms
    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.25)
    engine.advance(1)
    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.4)
    engine.advance(2)
    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.55)


def test_fire_blocks_stairwell() -> None:
    building = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building)
    stairwell_id, origin_room_id = _find_stairwell_entry(building)
    engine = FireSpread(building, origin_room_id, seed=7)

    for step in range(1, 4):
        engine.advance(step)
    assert _unique_stairwells(building)[stairwell_id].blocked is False

    engine.advance(4)
    assert _unique_stairwells(building)[stairwell_id].blocked is True


def test_fire_room_impassable() -> None:
    building = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building)
    _, origin_room_id = _find_stairwell_entry(building)
    engine = FireSpread(building, origin_room_id, seed=7)

    for step in range(1, 5):
        engine.advance(step)
    assert _room_map(building)[origin_room_id].hazard.passable is True
    engine.advance(5)
    assert _room_map(building)[origin_room_id].hazard.passable is False


def test_flood_rise_rate() -> None:
    building = generate_building("task_2_flood_medium", 42)
    engine = FloodRise(building, origin_room_id=building.floors[0].rooms[0].room_id, seed=11)

    assert _count_submerged_floors(building) == 0
    assert all(
        room.hazard.hazard_type == HazardType.flood
        and room.hazard.severity == pytest.approx(0.3)
        and room.hazard.passable is True
        for room in building.floors[0].rooms
    )
    for step in range(1, 5):
        engine.advance(step)
    assert _count_submerged_floors(building) == 0

    engine.advance(5)
    assert _count_submerged_floors(building) == 1

    for step in range(6, 10):
        engine.advance(step)
    engine.advance(10)
    assert _count_submerged_floors(building) == 2


def test_flood_keeps_upper_stairwells_usable() -> None:
    building = generate_building("task_2_flood_medium", 42)
    FloodRise(building, origin_room_id=building.floors[0].rooms[0].room_id, seed=11)

    assert any(not stairwell.blocked for stairwell in _unique_stairwells(building).values())


def test_flood_profile_has_no_elevators() -> None:
    building = generate_building("task_2_flood_medium", 42)
    assert all(not floor.elevators for floor in building.floors)


def test_gas_spread_corridors_first() -> None:
    building = generate_building("task_4_cascade_hard", 42)
    _clear_hazards(building)
    origin_room_id, other_room_id, corridor_id = _find_corridor_origin(building)
    engine = GasLeak(building, origin_room_id, seed=5)

    engine.advance(1)
    assert building.floors[0].corridors[0].hazard.severity == pytest.approx(0.0)

    engine.advance(2)
    assert building.floors[0].corridors[0].hazard.severity == pytest.approx(0.0)

    engine.advance(3)
    assert building.floors[0].corridors[0].hazard.hazard_type == HazardType.gas
    assert building.floors[0].corridors[0].hazard.severity > 0.0
    assert _room_map(building)[other_room_id].hazard.severity == pytest.approx(0.0)


def test_gas_severity_rate() -> None:
    building = generate_building("task_4_cascade_hard", 42)
    _clear_hazards(building)
    origin_room_id, _, _ = _find_corridor_origin(building)
    engine = GasLeak(building, origin_room_id, seed=5)

    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.2)
    engine.advance(1)
    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.3)
    engine.advance(2)
    assert _room_map(building)[origin_room_id].hazard.severity == pytest.approx(0.4)


def test_gas_fire_explosion() -> None:
    building = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building)
    origin_room_id = building.floors[2].rooms[0].room_id
    _room_map(building)[origin_room_id].hazard = HazardState(
        hazard_type=HazardType.fire,
        severity=0.6,
        smoke=0.2,
    )
    engine = GasLeak(building, origin_room_id, seed=13)

    assert engine.advance(1) == []
    events = engine.advance(2)
    room = _room_map(building)[origin_room_id]

    assert any(event.event_type.value == "explosion" for event in events)
    assert room.hazard.severity == pytest.approx(1.0)
    assert room.hazard.structural_integrity == pytest.approx(0.0)
    assert room.hazard.passable is False


def test_structural_collapse_rate() -> None:
    building = generate_building("task_3_earthquake_hard", 42)
    risk_rooms = sorted(building.disaster_zones)[:3]
    engine = StructuralDamage(building, origin_room_id=risk_rooms[0], seed=17, disaster_zones=risk_rooms)
    initial_structural = sum(
        1 for room in _all_rooms(building) if room.hazard.hazard_type == HazardType.structural
    )

    for step in range(1, 8):
        assert engine.advance(step) == []
    first_events = engine.advance(8)
    collapsed_after_8 = sum(
        1 for room in _all_rooms(building) if room.hazard.hazard_type == HazardType.structural
    )
    second_events = []
    for step in range(9, 16):
        second_events = engine.advance(step)
    second_events = engine.advance(16)
    collapsed_after_16 = sum(
        1 for room in _all_rooms(building) if room.hazard.hazard_type == HazardType.structural
    )

    assert len(first_events) == 1
    assert collapsed_after_8 == initial_structural + 1
    assert len(second_events) == 1
    assert collapsed_after_16 == initial_structural + 2


def test_structural_stairwell_block() -> None:
    building = generate_building("task_3_earthquake_hard", 42)
    stairwell_id, stairwell = sorted(_unique_stairwells(building).items())[0]
    risk_rooms = [stairwell.entry_room_ids[floor_id] for floor_id in sorted(stairwell.entry_room_ids)[:2]]
    engine = StructuralDamage(building, origin_room_id=risk_rooms[0], seed=17, disaster_zones=risk_rooms)

    engine.advance(8)
    assert _unique_stairwells(building)[stairwell_id].blocked is False
    engine.advance(16)
    assert _unique_stairwells(building)[stairwell_id].blocked is True


def test_multi_cascade_timed_events() -> None:
    building = generate_building("task_4_cascade_hard", 42)
    origin_room_id = building.floors[1].rooms[0].room_id
    engine = MultiCascade(building, origin_room_id, seed=23)

    for step in range(1, 10):
        engine.advance(step)
    assert "gas" not in engine._active_engines

    engine.advance(10)
    assert "gas" in engine._active_engines
    floor_3_rooms = [room for room in _all_rooms(building) if room.floor_id == 3]
    assert any(room.hazard.hazard_type in {HazardType.gas, HazardType.structural} for room in floor_3_rooms)

    blocked_before = sum(stairwell.blocked for stairwell in _unique_stairwells(building).values())
    for step in range(11, 15):
        engine.advance(step)
    engine.advance(15)
    blocked_after = sum(stairwell.blocked for stairwell in _unique_stairwells(building).values())

    assert "structural" in engine._active_engines
    assert blocked_after >= blocked_before + 1


def test_multi_cascade_preserves_scheduled_event_id() -> None:
    building = generate_building("task_lh_cascade_hard", 42)
    origin_room_id = building.floors[1].rooms[0].room_id
    gas_origin = building.floors[-1].rooms[0].room_id
    engine = MultiCascade(building, origin_room_id, seed=23)
    scheduled_event = ScheduledEvent(
        event_id="cascade_gas_leak",
        trigger_step=80,
        event_type=EventType.gas_rupture,
        target_id=gas_origin,
        payload={"origin_room_id": gas_origin},
    )

    events = engine._activate_event(scheduled_event, step=80)

    assert len(events) == 1
    assert events[0].event_id == scheduled_event.event_id
    assert events[0].target_id == gas_origin


def test_multi_cascade_stairwell_noop_still_emits_summary() -> None:
    building = generate_building("task_lh_cascade_hard", 42)
    origin_room_id = building.floors[1].rooms[0].room_id
    engine = MultiCascade(building, origin_room_id, seed=23)

    for stairwell_id in list(engine._stairwell_status):
        engine._set_stairwell_blocked(stairwell_id, True)

    scheduled_event = ScheduledEvent(
        event_id="cascade_structural_collapse",
        trigger_step=160,
        event_type=EventType.stairwell_collapse,
        target_id="structural_activation",
        payload={"floor_id": 3},
    )

    events = engine._activate_event(scheduled_event, step=160)

    assert len(events) == 1
    assert events[0].event_id == scheduled_event.event_id
    assert events[0].event_type == EventType.stairwell_collapse
    assert "already blocked" in events[0].description


def test_deterministic_replay() -> None:
    building_a = generate_building("task_1_fire_easy", 42)
    building_b = generate_building("task_1_fire_easy", 42)
    _clear_hazards(building_a)
    _clear_hazards(building_b)
    _, origin_room_id = _find_stairwell_entry(building_a)
    engine_a = FireSpread(building_a, origin_room_id, seed=29)
    engine_b = FireSpread(building_b, origin_room_id, seed=29)

    trace_a = []
    trace_b = []
    for step in range(1, 7):
        events_a = [event.model_dump() for event in engine_a.advance(step)]
        events_b = [event.model_dump() for event in engine_b.advance(step)]
        trace_a.append((engine_a.get_hazard_map(), events_a))
        trace_b.append((engine_b.get_hazard_map(), events_b))

    assert trace_a == trace_b
