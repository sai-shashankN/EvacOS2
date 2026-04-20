from __future__ import annotations

from evacos_ma.models import Building, Corridor, Floor, HazardState, HazardType, Occupancy, Rect, Room
from evacos_ma.observability import FloorVisibilityState, VisibilityConfig, build_floor_observation


def _synthetic_building(hazard_severity: float, smoke_level: float) -> Building:
    room_a = Room(
        room_id="F0_R0",
        floor_id=0,
        geometry=Rect(x=0, y=0, w=10, h=10),
        occupancy=Occupancy(mobile=4, injured=1, mobility_impaired=1),
        hazard=HazardState(
            hazard_type=HazardType.fire if hazard_severity > 0 else None,
            severity=hazard_severity,
            smoke=smoke_level,
            passable=True,
        ),
        adjacent_node_ids=["F0_R1"],
    )
    room_b = Room(
        room_id="F0_R1",
        floor_id=0,
        geometry=Rect(x=10, y=0, w=10, h=10),
        occupancy=Occupancy(mobile=2),
        hazard=HazardState(severity=hazard_severity / 2, smoke=smoke_level / 2, passable=True),
        adjacent_node_ids=["F0_R0"],
    )
    corridor = Corridor(
        corridor_id="C0",
        from_node_id="F0_R0",
        to_node_id="F0_R1",
        hazard=HazardState(severity=hazard_severity / 2, smoke=smoke_level / 2, passable=True),
    )
    return Building(
        building_id="synthetic",
        seed=7,
        floors=[Floor(floor_id=0, rooms=[room_a, room_b], corridors=[corridor])],
        graph_edges=[],
    )


def _revealed_state() -> FloorVisibilityState:
    return FloorVisibilityState(scouted_rooms_this_round={"F0_R0"})


def test_sensor_quality_degrades_under_high_hazard() -> None:
    config = VisibilityConfig()
    building = _synthetic_building(0.7, 0.8)
    _, _, _, sensor_quality = build_floor_observation(
        floor_id=0,
        building=building,
        hazard_engine=None,
        vis_state=_revealed_state(),
        current_round=0,
        rng_seed=123,
        config=config,
    )

    assert config.sensor_quality_min < sensor_quality < 1.0


def test_hazard_free_state_has_no_jitter() -> None:
    config = VisibilityConfig()
    building = _synthetic_building(0.0, 0.0)
    visible_rooms, _, _, sensor_quality = build_floor_observation(
        floor_id=0,
        building=building,
        hazard_engine=None,
        vis_state=_revealed_state(),
        current_round=0,
        rng_seed=123,
        config=config,
    )

    room = next(room for room in visible_rooms if room.room_id == "F0_R0")
    assert sensor_quality == 1.0
    assert room.occupancy_mobile == 4
    assert room.occupancy_injured == 1
    assert room.occupancy_mobility_impaired == 1


def test_jitter_is_bounded_by_sensor_quality() -> None:
    config = VisibilityConfig()
    building = _synthetic_building(0.7, 0.8)
    visible_rooms, _, _, sensor_quality = build_floor_observation(
        floor_id=0,
        building=building,
        hazard_engine=None,
        vis_state=_revealed_state(),
        current_round=0,
        rng_seed=123,
        config=config,
    )

    max_jitter = round(
        config.occupancy_jitter_at_min
        * (1 - sensor_quality)
        / (1 - config.sensor_quality_min)
    )
    ground_truth = {"F0_R0": 4, "F0_R1": 2}
    for room in visible_rooms:
        assert abs(room.occupancy_mobile - ground_truth[room.room_id]) <= max_jitter


def test_degraded_sensor_jitter_is_deterministic() -> None:
    config = VisibilityConfig()
    building = _synthetic_building(0.7, 0.8)

    def snapshot():
        visible_rooms, _, _, _ = build_floor_observation(
            floor_id=0,
            building=building,
            hazard_engine=None,
            vis_state=_revealed_state(),
            current_round=0,
            rng_seed=123,
            config=config,
        )
        return [(room.room_id, room.occupancy_mobile, room.hazard_severity, room.smoke_level) for room in visible_rooms]

    assert snapshot() == snapshot()
