from __future__ import annotations

from procgen.generator import GeneratedInstance, GeneratorConfig
from procgen.validator import validate
from tests.procgen_helpers import make_five_floor_building, scheduled_room_block


def _instance(building, events=None) -> GeneratedInstance:
    return GeneratedInstance(
        building=building,
        scheduled_events=events or [],
        generator_config_hash="fixture",
        config=GeneratorConfig(tier="easy", disaster_family="fire"),
        seed=0,
    )


def test_rejects_floor_disconnected_from_inter_floor_sink_graph() -> None:
    building = make_five_floor_building(stairwell_floors=(0, 1, 2, 3), exits_on_floors=(0,))
    report = validate(_instance(building))
    assert report.valid is False
    assert any("Floor 4 is disconnected" in reason for reason in report.reasons)


def test_rejects_when_only_exit_floor_is_unreachable_from_upper_floors() -> None:
    building = make_five_floor_building(stairwell_floors=(2, 3, 4), exits_on_floors=(0,))
    report = validate(_instance(building))
    assert report.valid is False
    assert any("Floor 2 is disconnected" in reason for reason in report.reasons)


def test_rejects_when_room_has_no_path_to_exit() -> None:
    building = make_five_floor_building(stairwell_floors=(), exits_on_floors=(0,))
    report = validate(_instance(building))
    assert report.valid is False
    assert any("Room F1_R0 on floor 1 has no path to exit" in reason for reason in report.reasons)


def test_reports_earliest_blockage_round_for_collapse_schedule() -> None:
    building = make_five_floor_building(stairwell_floors=(0, 1, 2, 3, 4), exits_on_floors=(0,))
    report = validate(
        _instance(
            building,
            [
                scheduled_room_block(1, "F1_R0"),
                scheduled_room_block(1, "F2_R0"),
                scheduled_room_block(1, "F3_R0"),
                scheduled_room_block(1, "F4_R0"),
            ],
        )
    )
    assert report.earliest_blockage_round == 1
