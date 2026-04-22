from __future__ import annotations

from procgen.generator import GeneratedInstance, GeneratorConfig, generate_instance
from procgen.validator import validate
from tests.procgen_helpers import make_five_floor_building, scheduled_room_block


def _instance(building, events=None, *, config: GeneratorConfig | None = None) -> GeneratedInstance:
    return GeneratedInstance(
        building=building,
        scheduled_events=events or [],
        generator_config_hash="fixture",
        config=config or GeneratorConfig(tier="easy", disaster_family="fire"),
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
    assert report.valid is False
    assert any(
        "impossible" in reason.lower() or "blockage" in reason.lower()
        for reason in report.reasons
    )


def test_min_playable_blockage_round_config_default_passes_later_blockage() -> None:
    building = make_five_floor_building(stairwell_floors=(0, 1, 2, 3, 4), exits_on_floors=(0,))
    report = validate(
        _instance(
            building,
            [
                scheduled_room_block(3, "F1_R0"),
                scheduled_room_block(3, "F2_R0"),
                scheduled_room_block(3, "F3_R0"),
                scheduled_room_block(3, "F4_R0"),
            ],
        )
    )

    assert report.earliest_blockage_round == 3
    assert report.valid is True


def test_min_playable_blockage_round_config_override_rejects_later_blockage() -> None:
    building = make_five_floor_building(stairwell_floors=(0, 1, 2, 3, 4), exits_on_floors=(0,))
    config = GeneratorConfig(
        tier="easy",
        disaster_family="fire",
        min_playable_blockage_round=5,
    )
    report = validate(
        _instance(
            building,
            [
                scheduled_room_block(3, "F1_R0"),
                scheduled_room_block(3, "F2_R0"),
                scheduled_room_block(3, "F3_R0"),
                scheduled_room_block(3, "F4_R0"),
            ],
            config=config,
        )
    )

    assert report.earliest_blockage_round == 3
    assert report.valid is False
    assert any("min_playable_blockage_round=5" in reason for reason in report.reasons)


def test_generator_config_hash_changes_when_min_playable_blockage_round_changes() -> None:
    default_instance = generate_instance(42, "easy", "fire")
    stricter_instance = generate_instance(42, "easy", "fire", min_playable_blockage_round=5)

    assert default_instance.generator_config_hash != stricter_instance.generator_config_hash
