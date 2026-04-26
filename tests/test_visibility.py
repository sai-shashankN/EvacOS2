from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import Building, Floor, Occupancy, Rect, Room
from evacos_ma.observability import FloorVisibilityState, VisibilityConfig, build_floor_observation
from evacos_ma.schemas.multi_agent import ActionTypeMA


def _age_snapshot(env: EvacEnvironment, episode_id: str, agent_id: str) -> dict[str, int]:
    return env.build_floor_agent_observation(episode_id, int(agent_id.split("_")[1]), agent_id=agent_id).visibility_age_by_room


def test_fog_of_war_excludes_full_floor_at_reset() -> None:
    env = EvacEnvironment()
    _, observations = env.reset_multi_agent("task_lh_fire_easy", 42)

    floor_obs = observations.floors["floor_2_agent"]
    assert any(age > 0 for age in floor_obs.visibility_age_by_room.values())
    assert len(floor_obs.visible_rooms) < len(floor_obs.visibility_age_by_room)
    assert floor_obs.visible_rooms
    assert floor_obs.visible_civilian_groups


def test_visibility_age_increases_monotonically_for_unseen_room() -> None:
    env = EvacEnvironment()
    episode_id, observations = env.reset_multi_agent("task_lh_fire_easy", 42)
    floor_obs = observations.floors["floor_2_agent"]
    target_room = next(
        room_id for room_id, age in floor_obs.visibility_age_by_room.items() if age > 0
    )

    ages = [_age_snapshot(env, episode_id, "floor_2_agent")[target_room]]
    for _ in range(5):
        env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})
        ages.append(_age_snapshot(env, episode_id, "floor_2_agent")[target_room])

    assert ages == sorted(ages)
    assert ages[-1] >= ages[0]


def test_room_drops_from_visible_rooms_after_staleness_cap_but_age_is_preserved() -> None:
    building = Building(
        building_id="vis-test",
        seed=1,
        floors=[
            Floor(
                floor_id=0,
                rooms=[
                    Room(room_id="F0_R0", floor_id=0, geometry=Rect(x=0, y=0, w=10, h=10), occupancy=Occupancy(), adjacent_node_ids=["F0_R1"]),
                    Room(room_id="F0_R1", floor_id=0, geometry=Rect(x=10, y=0, w=10, h=10), occupancy=Occupancy(), adjacent_node_ids=["F0_R0"]),
                ],
            )
        ],
        graph_edges=[],
    )
    config = VisibilityConfig(base_visibility_radius=0)
    state = FloorVisibilityState(scouted_rooms_this_round={"F0_R0"})
    visible_rooms, _, visibility_age_by_room, _ = build_floor_observation(
        floor_id=0,
        building=building,
        hazard_engine=None,
        vis_state=state,
        current_round=0,
        rng_seed=11,
        config=config,
    )
    assert "F0_R0" in {room.room_id for room in visible_rooms}

    state.scouted_rooms_this_round.clear()
    for round_id in range(1, 10):
        visible_rooms, _, visibility_age_by_room, _ = build_floor_observation(
            floor_id=0,
            building=building,
            hazard_engine=None,
            vis_state=state,
            current_round=round_id,
            rng_seed=11,
            config=config,
        )

    assert "F0_R0" not in {room.room_id for room in visible_rooms}
    assert visibility_age_by_room["F0_R0"] == 8


def test_visibility_age_is_deterministic_across_replays() -> None:
    def run() -> dict[str, int]:
        env = EvacEnvironment()
        episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)
        for _ in range(5):
            env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})
        return _age_snapshot(env, episode_id, "floor_2_agent")

    assert run() == run()
