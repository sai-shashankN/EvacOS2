from __future__ import annotations

import re

import pytest

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import (
    ActionType,
    EvacuateFloorAction,
    LockdownRoomAction,
    Occupancy,
    RouteCiviliansAction,
    StateView,
    TerminationReason,
    WaitAction,
)


def _room_by_id(ep, room_id: str):
    for floor in ep.building.floors:
        for room in floor.rooms:
            if room.room_id == room_id:
                return room
    raise AssertionError(f"Missing room {room_id}")


def _normalized_observation(observation):
    data = observation.model_dump(mode="json")
    data["episode_id"] = "<episode>"
    return data


def _reward_component_sum(reward) -> float:
    return (
        reward.civilians_saved_delta
        + reward.civilians_lost_delta
        + reward.hazard_avoidance_bonus
        + reward.vulnerable_group_bonus
        + reward.efficiency_bonus
        + reward.invalid_action_penalty
        + reward.idle_penalty
        + reward.completion_bonus
    )


def test_reset_creates_episode() -> None:
    env = EvacEnvironment()

    episode_id, observation = env.reset("task_1_fire_easy", 42)

    assert re.fullmatch(r"[0-9a-f]{32}", episode_id)
    assert observation.episode_id == episode_id
    assert observation.task_id == "task_1_fire_easy"
    assert observation.step == 0
    assert observation.summary.total_civilians == 15


def test_reset_deterministic() -> None:
    env = EvacEnvironment()

    episode_a, observation_a = env.reset("task_1_fire_easy", 42)
    episode_b, observation_b = env.reset("task_1_fire_easy", 42)

    assert episode_a != episode_b
    assert _normalized_observation(observation_a) == _normalized_observation(observation_b)


def test_step_wait_action() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    observation, reward, done, info = env.step(
        WaitAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.wait,
        )
    )

    assert observation.step == 1
    assert reward.total == pytest.approx(-0.2)
    assert done is False
    assert info.invalid_action is False


def test_step_route_civilians() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_2_flood_medium", 42)

    env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F1_R0",
            to_node_id="F1_R1",
            occupancy=Occupancy(injured=1),
            preference="fastest",
        )
    )

    ep = env.get_internal_state(episode_id)
    source_room = _room_by_id(ep, "F1_R0")

    assert source_room.occupancy.injured == 0
    assert len(ep.civilians_in_transit) == 1
    assert ep.civilians_in_transit[0].occupancy.injured == 1
    assert ep.civilians_in_transit[0].steps_remaining == 1


def test_civilians_arrive_at_exit() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    observation, reward, done, info = env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R0",
            to_node_id="EX0",
            occupancy=Occupancy(mobile=2),
            preference="fastest",
        )
    )

    ep = env.get_internal_state(episode_id)

    assert ep.civilians_saved.mobile == 2
    assert ep.civilians_in_transit == []
    assert observation.summary.incident_outcomes.safe == 2
    assert observation.summary.incident_outcomes.deaths == 0
    assert reward.total > 0
    assert done is False
    assert info.invalid_action is False


def test_invalid_action_rejected() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    observation, reward, done, info = env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F99_R99",
            to_node_id="F0_R0",
            occupancy=Occupancy(mobile=1),
            preference="fastest",
        )
    )

    assert info.invalid_action is True
    assert info.invalid_reason is not None
    assert observation.step == 1

    wrong_step = env.step(
        WaitAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.wait,
        )
    )
    assert wrong_step[3].invalid_action is True

    with pytest.raises(ValueError, match="Unknown episode_id"):
        env.step(
            WaitAction(
                episode_id="missing",
                expected_step=0,
                action_type=ActionType.wait,
            )
        )


def test_episode_terminates_all_saved() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)
    ep = env.get_internal_state(episode_id)

    for floor in ep.building.floors:
        for room in floor.rooms:
            room.occupancy = Occupancy()
    _room_by_id(ep, "F0_R0").occupancy = Occupancy(mobile=2)
    ep.total_civilians = Occupancy(mobile=2)

    observation, reward, done, info = env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R0",
            to_node_id="EX0",
            occupancy=Occupancy(mobile=2),
            preference="fastest",
        )
    )

    assert done is True
    assert observation.step == 1
    assert info.termination_reason == TerminationReason.all_saved
    assert reward.completion_bonus > 0


def test_episode_terminates_max_steps() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)
    ep = env.get_internal_state(episode_id)
    ep.task.max_steps = 1

    _, _, done, info = env.step(
        WaitAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.wait,
        )
    )

    assert done is True
    assert info.termination_reason == TerminationReason.max_steps


def test_reward_total_matches_components() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    _, reward, _, _ = env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R0",
            to_node_id="EX0",
            occupancy=Occupancy(mobile=2),
            preference="fastest",
        )
    )

    assert reward.total == pytest.approx(_reward_component_sum(reward))


def test_deterministic_replay() -> None:
    env_a = EvacEnvironment()
    env_b = EvacEnvironment()
    episode_a, _ = env_a.reset("task_1_fire_easy", 42)
    episode_b, _ = env_b.reset("task_1_fire_easy", 42)

    actions = [
        RouteCiviliansAction(
            episode_id="",
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R0",
            to_node_id="EX0",
            occupancy=Occupancy(mobile=2),
            preference="fastest",
        ),
        EvacuateFloorAction(
            episode_id="",
            expected_step=1,
            action_type=ActionType.evacuate_floor,
            floor_id=2,
        ),
        WaitAction(
            episode_id="",
            expected_step=2,
            action_type=ActionType.wait,
        ),
    ]

    trace_a = []
    trace_b = []
    for template in actions:
        action_a = template.model_copy(update={"episode_id": episode_a})
        action_b = template.model_copy(update={"episode_id": episode_b})
        step_a = env_a.step(action_a)
        step_b = env_b.step(action_b)
        trace_a.append(
            (
                _normalized_observation(step_a[0]),
                step_a[1].model_dump(mode="json"),
                step_a[2],
                step_a[3].model_dump(mode="json"),
            )
        )
        trace_b.append(
            (
                _normalized_observation(step_b[0]),
                step_b[1].model_dump(mode="json"),
                step_b[2],
                step_b[3].model_dump(mode="json"),
            )
        )

    assert trace_a == trace_b


def test_state_returns_public_view() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    state = env.state(episode_id)

    assert isinstance(state, StateView)
    assert "action_history" not in state.model_dump(mode="json")
    assert state.episode_id == episode_id


def test_lockdown_room_blocks_access() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)

    env.step(
        LockdownRoomAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.lockdown_room,
            room_id="F1_R1",
        )
    )

    ep = env.get_internal_state(episode_id)
    assert _room_by_id(ep, "F1_R1").accessible is False


def test_evacuate_floor_routes_all() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", 42)
    ep = env.get_internal_state(episode_id)
    before = {
        room.room_id: room.occupancy.total
        for room in ep.building.floors[2].rooms
        if room.occupancy.total > 0
    }

    env.step(
        EvacuateFloorAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.evacuate_floor,
            floor_id=2,
        )
    )

    ep = env.get_internal_state(episode_id)
    for room_id in before:
        assert _room_by_id(ep, room_id).occupancy.total == 0
