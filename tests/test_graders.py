from __future__ import annotations

import random

from evacos_ma.env import EvacEnvironment
from evacos_ma.grader import SCORE_EPSILON, grade_episode
from evacos_ma.models import (
    ActionType,
    IncidentOutcomes,
    Occupancy,
    RouteCiviliansAction,
    WaitAction,
)
from evacos_ma.task_registry import get_all_tasks


def _iter_rooms(ep):
    for floor in ep.building.floors:
        for room in floor.rooms:
            yield room


def _room_lookup(ep) -> dict[str, object]:
    return {room.room_id: room for room in _iter_rooms(ep)}


def _exit_lookup(ep) -> dict[str, object]:
    exits = {}
    for floor in ep.building.floors:
        for exit_obj in floor.exits:
            exits.setdefault(exit_obj.exit_id, exit_obj)
    return exits


def _clear_episode_population(ep) -> None:
    for room in _iter_rooms(ep):
        room.occupancy = Occupancy()
    ep.civilians_saved = Occupancy()
    ep.civilians_lost = Occupancy()
    ep.civilians_in_transit = []
    ep.total_civilians = Occupancy()
    ep.resolved_incident_outcomes = IncidentOutcomes()
    ep.room_incident_outcomes = {room.room_id: IncidentOutcomes() for room in _iter_rooms(ep)}


def _wait_until_done(env: EvacEnvironment, episode_id: str) -> None:
    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            return
        env.step(
            WaitAction(
                episode_id=episode_id,
                expected_step=ep.step,
                action_type=ActionType.wait,
            )
        )


def _run_perfect_task1(seed: int = 42):
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", seed)

    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            return env, ep

        occupied_rooms = sorted(
            [room for room in _iter_rooms(ep) if room.occupancy.total > 0],
            key=lambda room: (-room.floor_id, -room.hazard.severity, room.room_id),
        )

        for room in occupied_rooms:
            path = env._find_path_to_exit(ep, room.room_id)
            if path is None or len(path) < 2:
                continue
            env.step(
                RouteCiviliansAction(
                    episode_id=episode_id,
                    expected_step=ep.step,
                    action_type=ActionType.route_civilians,
                    from_node_id=room.room_id,
                    to_node_id=path[1],
                    occupancy=room.occupancy.model_copy(deep=True),
                    preference="fastest",
                )
            )
            break
        else:
            env.step(
                WaitAction(
                    episode_id=episode_id,
                    expected_step=ep.step,
                    action_type=ActionType.wait,
                )
            )


def _run_zero_task1(seed: int = 42):
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", seed)
    _wait_until_done(env, episode_id)
    return env, env.get_internal_state(episode_id)


def _run_partial_task1(seed: int = 42):
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_1_fire_easy", seed)

    env.step(
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
    env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=1,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R1",
            to_node_id="F0_R0",
            occupancy=Occupancy(mobile=3),
            preference="fastest",
        )
    )
    env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=2,
            action_type=ActionType.route_civilians,
            from_node_id="F0_R0",
            to_node_id="EX0",
            occupancy=Occupancy(mobile=3),
            preference="fastest",
        )
    )
    _wait_until_done(env, episode_id)
    return env, env.get_internal_state(episode_id)


def _build_task2_episode(save_injured: bool):
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_2_flood_medium", 42)
    ep = env.get_internal_state(episode_id)
    ep.task.max_steps = 2
    _clear_episode_population(ep)

    rooftop_exit = next(
        exit_obj
        for exit_obj in _exit_lookup(ep).values()
        if exit_obj.exit_type.value == "rooftop"
    )
    rooftop_room = _room_lookup(ep)[rooftop_exit.adjacent_room_id]
    rooftop_room.occupancy = Occupancy(mobile=1, injured=1)
    ep.total_civilians = Occupancy(mobile=1, injured=1)

    routed = Occupancy(injured=1) if save_injured else Occupancy(mobile=1)
    env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id=rooftop_room.room_id,
            to_node_id=rooftop_exit.exit_id,
            occupancy=routed,
            preference="fastest",
        )
    )
    env.step(
        WaitAction(
            episode_id=episode_id,
            expected_step=1,
            action_type=ActionType.wait,
        )
    )
    return env.get_internal_state(episode_id)


def _build_task3_episode(save_impaired: bool):
    env = EvacEnvironment()
    episode_id, _ = env.reset("task_3_earthquake_hard", 42)
    ep = env.get_internal_state(episode_id)
    ep.task.max_steps = 1
    _clear_episode_population(ep)

    ground_exit = next(
        exit_obj
        for exit_obj in _exit_lookup(ep).values()
        if exit_obj.exit_type.value == "ground"
    )
    exit_room = _room_lookup(ep)[ground_exit.adjacent_room_id]
    exit_room.occupancy = Occupancy(mobile=1)
    ep.total_civilians = Occupancy(mobile=1, mobility_impaired=1)
    if save_impaired:
        ep.civilians_saved = Occupancy(mobility_impaired=1)

    env.step(
        RouteCiviliansAction(
            episode_id=episode_id,
            expected_step=0,
            action_type=ActionType.route_civilians,
            from_node_id=exit_room.room_id,
            to_node_id=ground_exit.exit_id,
            occupancy=Occupancy(mobile=1),
            preference="fastest",
        )
    )
    return env.get_internal_state(episode_id)


def _random_valid_action(env: EvacEnvironment, ep, rng: random.Random):
    candidates = [
        WaitAction(
            episode_id=ep.episode_id,
            expected_step=ep.step,
            action_type=ActionType.wait,
        )
    ]
    room_lookup = _room_lookup(ep)
    exit_lookup = _exit_lookup(ep)

    for room in room_lookup.values():
        if room.occupancy.total <= 0 or not room.accessible or not room.hazard.passable:
            continue

        for target in sorted(room.adjacent_node_ids):
            if target in room_lookup:
                target_room = room_lookup[target]
                if not target_room.accessible or not target_room.hazard.passable:
                    continue
            elif target in exit_lookup:
                if exit_lookup[target].blocked:
                    continue
            else:
                continue

            connection = env._resolve_connection(ep, room.room_id, target)
            if connection is None:
                continue

            path_kind, _ = connection
            occupancy = (
                room.occupancy.model_copy(deep=True)
                if path_kind == "elevator"
                else Occupancy(
                    mobile=room.occupancy.mobile,
                    injured=room.occupancy.injured,
                )
            )
            if occupancy.total <= 0:
                continue

            candidates.append(
                RouteCiviliansAction(
                    episode_id=ep.episode_id,
                    expected_step=ep.step,
                    action_type=ActionType.route_civilians,
                    from_node_id=room.room_id,
                    to_node_id=target,
                    occupancy=occupancy,
                    preference="fastest",
                )
            )

    return rng.choice(candidates)


def test_grade_perfect_task1() -> None:
    _, episode = _run_perfect_task1()

    graded = grade_episode(episode)
    breakdown = graded["breakdown"]

    assert episode.civilians_saved.total == 15
    assert graded["score"] >= 0.9
    assert graded["score"] == 1.0 - SCORE_EPSILON
    assert graded["score"] < 1.0
    assert breakdown["incident_deaths"] == 0.0
    assert (
        breakdown["incident_safe"]
        + breakdown["incident_mild_injury"]
        + breakdown["incident_severe_injury"]
        + breakdown["incident_deaths"]
    ) == 15.0


def test_grade_zero_task1() -> None:
    _, episode = _run_zero_task1()

    graded = grade_episode(episode)

    assert graded["score"] <= 0.2
    assert graded["score"] == SCORE_EPSILON
    assert graded["score"] > 0.0
    assert (
        graded["breakdown"]["incident_deaths"] + graded["breakdown"]["incident_severe_injury"]
    ) == float(episode.total_civilians.total)


def test_grade_partial_task1() -> None:
    _, episode = _run_partial_task1()

    graded = grade_episode(episode)

    assert 0.3 <= graded["score"] <= 0.8
    assert graded["breakdown"]["incident_safe"] <= float(episode.total_civilians.total)


def test_grade_task2_with_injured() -> None:
    mobile_only_episode = _build_task2_episode(save_injured=False)
    injured_episode = _build_task2_episode(save_injured=True)

    mobile_only_grade = grade_episode(mobile_only_episode)
    injured_grade = grade_episode(injured_episode)

    assert injured_episode.civilians_saved.injured == 1
    assert injured_grade["score"] > mobile_only_grade["score"]


def test_grade_task3_mobility_weight() -> None:
    no_impaired_saved_episode = _build_task3_episode(save_impaired=False)
    impaired_saved_episode = _build_task3_episode(save_impaired=True)

    no_impaired_grade = grade_episode(no_impaired_saved_episode)
    impaired_grade = grade_episode(impaired_saved_episode)

    assert impaired_saved_episode.civilians_saved.mobility_impaired == 1
    assert impaired_grade["score"] > no_impaired_grade["score"]


def test_grade_all_tasks_bounded() -> None:
    rng = random.Random(7)

    for task in get_all_tasks():
        env = EvacEnvironment()
        episode_id, _ = env.reset(task.task_id, 42)

        while True:
            episode = env.get_internal_state(episode_id)
            if episode.done:
                graded = grade_episode(episode)
                assert 0.0 < graded["score"] < 1.0
                break

            action = _random_valid_action(env, episode, rng)
            _, _, _, info = env.step(action)
            assert info.invalid_action is False


def test_grade_deterministic() -> None:
    _, episode = _run_perfect_task1()

    first = grade_episode(episode)
    second = grade_episode(episode)

    assert first == second


def test_grade_distinguishes_quality() -> None:
    _, good_episode = _run_perfect_task1()
    _, bad_episode = _run_zero_task1()

    good_grade = grade_episode(good_episode)
    bad_grade = grade_episode(bad_episode)

    assert good_grade["score"] > bad_grade["score"]


def test_grade_reports_incident_outcomes_sum_to_total() -> None:
    _, episode = _run_perfect_task1()

    graded = grade_episode(episode)
    breakdown = graded["breakdown"]

    assert (
        breakdown["incident_safe"]
        + breakdown["incident_mild_injury"]
        + breakdown["incident_severe_injury"]
        + breakdown["incident_deaths"]
    ) == float(episode.total_civilians.total)


def test_grade_breakdown_avoids_intermediate_score_keys() -> None:
    _, episode = _run_perfect_task1()

    graded = grade_episode(episode)

    assert all(key == "score" or "score" not in key for key in graded["breakdown"])
