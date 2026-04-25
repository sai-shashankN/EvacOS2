from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import Elevator
from evacos_ma.round_protocol import RoundProtocol
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
)


def _first_exit_id(ep) -> str:
    for floor in ep.building.floors:
        if floor.exits:
            return floor.exits[0].exit_id
    raise AssertionError("episode has no exits")


def _first_stairwell_id(ep) -> str:
    for floor in ep.building.floors:
        if floor.stairwells:
            return floor.stairwells[0].stairwell_id
    raise AssertionError("episode has no stairwells")


def _first_elevator_id(ep) -> str:
    for floor in ep.building.floors:
        if floor.elevators:
            return floor.elevators[0].elevator_id
    raise AssertionError("episode has no elevators")


def test_apply_open_exit_unblocks_exit_on_real_ep():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    exit_id = _first_exit_id(ep)
    exit_obj = env._exit_lookup(ep.building)[exit_id]
    exit_obj.blocked = True

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="open_exit_apply",
                action_type=ActionTypeMA.open_exit,
                arguments={"exit_id": exit_id},
            )
        ],
    )

    assert exit_obj.blocked is False
    assert ep.civilians_saved.mobile == 1


def test_apply_route_within_floor_increments_civilians_saved():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    exit_id = _first_exit_id(ep)
    exit_obj = env._exit_lookup(ep.building)[exit_id]
    exit_obj.blocked = False

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="route_apply",
                action_type=ActionTypeMA.route_within_floor,
                arguments={"exit_id": exit_id},
            )
        ],
    )

    assert ep.civilians_saved.mobile == 1


def test_apply_route_within_floor_treats_legacy_to_room_exit_as_exit():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    exit_id = _first_exit_id(ep)
    env._exit_lookup(ep.building)[exit_id].blocked = False

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="route_apply_legacy_exit",
                action_type=ActionTypeMA.route_within_floor,
                arguments={"from_room_id": "room_01", "to_room_id": exit_id},
            )
        ],
    )

    assert ep.civilians_saved.mobile == 1


def test_apply_returns_per_agent_saved_lost_delta():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    exit_id = _first_exit_id(ep)
    stairwell_id = _first_stairwell_id(ep)

    env._exit_lookup(ep.building)[exit_id].blocked = False
    env._stairwell_lookup(ep.building)[stairwell_id].blocked = True

    deltas = RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="save_apply",
                action_type=ActionTypeMA.route_within_floor,
                arguments={"exit_id": exit_id},
            ),
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_1_agent",
                action_id="lose_apply",
                action_type=ActionTypeMA.route_between_floors,
                arguments={"stairwell_id": stairwell_id},
            ),
        ],
    )

    assert deltas == {
        "floor_0_agent": {"saved": 1, "lost": 0},
        "floor_1_agent": {"saved": 0, "lost": 1},
    }


def test_apply_route_between_floors_increments_civilians_saved_when_stairwell_open():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    stairwell_id = _first_stairwell_id(ep)
    env._stairwell_lookup(ep.building)[stairwell_id].blocked = False

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="route_between_floors_open",
                action_type=ActionTypeMA.route_between_floors,
                arguments={"stairwell_id": stairwell_id},
            )
        ],
    )

    assert ep.civilians_saved.mobile == 1
    assert ep.civilians_lost.mobile == 0


def test_apply_route_between_floors_increments_civilians_lost_when_stairwell_blocked():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    stairwell_id = _first_stairwell_id(ep)
    env._stairwell_lookup(ep.building)[stairwell_id].blocked = True

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="route_between_floors_blocked",
                action_type=ActionTypeMA.route_between_floors,
                arguments={"stairwell_id": stairwell_id},
            )
        ],
    )

    assert ep.civilians_saved.mobile == 0
    assert ep.civilians_lost.mobile == 1


def test_apply_call_elevator_increments_saved_when_operable():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    ep.building.floors[0].elevators.append(
        Elevator(
            elevator_id="elevator_main",
            floor_ids=[floor.floor_id for floor in ep.building.floors],
            current_floor=ep.building.floors[0].floor_id,
            operational=True,
        )
    )
    elevator_id = _first_elevator_id(ep)

    RoundProtocol()._apply(
        env,
        ep,
        [
            ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="floor_0_agent",
                action_id="call_elevator_operable",
                action_type=ActionTypeMA.call_elevator,
                arguments={"elevator_id": elevator_id},
            )
        ],
    )

    assert ep.civilians_saved.mobile == 1
