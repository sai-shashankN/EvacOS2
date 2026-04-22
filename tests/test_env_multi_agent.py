from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import WaitAction
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
)


def test_step_multi_agent_applies_actions_before_physics_tick(monkeypatch):
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    saved_before = ep.civilians_saved.total
    marker = {"apply_seen": False, "saved_at_step": None, "accepted_count": 0}

    original_step = env.step

    def step_spy(action):
        if isinstance(action, WaitAction):
            marker["saved_at_step"] = ep.civilians_saved.total
            assert marker["apply_seen"] is True
            assert ep.civilians_saved.total == saved_before + marker["accepted_count"]
        return original_step(action)

    def apply_spy(_env, _ep, accepted):
        marker["apply_seen"] = True
        marker["accepted_count"] = len(accepted)
        _ep.civilians_saved.mobile += len(accepted)

    monkeypatch.setattr(env, "step", step_spy)
    monkeypatch.setattr(env._round_protocol, "_apply", apply_spy)

    result = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={
                "floor_0_agent": ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id="floor_0_agent",
                    action_id="lockdown_before_tick",
                    action_type=ActionTypeMA.lockdown_room,
                    arguments={"room_id": "F0_R0"},
                )
            },
        )
    )

    assert result.observations_by_role.orchestrator.round_id == 1
    assert marker["saved_at_step"] == saved_before + marker["accepted_count"]


def test_team_progress_dense_emits_when_civilians_delta_nonzero():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    exit_id = next(exit_obj.exit_id for floor in ep.building.floors for exit_obj in floor.exits)

    result = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={
                "floor_0_agent": ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id="floor_0_agent",
                    action_id="team_progress_dense",
                    action_type=ActionTypeMA.open_exit,
                    arguments={"exit_id": exit_id},
                )
            },
        )
    )

    assert result.rewards_by_role.floors["floor_0_agent"].breakdown.team_progress_dense != 0.0
    assert result.rewards_by_role.orchestrator.breakdown.team_progress_dense != 0.0


def test_floor_invalid_action_penalty_emits_on_rejected_action():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)

    result = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={
                "floor_0_agent": ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id="floor_0_agent",
                    action_id="invalid_scout",
                    action_type=ActionTypeMA.scout,
                    arguments={},
                )
            },
        )
    )

    assert result.rewards_by_role.floors["floor_0_agent"].breakdown.floor_invalid_action < 0.0


def test_terminal_components_only_on_done_round():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    ep.task.max_steps = 2
    exit_id = next(exit_obj.exit_id for floor in ep.building.floors for exit_obj in floor.exits)

    result_one = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={
                "floor_0_agent": ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id="floor_0_agent",
                    action_id="terminal_round_one",
                    action_type=ActionTypeMA.open_exit,
                    arguments={"exit_id": exit_id},
                )
            },
        )
    )

    assert result_one.done is False
    assert result_one.rewards_by_role.orchestrator.breakdown.total_saved_terminal == 0.0
    assert result_one.rewards_by_role.orchestrator.breakdown.total_lost_terminal == 0.0

    result_two = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={},
        )
    )

    assert result_two.done is True
    assert result_two.rewards_by_role.orchestrator.breakdown.total_saved_terminal > 0.0


def test_recent_floor_actions_reflects_post_override_accepted_actions():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    room_id = ep.building.floors[0].rooms[0].room_id

    env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=ActionEnvelopeMA(
                episode_id=episode_id,
                round_id=ep.step,
                agent_id="orchestrator",
                action_id="override_lockdown",
                action_type=ActionTypeMA.override_floor_agent,
                arguments={
                    "target_floor_agent_id": "floor_0_agent",
                    "replacement_action_type": "lockdown_room",
                    "replacement_arguments": {"room_id": room_id},
                    "rationale": "hazard containment",
                },
            ),
            floor_actions={
                "floor_0_agent": ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=ep.step,
                    agent_id="floor_0_agent",
                    action_id="route_request",
                    action_type=ActionTypeMA.route_within_floor,
                    arguments={"from_room_id": room_id, "to_room_id": room_id},
                )
            },
        )
    )

    result = env.step_multi_agent(
        ActionBundleMA(
            episode_id=episode_id,
            round_id=ep.step,
            orchestrator_action=None,
            floor_actions={},
        )
    )

    recent_actions = [
        entry
        for entry in result.observations_by_role.orchestrator.recent_floor_actions
        if entry.agent_id == "floor_0_agent"
    ]
    assert any(entry.action_type == "lockdown_room" for entry in recent_actions)


def test_orchestrator_observation_carries_cascade_hint_on_multi_cascade():
    env = EvacEnvironment()
    _, obs = env.reset_multi_agent("task_lh_cascade_hard", seed=42)

    hint = obs.orchestrator.cascade_hint

    assert hint is not None
    assert hint.get("next_cascade_round") is not None
    assert hint.get("type")


def test_floor_summary_queue_pressure_nonzero_when_civilians_exceed_outflow():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    floor = ep.building.floors[0]

    for room in floor.rooms:
        room.occupancy.mobile = 0
        room.occupancy.injured = 0
        room.occupancy.mobility_impaired = 0
    for exit_obj in floor.exits:
        exit_obj.blocked = False
    for stairwell in floor.stairwells:
        stairwell.blocked = False

    outflow = len(floor.exits) * 10 + sum(
        stairwell.capacity_per_step for stairwell in floor.stairwells
    )
    assert outflow > 0
    floor.rooms[0].occupancy.mobile = max(1, outflow // 2)
    expected = min(1.0, floor.rooms[0].occupancy.total / outflow)

    obs = env._build_orchestrator_observation(ep)
    summary = next(item for item in obs.floor_summaries if item.floor_id == "floor_0")

    assert summary.queue_pressure > 0.0
    assert summary.queue_pressure <= 1.0
    assert summary.queue_pressure == expected


def test_floor_summary_queue_pressure_saturates_at_1_when_no_outflow():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    floor = ep.building.floors[0]

    for exit_obj in floor.exits:
        exit_obj.blocked = True
    for stairwell in floor.stairwells:
        stairwell.blocked = True
    for room in floor.rooms:
        room.occupancy.mobile = 0
        room.occupancy.injured = 0
        room.occupancy.mobility_impaired = 0
    floor.rooms[0].occupancy.mobile = 1

    obs = env._build_orchestrator_observation(ep)
    summary = next(item for item in obs.floor_summaries if item.floor_id == "floor_0")

    assert summary.queue_pressure == 1.0
