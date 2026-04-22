from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.round_protocol import RoundProtocol
from evacos_ma.schemas.multi_agent import ActionEnvelopeMA, ActionTypeMA


def test_malformed_override_replacement_type_is_rejected_not_coerced():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    room_id = ep.building.floors[0].rooms[0].room_id

    floor_action = ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=ep.step,
        agent_id="floor_0_agent",
        action_id="orig_prioritize",
        action_type=ActionTypeMA.prioritize_room,
        arguments={"room_id": room_id},
    )
    override_action = ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=ep.step,
        agent_id="orchestrator",
        action_id="bad_override",
        action_type=ActionTypeMA.override_floor_agent,
        arguments={
            "target_floor_agent_id": "floor_0_agent",
            "replacement_action_type": "fly_to_moon",
            "replacement_arguments": {},
        },
    )

    result = RoundProtocol().run_round(
        env=env,
        ep=ep,
        orchestrator_action=override_action,
        floor_actions={"floor_0_agent": floor_action},
        round_id=ep.step,
        directive_store=env._directive_stores[episode_id],
        handoff_store=env._handoff_stores[episode_id],
    )

    assert any(action.action_id == "orig_prioritize" for action in result.accepted_actions)
    assert not any(
        action.action_id == "override_bad_override"
        for action in result.accepted_actions
    )
    assert any(
        rejection["reason"].startswith("invalid_override_replacement_type")
        for rejection in result.rejected_actions
    )
    assert result.counterfactual_deltas == {}


def test_override_with_missing_target_floor_agent_is_rejected():
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_1_fire_easy", seed=42)
    ep = env.get_internal_state(episode_id)
    room_id = ep.building.floors[0].rooms[0].room_id

    floor_action = ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=ep.step,
        agent_id="floor_0_agent",
        action_id="orig_prioritize",
        action_type=ActionTypeMA.prioritize_room,
        arguments={"room_id": room_id},
    )
    override_action = ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=ep.step,
        agent_id="orchestrator",
        action_id="missing_target",
        action_type=ActionTypeMA.override_floor_agent,
        arguments={
            "target_floor_agent_id": "floor_99_agent",
            "replacement_action_type": "wait",
            "replacement_arguments": {},
        },
    )

    result = RoundProtocol().run_round(
        env=env,
        ep=ep,
        orchestrator_action=override_action,
        floor_actions={"floor_0_agent": floor_action},
        round_id=ep.step,
        directive_store=env._directive_stores[episode_id],
        handoff_store=env._handoff_stores[episode_id],
    )

    assert any(action.action_id == "orig_prioritize" for action in result.accepted_actions)
    assert any(
        rejection["reason"] == "invalid_override_target: floor_99_agent"
        for rejection in result.rejected_actions
    )
