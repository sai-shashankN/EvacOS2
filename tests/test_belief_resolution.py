from __future__ import annotations

from evacos_ma.beliefs import BeliefStore
from evacos_ma.env import EvacEnvironment
from evacos_ma.schemas.multi_agent import ActionTypeMA, BeliefAuditRow, StructuredBelief


def test_perfect_belief_resolves_with_high_score_and_reward() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)
    ep = env.get_internal_state(episode_id)
    room = ep.building.floors[2].rooms[2]
    belief = StructuredBelief(
        belief_id="belief-perfect",
        predictor_agent_id="floor_2_agent",
        target_entity_ids=[room.room_id],
        horizon=2,
        prediction_payload={"expected_civilians_in_room": room.occupancy.total},
        confidence=0.9,
    )

    env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.predict_state, {"belief": belief.model_dump(mode="json")})
    _, reward, _, _ = env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})

    assert reward.breakdown.floor_prediction >= 0.95 * ep.task.reward_weights.floor_prediction


def test_wildly_wrong_belief_scores_near_zero() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)
    ep = env.get_internal_state(episode_id)
    room = ep.building.floors[2].rooms[2]
    belief = StructuredBelief(
        belief_id="belief-bad",
        predictor_agent_id="floor_2_agent",
        target_entity_ids=[room.room_id],
        horizon=2,
        prediction_payload={"expected_civilians_in_room": room.occupancy.total + 10},
        confidence=0.9,
    )

    env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.predict_state, {"belief": belief.model_dump(mode="json")})
    _, reward, _, _ = env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})

    assert reward.breakdown.floor_prediction <= 0.1 * ep.task.reward_weights.floor_prediction


def test_belief_audit_row_emitted_on_resolution() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)
    ep = env.get_internal_state(episode_id)
    room = ep.building.floors[2].rooms[2]
    belief = StructuredBelief(
        belief_id="belief-audit",
        predictor_agent_id="floor_2_agent",
        target_entity_ids=[room.room_id],
        horizon=2,
        prediction_payload={"expected_civilians_in_room": room.occupancy.total},
        confidence=0.7,
    )

    env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.predict_state, {"belief": belief.model_dump(mode="json")})
    env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})

    audit = BeliefAuditRow.model_validate(ep.belief_audit_log[-1])
    assert audit.belief_id == "belief-audit"
    assert audit.predictor_agent_id == "floor_2_agent"
    assert audit.resolved is True
    assert isinstance(audit.score, float)


def test_resolved_round_is_set_once_and_not_rescored() -> None:
    store = BeliefStore(
        episode_id="ep",
        seed=42,
        tier="easy",
        disaster_family="fire",
    )
    belief = StructuredBelief(
        belief_id="belief-once",
        predictor_agent_id="floor_2_agent",
        target_entity_ids=["F2_R2"],
        horizon=2,
        prediction_payload={"expected_civilians_in_room": 3},
        confidence=0.5,
        created_round=0,
    )
    registration = store.register(belief, predictor="floor_2_agent")
    assert registration.accepted is True

    first = store.tick(2, lambda _belief, _round: {"expected_civilians_in_room": 3})
    second = store.tick(2, lambda _belief, _round: {"expected_civilians_in_room": 3})

    stored = next(item for item in store.beliefs() if item.belief_id == "belief-once")
    assert len(first) == 1
    assert second == []
    assert stored.resolved_round_or_null == 2
