from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.schemas.multi_agent import ActionTypeMA, StructuredBelief


def _predict(
    env: EvacEnvironment,
    episode_id: str,
    agent_id: str,
    belief_id: str,
    room_id: str,
    horizon: int,
) -> tuple[object, object, bool, dict]:
    belief = StructuredBelief(
        belief_id=belief_id,
        predictor_agent_id=agent_id,
        target_entity_ids=[room_id],
        horizon=horizon,
        prediction_payload={"expected_civilians_in_room": 1},
        confidence=0.8,
    )
    return env.step_floor_agent_action(
        episode_id,
        agent_id,
        ActionTypeMA.predict_state,
        {"belief": belief.model_dump(mode="json")},
    )


def test_duplicate_unresolved_belief_is_rejected_and_penalized() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)

    _predict(env, episode_id, "floor_2_agent", "b1", "F2_R2", horizon=3)
    _, reward, _, info = _predict(env, episode_id, "floor_2_agent", "b2", "F2_R2", horizon=3)

    assert reward.breakdown.duplicate_belief_penalty < 0
    assert info["score_snapshot"]["registrations"]["floor_2_agent"]["accepted"] is False
    assert info["score_snapshot"]["registrations"]["floor_2_agent"]["reason"] == "duplicate_target"


def test_same_target_is_accepted_again_after_resolution() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)

    _predict(env, episode_id, "floor_2_agent", "b1", "F2_R2", horizon=2)
    env.step_floor_agent_action(episode_id, "floor_2_agent", ActionTypeMA.wait, {})
    _, reward, _, info = _predict(env, episode_id, "floor_2_agent", "b3", "F2_R2", horizon=2)

    assert "duplicate_belief_penalty" not in reward.breakdown.get_components()
    assert info["score_snapshot"]["registrations"]["floor_2_agent"]["accepted"] is True


def test_slot_limit_rejects_second_pending_target_for_same_predictor() -> None:
    env = EvacEnvironment()
    episode_id, _ = env.reset_multi_agent("task_lh_fire_easy", 42)

    _predict(env, episode_id, "floor_2_agent", "b1", "F2_R2", horizon=5)
    _, reward, _, info = _predict(env, episode_id, "floor_2_agent", "b2", "F2_R3", horizon=5)

    assert reward.breakdown.duplicate_belief_penalty < 0
    assert info["score_snapshot"]["registrations"]["floor_2_agent"]["accepted"] is False
    assert info["score_snapshot"]["registrations"]["floor_2_agent"]["reason"] == "slot_limit"
