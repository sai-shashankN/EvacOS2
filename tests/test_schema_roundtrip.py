"""Round-trip fixture tests for golden JSON fixtures.

Each fixture is loaded, parsed into the corresponding Pydantic model,
re-serialized, and compared for structural equality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evacos_ma.schemas.multi_agent import (
    ACTION_TYPE_TO_ARGS,
    REWARD_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    ActionBundleMA,
    ActionTypeMA,
    FloorAgentObservationMA,
    OrchestratorObservationMA,
    PredictStateArgs,
    StepResultMA,
    StructuredBelief,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.loads(f.read())


class TestFloorObservationRoundTrip:
    def test_parse_and_reserialize(self):
        raw = _load_fixture("floor_observation.golden.json")
        model = FloorAgentObservationMA.model_validate(raw)
        reserialized = json.loads(model.model_dump_json())
        # Structural equality: re-parse original to normalize types
        assert json.loads(json.dumps(raw)) == reserialized

    def test_version_fields(self):
        raw = _load_fixture("floor_observation.golden.json")
        model = FloorAgentObservationMA.model_validate(raw)
        assert model.trace_schema_version == TRACE_SCHEMA_VERSION
        assert model.generator_config_hash == "sha256:abc123"

    def test_golden_values(self):
        raw = _load_fixture("floor_observation.golden.json")
        model = FloorAgentObservationMA.model_validate(raw)
        assert model.episode_id == "ep_goldfixture_0001"
        assert model.agent_id == "floor_2_agent"
        assert model.floor_id == "floor_2"
        assert model.round_id == 17
        assert model.step == 17
        assert model.max_steps == 350
        assert model.seed == 42
        assert model.tier.value == "easy"
        assert model.disaster_family == "fire"
        assert len(model.visible_rooms) >= 2
        assert len(model.local_hazards) >= 1
        assert len(model.visible_civilian_groups) >= 1
        assert model.active_directive is not None
        assert "room_201" in model.visibility_age_by_room

    def test_has_rooms_and_hazards(self):
        raw = _load_fixture("floor_observation.golden.json")
        model = FloorAgentObservationMA.model_validate(raw)
        room_ids = [r.room_id for r in model.visible_rooms]
        assert "room_201" in room_ids
        assert any(h.hazard_type == "fire" for h in model.local_hazards)


class TestOrchestratorObservationRoundTrip:
    def test_parse_and_reserialize(self):
        raw = _load_fixture("orchestrator_observation.golden.json")
        model = OrchestratorObservationMA.model_validate(raw)
        reserialized = json.loads(model.model_dump_json())
        assert json.loads(json.dumps(raw)) == reserialized

    def test_version_fields(self):
        raw = _load_fixture("orchestrator_observation.golden.json")
        model = OrchestratorObservationMA.model_validate(raw)
        assert model.trace_schema_version == TRACE_SCHEMA_VERSION
        assert model.generator_config_hash == "sha256:abc123"

    def test_golden_values(self):
        raw = _load_fixture("orchestrator_observation.golden.json")
        model = OrchestratorObservationMA.model_validate(raw)
        assert model.episode_id == "ep_goldfixture_0001"
        assert model.agent_id == "orchestrator"
        assert len(model.floor_summaries) == 5
        assert len(model.inter_floor.stairwells) >= 1
        assert model.inter_floor.elevator is not None
        assert len(model.inter_floor.global_exit_queue) >= 1
        assert model.belief_rollup.total_beliefs > 0
        assert len(model.recent_floor_actions) == 2
        assert len(model.recent_directive_outcomes) >= 1


class TestActionBundleRoundTrip:
    def test_parse_and_reserialize(self):
        raw = _load_fixture("action_bundle.golden.json")
        model = ActionBundleMA.model_validate(raw)
        reserialized = json.loads(model.model_dump_json())
        assert json.loads(json.dumps(raw)) == reserialized

    def test_orchestrator_action_present(self):
        raw = _load_fixture("action_bundle.golden.json")
        model = ActionBundleMA.model_validate(raw)
        assert model.orchestrator_action is not None
        assert model.orchestrator_action.action_type == ActionTypeMA.broadcast_directive

    def test_floor_actions_count(self):
        raw = _load_fixture("action_bundle.golden.json")
        model = ActionBundleMA.model_validate(raw)
        assert len(model.floor_actions) == 5

    def test_action_type_dispatch(self):
        """For each action, validate arguments parse as the typed arg model."""
        raw = _load_fixture("action_bundle.golden.json")
        model = ActionBundleMA.model_validate(raw)

        all_actions = list(model.floor_actions.values())
        if model.orchestrator_action:
            all_actions.append(model.orchestrator_action)

        for action in all_actions:
            action_type = ActionTypeMA(action.action_type)
            args_model_cls = ACTION_TYPE_TO_ARGS[action_type]
            # This must not raise
            parsed_args = args_model_cls.model_validate(action.arguments)

    def test_predict_state_has_structured_belief(self):
        raw = _load_fixture("action_bundle.golden.json")
        model = ActionBundleMA.model_validate(raw)
        predict_action = model.floor_actions["floor_2_agent"]
        assert predict_action.action_type == ActionTypeMA.predict_state
        parsed = PredictStateArgs.model_validate(predict_action.arguments)
        assert isinstance(parsed.belief, StructuredBelief)
        assert parsed.belief.belief_id == "b_gold_001"
        assert parsed.belief.confidence == 0.75
        assert parsed.belief.justification != ""


class TestStepResultRoundTrip:
    def test_parse_and_reserialize(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        reserialized = json.loads(model.model_dump_json())
        assert json.loads(json.dumps(raw)) == reserialized

    def test_observations_by_role(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        assert model.observations_by_role.orchestrator is not None
        assert len(model.observations_by_role.floors) == 5

    def test_rewards_by_role(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        assert model.rewards_by_role.orchestrator is not None
        assert len(model.rewards_by_role.floors) == 5

    def test_invalid_actions_present(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        assert len(model.invalid_actions) >= 1

    def test_round_events_present(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        assert len(model.round_events) >= 2

    def test_info_traces(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        assert len(model.info.reservation_trace) >= 1
        assert len(model.info.arbitration_trace) >= 1

    def test_reward_schema_version_in_rewards(self):
        raw = _load_fixture("step_result.golden.json")
        model = StepResultMA.model_validate(raw)
        rewards = model.rewards_by_role
        assert rewards.orchestrator.reward_schema_version == REWARD_SCHEMA_VERSION
        for agent_id, rr in rewards.floors.items():
            assert rr.reward_schema_version == REWARD_SCHEMA_VERSION
