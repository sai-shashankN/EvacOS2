"""Tests for policy adapter: StubPolicy, generation token capture, and parsing.

Heavy-dep-free.
"""

import json
import sys
import types

import pytest

from evacos_ma.schemas.multi_agent import ActionTypeMA
from training.policy_adapter import (
    RoleRoutedPolicy,
    ScopeRoutedFloorPolicy,
    StubPolicy,
    UnslothPolicy,
    parse_completion_to_action,
)


class TestStubPolicy:
    def test_stub_emits_valid_json_orchestrator_and_floor(self):
        """StubPolicy emits valid-JSON completions for orchestrator + floor_0_agent across 5 seeds."""
        for seed in range(5):
            policy = StubPolicy(seed=seed)
            # Orchestrator prompt (minimal)
            orch_prompt = [
                {"role": "system", "content": "Round: 0\nepisode_id: ep1"},
                {"role": "user", "content": "Floor summaries: []\nBeliefs: total=0"},
            ]
            orch_completion, orch_token_ids = policy.act(orch_prompt, "orchestrator", "orchestrator")
            orch_obj = json.loads(orch_completion)
            assert "action_type" in orch_obj
            assert orch_obj["agent_id"] == "orchestrator"
            assert orch_token_ids == []

            # Floor prompt (minimal)
            floor_prompt = [
                {"role": "system", "content": "Floor: floor_0\nRound: 0\nepisode_id: ep1"},
                {"role": "user", "content": 'Rooms: []\nExits: [{"exit_id":"E1","blocked":false}]'},
            ]
            floor_completion, floor_token_ids = policy.act(floor_prompt, "floor_0_agent", "floor_agent")
            floor_obj = json.loads(floor_completion)
            assert "action_type" in floor_obj
            assert floor_obj["agent_id"] == "floor_0_agent"
            assert floor_token_ids == []


class TestUnslothGenerationTokenCapture:
    def test_hf_generate_returns_text_and_token_ids_per_row_for_mixed_prompt_lengths(self, monkeypatch):
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        class FakeFastLanguageModel:
            @staticmethod
            def for_inference(model):
                model.eval()

            @staticmethod
            def for_training(model):
                model.train()

        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastLanguageModel = FakeFastLanguageModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

        class FakeTensor:
            def __init__(self, values):
                self._values = list(values)

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return FakeTensor(self._values[item])
                return self._values[item]

            def tolist(self):
                return list(self._values)

            def __iter__(self):
                return iter(self._values)

        class FakeMatrix:
            def __init__(self, rows):
                self._rows = [FakeTensor(row) for row in rows]
                self.shape = (len(rows), len(rows[0]))

            def __iter__(self):
                return iter(self._rows)

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"
            pad_token_id = 0
            padding_side = "left"

            def __call__(self, rendered, return_tensors="pt", padding=True, truncation=False):
                del rendered, return_tensors, padding, truncation
                return FakeBatch(
                    {
                        "input_ids": FakeMatrix(
                            [
                                [0, 0, 10, 11, 12],
                                [20, 21, 22, 23, 24],
                            ]
                        ),
                        "attention_mask": FakeMatrix(
                            [
                                [0, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                            ]
                        ),
                    }
                )

            @staticmethod
            def decode(generated, skip_special_tokens=True):
                del skip_special_tokens
                return ",".join(str(tok) for tok in generated.tolist() if tok != 0)

        class FakeModel:
            def __init__(self):
                self.training = True
                self.device = "cpu"

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def generate(self, **kwargs):
                return FakeMatrix(
                    [
                        [0, 0, 10, 11, 12, 90, 91, 0],
                        [20, 21, 22, 23, 24, 92, 93, 94],
                    ]
                )

        policy = UnslothPolicy.__new__(UnslothPolicy)
        policy._tokenizer = FakeTokenizer()
        policy._model = FakeModel()
        policy._max_new_tokens = 8
        policy._temperature = 0.0

        outputs = policy._hf_generate(["short", "much longer"])

        assert outputs == [
            ("90,91", [90, 91]),
            ("92,93,94", [92, 93, 94]),
        ]


class TestRoleRoutedPolicy:
    def test_act_batch_routes_by_role_and_preserves_order(self):
        class FakePolicy:
            def __init__(self, label):
                self.label = label
                self.calls = []

            def act_batch(self, prompts, agent_ids, roles):
                self.calls.append((prompts, agent_ids, roles))
                return [(f"{self.label}:{agent_id}", [idx]) for idx, agent_id in enumerate(agent_ids)]

        orch = FakePolicy("orch")
        floor = FakePolicy("floor")
        policy = RoleRoutedPolicy(orchestrator_policy=orch, floor_policy=floor)

        outputs = policy.act_batch(
            prompts=[
                [{"role": "user", "content": "o"}],
                [{"role": "user", "content": "f1"}],
                [{"role": "user", "content": "f2"}],
            ],
            agent_ids=["orchestrator", "floor_0_agent", "floor_1_agent"],
            roles=["orchestrator", "floor_agent", "floor_agent"],
        )

        assert outputs == [
            ("orch:orchestrator", [0]),
            ("floor:floor_0_agent", [0]),
            ("floor:floor_1_agent", [1]),
        ]
        assert len(orch.calls) == 1
        assert len(floor.calls) == 1


class TestScopeRoutedFloorPolicy:
    def test_routes_floor_prompts_by_disaster_and_preserves_batch_order(self):
        class FakePolicy:
            def __init__(self, label):
                self.label = label
                self.calls = []

            def act_batch(self, prompts, agent_ids, roles):
                self.calls.append((prompts, agent_ids, roles))
                return [(f"{self.label}:{agent_id}", [len(self.calls)]) for agent_id in agent_ids]

        fire = FakePolicy("fire")
        flood = FakePolicy("flood")
        gas = FakePolicy("gas")
        policy = ScopeRoutedFloorPolicy(
            specialist_policies={
                "fire": fire,
                "flood": flood,
                "gas": gas,
            }
        )

        outputs = policy.act_batch(
            prompts=[
                [{"role": "system", "content": "Disaster: gas\nRound: 0"}],
                [{"role": "system", "content": "Disaster: fire\nRound: 0"}],
                [{"role": "system", "content": "Disaster: flood\nRound: 0"}],
                [{"role": "system", "content": "Disaster: fire\nRound: 1"}],
            ],
            agent_ids=["floor_2_agent", "floor_0_agent", "floor_1_agent", "floor_3_agent"],
            roles=["floor_agent", "floor_agent", "floor_agent", "floor_agent"],
        )

        assert outputs == [
            ("gas:floor_2_agent", [1]),
            ("fire:floor_0_agent", [1]),
            ("flood:floor_1_agent", [1]),
            ("fire:floor_3_agent", [1]),
        ]
        assert len(fire.calls) == 1
        assert fire.calls[0][1] == ["floor_0_agent", "floor_3_agent"]
        assert len(flood.calls) == 1
        assert len(gas.calls) == 1

    def test_uses_generalist_for_unknown_or_missing_specialist(self):
        class FakePolicy:
            def __init__(self, label):
                self.label = label

            def act(self, prompt, agent_id, role):
                return f"{self.label}:{agent_id}", []

        policy = ScopeRoutedFloorPolicy(
            specialist_policies={"fire": FakePolicy("fire")},
            generalist_policy=FakePolicy("generalist"),
        )

        assert policy.act(
            [{"role": "system", "content": "Disaster: structural\nRound: 0"}],
            "floor_0_agent",
            "floor_agent",
        ) == ("generalist:floor_0_agent", [])
        assert policy.act(
            [{"role": "system", "content": "Disaster: flood\nRound: 0"}],
            "floor_1_agent",
            "floor_agent",
        ) == ("generalist:floor_1_agent", [])

    def test_unknown_route_without_generalist_fails_loudly(self):
        class FakePolicy:
            def act(self, prompt, agent_id, role):
                return "{}", []

        policy = ScopeRoutedFloorPolicy(specialist_policies={"fire": FakePolicy()})

        with pytest.raises(RuntimeError, match="No frozen floor policy"):
            policy.act(
                [{"role": "system", "content": "Disaster: flood\nRound: 0"}],
                "floor_0_agent",
                "floor_agent",
            )


class TestParseCompletionToAction:
    def test_invalid_json_returns_none_invalid_json(self):
        """parse_completion_to_action returns (None, 'invalid_json') on garbage text."""
        action, reason = parse_completion_to_action(
            "garbage text!!!",
            "orchestrator",
            "orchestrator",
            "ep1",
            0,
        )
        assert action is None
        assert reason == "invalid_json"

    def test_prose_wrapped_json_is_salvaged(self):
        completion = """I think the best move is:

        {
          "episode_id": "ep1",
          "round_id": 0,
          "agent_id": "orchestrator",
          "action_id": "a1",
          "action_type": "wait",
          "arguments": {}
        }
        """

        action, reason = parse_completion_to_action(
            completion,
            "orchestrator",
            "orchestrator",
            "ep1",
            0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.action_type.value == "wait"

    def test_fenced_json_is_salvaged(self):
        completion = """Sure, here is the action:

        ```json
        {
          "episode_id": "ep1",
          "round_id": 0,
          "agent_id": "floor_0_agent",
          "action_id": "a2",
          "action_type": "wait",
          "arguments": {}
        }
        ```
        """

        action, reason = parse_completion_to_action(
            completion,
            "floor_0_agent",
            "floor_agent",
            "ep1",
            0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.agent_id == "floor_0_agent"

    def test_role_forbidden_floor_broadcast_directive(self):
        """parse_completion_to_action returns (None, 'role_forbidden') when a floor agent
        emits broadcast_directive."""
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "broadcast_directive",
            "arguments": {
                "directive": {
                    "directive_id": "d1",
                    "target": "floor_0",
                    "directive_type": "evacuation_priority",
                    "params": {},
                }
            },
        })
        action, reason = parse_completion_to_action(
            completion,
            "floor_0_agent",
            "floor_agent",
            "ep1",
            0,
        )
        assert action is None
        assert reason == "role_forbidden"

    def test_parse_completion_overwrites_model_supplied_identity(self):
        completion = json.dumps({
            "episode_id": "wrong",
            "round_id": 999,
            "agent_id": "fake_agent",
            "action_id": "a1",
            "action_type": "wait",
            "arguments": {},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="real_episode",
            expected_round_id=3,
        )

        assert reason == "ok"
        assert action is not None
        assert action.episode_id == "real_episode"
        assert action.agent_id == "floor_0_agent"
        assert action.round_id == 3

    def test_parse_completion_salvages_null_client_metadata(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "orchestrator",
            "action_id": "a1",
            "action_type": "wait",
            "arguments": {},
            "client_metadata": None,
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="orchestrator",
            role="orchestrator",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.client_metadata == {}

    def test_parse_completion_salvages_string_round_ids(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": "ep1_round_7",
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "wait",
            "arguments": {},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="ep1",
            expected_round_id=7,
        )

        assert reason == "ok"
        assert action is not None
        assert action.round_id == 7

    def test_parse_completion_rejects_active_action_with_empty_arguments(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "route_within_floor",
            "arguments": {},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert action is None
        assert reason == "arguments_invalid"

    def test_parse_completion_accepts_route_with_explicit_exit_id(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "route_within_floor",
            "arguments": {"from_room_id": "room_01", "exit_id": "exit_0"},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.arguments["exit_id"] == "exit_0"

    def test_parse_completion_repairs_literal_action_type_route_hint(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "route_to_stair",
            "action_type": "action_type",
            "arguments": {"from_room_id": "room_01", "stairwell_id": "SW0"},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.action_type == ActionTypeMA.route_within_floor
        assert action.arguments["stairwell_id"] == "SW0"

    def test_parse_completion_normalizes_priority_floor_alias(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "orchestrator",
            "action_id": "orch_priority",
            "action_type": "evacuate_floor_priority",
            "arguments": {"priority_floor": "floor_1"},
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="orchestrator",
            role="orchestrator",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.arguments["ordered_floor_ids"] == ["floor_1"]

    def test_parse_completion_unwraps_nested_priority_arguments(self, caplog):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "orchestrator",
            "action_id": "orch_priority",
            "action_type": "evacuate_floor_priority",
            "arguments": {
                "evacuate_floor_priority_arguments": {
                    "ordered_floor_ids": ["floor_2", "floor_1"],
                },
            },
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="orchestrator",
            role="orchestrator",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert reason == "ok"
        assert action is not None
        assert action.arguments == {"ordered_floor_ids": ["floor_2", "floor_1"]}
        assert "Unwrapped nested evacuate_floor_priority_arguments" in caplog.text

    def test_parse_completion_rejects_active_action_with_non_dict_arguments(self):
        completion = json.dumps({
            "episode_id": "ep1",
            "round_id": 0,
            "agent_id": "floor_0_agent",
            "action_id": "a1",
            "action_type": "open_exit",
            "arguments": None,
        })

        action, reason = parse_completion_to_action(
            completion,
            agent_id="floor_0_agent",
            role="floor_agent",
            expected_episode_id="ep1",
            expected_round_id=0,
        )

        assert action is None
        assert reason == "arguments_invalid"
