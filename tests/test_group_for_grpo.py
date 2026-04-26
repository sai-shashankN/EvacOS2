from __future__ import annotations

from types import SimpleNamespace

from curriculum.controller import CurriculumController
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    CivilianGroupView,
    CorridorView,
    ExitView,
    FloorAgentObservationMA,
    RoomView,
    StairwellEntryView,
)
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from training.policy_adapter import StubPolicy
from training.rollout import collect_batch
from training.rollout import (
    _floor_candidate_reward,
    _floor_oracle_candidate_payload,
    _prompt_scoped_group_id,
    _select_floor_candidate,
)
from training.train import _group_for_grpo


def _sample(
    *,
    group_id: str,
    prompt_text: str,
    completion_text: str,
    raw_reward: float,
    normalized_reward: float,
):
    return SimpleNamespace(
        group_id=group_id,
        prompt=[{"role": "user", "content": prompt_text}],
        completion_text=completion_text,
        completion_token_ids=None,
        raw_reward=raw_reward,
        normalized_reward=normalized_reward,
    )


def test_group_for_grpo_uses_deterministic_key_order_without_exposing_group_names():
    grouped = _group_for_grpo(
        [
            SimpleNamespace(
                samples=[
                    _sample(
                        group_id="ep_demo_r_2_floor",
                        prompt_text="prompt-2a",
                        completion_text="completion-2a",
                        raw_reward=2.0,
                        normalized_reward=0.2,
                    ),
                    _sample(
                        group_id="ep_demo_r_10_floor",
                        prompt_text="prompt-10a",
                        completion_text="completion-10a",
                        raw_reward=10.0,
                        normalized_reward=1.0,
                    ),
                ]
            ),
            SimpleNamespace(
                samples=[
                    _sample(
                        group_id="ep_demo_r_2_floor",
                        prompt_text="prompt-2b",
                        completion_text="completion-2b",
                        raw_reward=3.0,
                        normalized_reward=0.3,
                    ),
                    _sample(
                        group_id="rollout_orchestrator",
                        prompt_text="prompt-orch",
                        completion_text="completion-orch",
                        raw_reward=1.0,
                        normalized_reward=0.1,
                    ),
                ]
            ),
        ]
    )

    # Lexicographic ordering is intentional: "r_10" sorts before "r_2".
    assert grouped["completions"] == [
        ["completion-10a"],
        ["completion-2a", "completion-2b"],
        ["completion-orch"],
    ]
    assert grouped["raw_rewards"] == [[10.0], [2.0, 3.0], [1.0]]
    assert "group_ids" not in grouped


def test_prompt_scoped_group_id_keeps_same_prompt_candidates_together():
    prompt = [{"role": "user", "content": "same floor observation"}]

    first = _prompt_scoped_group_id(
        episode_id="ep_same",
        round_id=2,
        role="floor_agent",
        agent_id="floor_0_agent",
        prompt=prompt,
    )
    second = _prompt_scoped_group_id(
        episode_id="ep_same",
        round_id=2,
        role="floor_agent",
        agent_id="floor_0_agent",
        prompt=list(prompt),
    )
    different_prompt = _prompt_scoped_group_id(
        episode_id="ep_same",
        round_id=2,
        role="floor_agent",
        agent_id="floor_0_agent",
        prompt=[{"role": "user", "content": "different floor observation"}],
    )

    assert first == second
    assert first != different_prompt


def test_floor_candidate_reward_prefers_evacuation_actions_over_waiting():
    obs = FloorAgentObservationMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_mask=[
            "route_within_floor",
            "open_exit",
            "scout",
            "wait",
        ],
        floor_id="floor_0",
        visible_rooms=[
            RoomView(
                room_id="room_001",
                floor_id="floor_0",
                occupancy_mobile=3,
                hazard_severity=0.4,
                smoke_level=0.3,
            )
        ],
        exits_on_floor=[
            ExitView(
                exit_id="exit_floor_0",
                floor_id=0,
                requires_open_action=True,
            )
        ],
        stairwell_entries=[
            StairwellEntryView(
                stairwell_id="stairwell_0",
                connects_floor_ids=[0, 1],
            )
        ],
        visible_civilian_groups=[
            CivilianGroupView(
                civilian_group_id="civ_1",
                location_room_id="room_001",
                count=3,
            )
        ],
    )
    route = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="a1",
        action_type=ActionTypeMA.route_within_floor,
        arguments={"from_room_id": "room_001", "exit_id": "exit_floor_0"},
    )
    wait = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="a2",
        action_type=ActionTypeMA.wait,
        arguments={},
    )

    assert _floor_candidate_reward(obs, route, "ok") > 0.8
    assert _floor_candidate_reward(obs, wait, "ok") < 0
    assert _floor_candidate_reward(obs, None, "parse_error") == -1.0


def test_floor_candidate_reward_prefers_egress_route_over_room_route_when_available():
    obs = FloorAgentObservationMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_mask=["route_within_floor", "wait"],
        floor_id="floor_0",
        visible_rooms=[
            RoomView(room_id="room_001", floor_id="floor_0", occupancy_mobile=3, hazard_severity=0.4),
            RoomView(room_id="room_002", floor_id="floor_0", occupancy_mobile=0, hazard_severity=0.0),
        ],
        exits_on_floor=[ExitView(exit_id="exit_floor_0", floor_id=0)],
        visible_civilian_groups=[
            CivilianGroupView(civilian_group_id="civ_1", location_room_id="room_001", count=3)
        ],
    )
    room_route = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="room_route",
        action_type=ActionTypeMA.route_within_floor,
        arguments={"from_room_id": "room_001", "to_room_id": "room_002"},
    )
    exit_route = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="exit_route",
        action_type=ActionTypeMA.route_within_floor,
        arguments={"from_room_id": "room_001", "exit_id": "exit_floor_0"},
    )

    assert _floor_candidate_reward(obs, exit_route, "ok") > _floor_candidate_reward(
        obs, room_route, "ok"
    )


def test_floor_candidate_reward_penalizes_room_route_into_unsafe_corridor_node():
    base_obs = dict(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_mask=["route_within_floor", "wait"],
        floor_id="floor_0",
        visible_rooms=[
            RoomView(room_id="room_001", floor_id="floor_0", occupancy_mobile=3, hazard_severity=0.4),
            RoomView(room_id="room_002", floor_id="floor_0", occupancy_mobile=0, hazard_severity=0.0),
        ],
        exits_on_floor=[ExitView(exit_id="exit_floor_0", floor_id=0)],
        visible_civilian_groups=[
            CivilianGroupView(civilian_group_id="civ_1", location_room_id="room_001", count=3)
        ],
    )
    clear_obs = FloorAgentObservationMA(**base_obs)
    unsafe_corridor_obs = FloorAgentObservationMA(
        **{
            **base_obs,
            "visible_corridors": [
                CorridorView(
                    corridor_id="corridor_gas_1",
                    from_node_id="room_001",
                    to_node_id="room_002",
                    hazard_severity=0.9,
                    passable=False,
                )
            ],
        }
    )
    room_route = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="room_route",
        action_type=ActionTypeMA.route_within_floor,
        arguments={"from_room_id": "room_001", "to_room_id": "room_002"},
    )

    assert _floor_candidate_reward(unsafe_corridor_obs, room_route, "ok") < _floor_candidate_reward(
        clear_obs, room_route, "ok"
    )


def test_floor_candidate_reward_penalizes_open_exit_when_exit_is_already_open():
    obs = FloorAgentObservationMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_mask=["route_within_floor", "open_exit", "wait"],
        floor_id="floor_0",
        visible_rooms=[RoomView(room_id="room_001", floor_id="floor_0", occupancy_mobile=3)],
        exits_on_floor=[
            ExitView(exit_id="exit_floor_0", floor_id=0, blocked=False, requires_open_action=False)
        ],
        visible_civilian_groups=[
            CivilianGroupView(civilian_group_id="civ_1", location_room_id="room_001", count=3)
        ],
    )
    open_exit = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="open",
        action_type=ActionTypeMA.open_exit,
        arguments={"exit_id": "exit_floor_0"},
    )
    wait = ActionEnvelopeMA(
        episode_id="ep_reward",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="wait",
        action_type=ActionTypeMA.wait,
        arguments={},
    )

    assert _floor_candidate_reward(obs, open_exit, "ok") < _floor_candidate_reward(obs, wait, "ok")


def test_floor_oracle_candidate_uses_exact_visible_exit_route():
    obs = FloorAgentObservationMA(
        episode_id="ep_oracle",
        round_id=2,
        agent_id="floor_0_agent",
        action_mask=["route_within_floor", "wait"],
        floor_id="floor_0",
        visible_rooms=[RoomView(room_id="room_001", floor_id="floor_0", occupancy_mobile=3)],
        exits_on_floor=[ExitView(exit_id="exit_floor_0", floor_id=0, blocked=False)],
        visible_civilian_groups=[
            CivilianGroupView(civilian_group_id="civ_1", location_room_id="room_001", count=3)
        ],
    )

    completion, token_ids = _floor_oracle_candidate_payload(obs)

    assert token_ids == []
    assert '"action_id":"oracle_route_exit"' in completion
    assert '"from_room_id":"room_001"' in completion
    assert '"exit_id":"exit_floor_0"' in completion
    assert '"to_room_id"' not in completion


def test_collect_batch_candidate_groups_mark_selected_candidate_reward(tmp_path):
    results = collect_batch(
        EvacEnvironment(),
        StubPolicy(seed=1),
        CurriculumController(),
        num_episodes=1,
        seed_generator=lambda: 20260426,
        disaster_families=[DisasterType.fire],
        max_rounds=1,
        checkpoint_tag="candidate-smoke",
        model_name="stub",
        jsonl_dir=tmp_path,
        cleanup_env_episodes=True,
        candidates_per_floor_prompt=4,
    )

    result = results[0]
    floor_groups: dict[str, list[object]] = {}
    for sample in result.samples:
        if sample.role == "floor_agent":
            floor_groups.setdefault(sample.group_id, []).append(sample)

    assert sorted(len(samples) for samples in floor_groups.values()) == [4, 4, 4, 4, 4]
    for samples in floor_groups.values():
        prompts = [sample.prompt for sample in samples]
        assert prompts == [prompts[0]] * 4
        for index, sample in enumerate(samples):
            assert sample.parsed_action["candidate_index"] == index
        selected = [sample for sample in samples if sample.parsed_action["selected_for_execution"]]
        assert len(selected) == 1
        assert selected[0].raw_reward == result.total_raw_reward_by_role[selected[0].agent_id]
        assert selected[0].normalized_reward == result.total_normalized_reward_by_role[selected[0].agent_id]


def test_select_floor_candidate_prefers_best_scored_route_over_wait():
    obs = FloorAgentObservationMA(
        episode_id="ep_select",
        round_id=0,
        agent_id="floor_0_agent",
        action_mask=[
            "route_within_floor",
            "open_exit",
            "scout",
            "wait",
        ],
        floor_id="floor_0",
        visible_rooms=[
            RoomView(
                room_id="room_001",
                floor_id="floor_0",
                occupancy_mobile=3,
                hazard_severity=0.4,
                smoke_level=0.3,
            )
        ],
        exits_on_floor=[
            ExitView(
                exit_id="exit_floor_0",
                floor_id=0,
                requires_open_action=True,
            )
        ],
        visible_civilian_groups=[
            CivilianGroupView(
                civilian_group_id="civ_1",
                location_room_id="room_001",
                count=3,
            )
        ],
    )
    wait = ActionEnvelopeMA(
        episode_id="ep_select",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="bad_wait",
        action_type=ActionTypeMA.wait,
        arguments={},
    )
    route = ActionEnvelopeMA(
        episode_id="ep_select",
        round_id=0,
        agent_id="floor_0_agent",
        action_id="good_route",
        action_type=ActionTypeMA.route_within_floor,
        arguments={"from_room_id": "room_001", "exit_id": "exit_floor_0"},
    )

    selected_index, selected_action, selected_status = _select_floor_candidate(
        obs,
        [wait, route],
        ["ok", "ok"],
    )

    assert selected_index == 1
    assert selected_action is route
    assert selected_status == "ok"
