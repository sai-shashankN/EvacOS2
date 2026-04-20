from __future__ import annotations

from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from evacos_ma.models import (
    Action,
    ActionRecord,
    ActionType,
    BaselineRun,
    BlockRouteAction,
    Building,
    CallElevatorAction,
    Corridor,
    DisasterType,
    EdgeObservation,
    EdgeRef,
    Elevator,
    ElevatorRequest,
    EpisodeMetrics,
    EpisodeStateInternal,
    ErrorResponse,
    EventSummary,
    EventType,
    EvacuateFloorAction,
    Exit,
    ExitObservation,
    ExitType,
    Floor,
    HazardState,
    HazardType,
    LockdownRoomAction,
    MetricsDelta,
    Observation,
    Occupancy,
    OpenExitAction,
    PrioritizeRoomAction,
    Rect,
    RenderObservation,
    RequestRenderAction,
    Reward,
    RewardWeights,
    Room,
    RoomObservation,
    RouteCiviliansAction,
    ScheduledEvent,
    Stairwell,
    StairwellObservation,
    StateView,
    StepInfo,
    SummaryObservation,
    TaskSpec,
    TaskSpecPublic,
    TerminationReason,
    ThreatState,
    TransitGroup,
    WaitAction,
)
from evacos_ma.task_registry import TASKS, get_all_tasks, get_task, get_tasks_public


def test_imports_and_symbols_available() -> None:
    imported_symbols = [
        Action,
        ActionRecord,
        ActionType,
        BaselineRun,
        BlockRouteAction,
        Building,
        CallElevatorAction,
        Corridor,
        DisasterType,
        EdgeObservation,
        EdgeRef,
        Elevator,
        ElevatorRequest,
        EpisodeMetrics,
        EpisodeStateInternal,
        ErrorResponse,
        EventSummary,
        EventType,
        EvacuateFloorAction,
        Exit,
        ExitObservation,
        ExitType,
        Floor,
        HazardState,
        HazardType,
        LockdownRoomAction,
        MetricsDelta,
        Observation,
        Occupancy,
        OpenExitAction,
        PrioritizeRoomAction,
        Rect,
        RenderObservation,
        RequestRenderAction,
        Reward,
        RewardWeights,
        Room,
        RoomObservation,
        RouteCiviliansAction,
        ScheduledEvent,
        Stairwell,
        StairwellObservation,
        StateView,
        StepInfo,
        SummaryObservation,
        TaskSpec,
        TaskSpecPublic,
        TerminationReason,
        ThreatState,
        TransitGroup,
        WaitAction,
    ]
    assert all(imported_symbols)


@pytest.mark.parametrize(
    ("action_cls", "payload"),
    [
        (
            RouteCiviliansAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.route_civilians,
                "from_node_id": "room-a",
                "to_node_id": "room-b",
                "occupancy": {"mobile": 2, "injured": 1, "mobility_impaired": 0},
                "preference": "safest",
            },
        ),
        (
            EvacuateFloorAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.evacuate_floor,
                "floor_id": 2,
                "preferred_exit_id": "exit-1",
            },
        ),
        (
            PrioritizeRoomAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.prioritize_room,
                "room_id": "room-a",
                "priority": "injured_first",
            },
        ),
        (
            BlockRouteAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.block_route,
                "edge_id": "corridor-1",
            },
        ),
        (
            CallElevatorAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.call_elevator,
                "elevator_id": "elevator-1",
                "source_floor": 1,
                "target_floor": 4,
            },
        ),
        (
            OpenExitAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.open_exit,
                "exit_id": "exit-1",
            },
        ),
        (
            LockdownRoomAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.lockdown_room,
                "room_id": "room-a",
            },
        ),
        (
            RequestRenderAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.request_render,
                "floor_id": 3,
            },
        ),
        (
            WaitAction,
            {
                "episode_id": "ep-1",
                "expected_step": 1,
                "action_type": ActionType.wait,
                "reason": "synchronizing movement",
            },
        ),
    ],
)
def test_action_variants_construct_with_expected_action_type(action_cls, payload) -> None:
    action = action_cls(**payload)
    assert action.action_type == payload["action_type"]


@pytest.mark.parametrize(
    ("action_type", "expected_cls", "extra_payload"),
    [
        (
            ActionType.route_civilians,
            RouteCiviliansAction,
            {
                "from_node_id": "room-a",
                "to_node_id": "room-b",
                "occupancy": {"mobile": 1},
            },
        ),
        (
            ActionType.evacuate_floor,
            EvacuateFloorAction,
            {"floor_id": 2},
        ),
        (
            ActionType.prioritize_room,
            PrioritizeRoomAction,
            {"room_id": "room-a"},
        ),
        (
            ActionType.block_route,
            BlockRouteAction,
            {"edge_id": "corridor-1"},
        ),
        (
            ActionType.call_elevator,
            CallElevatorAction,
            {"elevator_id": "elevator-1", "source_floor": 1, "target_floor": 2},
        ),
        (
            ActionType.open_exit,
            OpenExitAction,
            {"exit_id": "exit-1"},
        ),
        (
            ActionType.lockdown_room,
            LockdownRoomAction,
            {"room_id": "room-a"},
        ),
        (
            ActionType.request_render,
            RequestRenderAction,
            {"floor_id": 0},
        ),
        (
            ActionType.wait,
            WaitAction,
            {},
        ),
    ],
)
def test_action_discriminated_union_dispatches(action_type, expected_cls, extra_payload) -> None:
    adapter = TypeAdapter(Action)
    action = adapter.validate_python(
        {
            "episode_id": "ep-1",
            "expected_step": 0,
            "action_type": action_type,
            **extra_payload,
        }
    )
    assert isinstance(action, expected_cls)


def test_action_alias_contains_all_variants() -> None:
    union_type = get_args(Action)[0]
    variants = set(get_args(union_type))
    assert variants == {
        RouteCiviliansAction,
        EvacuateFloorAction,
        PrioritizeRoomAction,
        BlockRouteAction,
        CallElevatorAction,
        OpenExitAction,
        LockdownRoomAction,
        RequestRenderAction,
        WaitAction,
    }


def test_occupancy_total_property() -> None:
    occupancy = Occupancy(mobile=3, injured=2, mobility_impaired=1)
    assert occupancy.total == 6


def test_reward_total_matches_component_sum() -> None:
    reward = Reward(
        total=4.4,
        civilians_saved_delta=2,
        civilians_lost_delta=-1,
        hazard_avoidance_bonus=0.5,
        vulnerable_group_bonus=0.4,
        efficiency_bonus=0.3,
        invalid_action_penalty=-0.2,
        idle_penalty=-0.1,
        completion_bonus=2.5,
    )
    component_sum = (
        reward.civilians_saved_delta
        + reward.civilians_lost_delta
        + reward.hazard_avoidance_bonus
        + reward.vulnerable_group_bonus
        + reward.efficiency_bonus
        + reward.invalid_action_penalty
        + reward.idle_penalty
        + reward.completion_bonus
    )
    assert reward.total == pytest.approx(component_sum)


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        Occupancy(mobile=1, unexpected=2)


def test_task_spec_defaults_and_registry_functions() -> None:
    task = TaskSpec(
        task_id="custom",
        name="Custom",
        difficulty="easy",
        disaster_type=DisasterType.fire,
        building_profile="small",
        success_criteria="save all",
        goal="ground_exit",
        max_steps=10,
    )
    assert task.evaluation_seeds == [42, 123, 456]
    assert task.reward_weights == RewardWeights()
    assert get_task("task_1_fire_easy").task_id == "task_1_fire_easy"
    assert len(get_all_tasks()) >= 4
    public_tasks = get_tasks_public()
    assert len(public_tasks) == len(get_all_tasks())
    assert all(isinstance(item, TaskSpecPublic) for item in public_tasks)


def test_get_task_raises_for_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("missing-task")


def test_task_registry_contains_all_expected_tasks() -> None:
    expected_ids = {
        "task_1_fire_easy",
        "task_2_flood_medium",
        "task_3_earthquake_hard",
        "task_4_cascade_hard",
        "task_lh_fire_easy",
        "task_lh_flood_medium",
        "task_lh_cascade_hard",
        "task_lh_cascade_brutal",
    }
    assert expected_ids.issubset(set(TASKS))
    assert len(TASKS) >= 8

    for task_id in expected_ids:
        task = TASKS[task_id]
        assert task.task_id == task_id
        assert task.name
        assert task.max_steps > 0
        assert len(task.evaluation_seeds) >= 3
        assert len(task.expected_score_range) == 2
        assert 0.0 < task.expected_score_range[0] < task.expected_score_range[1] < 1.0
