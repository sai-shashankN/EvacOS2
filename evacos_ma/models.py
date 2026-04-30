from __future__ import annotations

from enum import Enum
from math import isclose
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvacBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ScoreValue = Annotated[float, Field(gt=0.0, lt=1.0)]


class DisasterType(str, Enum):
    fire = "fire"
    flood = "flood"
    gas = "gas"
    structural = "structural"
    active_threat = "active_threat"
    multi_cascade = "multi_cascade"


class HazardType(str, Enum):
    fire = "fire"
    flood = "flood"
    gas = "gas"
    structural = "structural"
    threat = "threat"


class ActionType(str, Enum):
    route_civilians = "route_civilians"
    evacuate_floor = "evacuate_floor"
    prioritize_room = "prioritize_room"
    block_route = "block_route"
    call_elevator = "call_elevator"
    open_exit = "open_exit"
    lockdown_room = "lockdown_room"
    request_render = "request_render"
    wait = "wait"


class ExitType(str, Enum):
    ground = "ground"
    rooftop = "rooftop"
    emergency_window = "emergency_window"


class TerminationReason(str, Enum):
    all_saved = "all_saved"
    all_routes_cut = "all_routes_cut"
    max_steps = "max_steps"
    all_lost = "all_lost"


class EventType(str, Enum):
    gas_rupture = "gas_rupture"
    stairwell_collapse = "stairwell_collapse"
    power_outage = "power_outage"
    fire_ignition = "fire_ignition"
    structural_collapse = "structural_collapse"
    flood_rise = "flood_rise"
    gas_spread = "gas_spread"
    explosion = "explosion"
    threat_move = "threat_move"
    civilian_loss = "civilian_loss"


class Rect(EvacBaseModel):
    x: int
    y: int
    w: int
    h: int


class EdgeRef(EvacBaseModel):
    from_id: str
    to_id: str
    edge_type: Literal["corridor", "stairwell", "elevator"]


class ElevatorRequest(EvacBaseModel):
    floor_id: int
    direction: Literal["up", "down"]


class RewardWeights(EvacBaseModel):
    civilian_saved: float = 2.0
    civilian_lost: float = -5.0
    hazard_avoidance: float = 0.3
    vulnerable_bonus: float = 0.5
    efficiency: float = 0.2
    invalid_action: float = -0.75
    idle: float = -0.1
    time_step: float = -0.1
    completion: float = 5.0
    floor_prediction: float = 0.3
    floor_scout_cost: float = 1.0
    duplicate_belief_penalty: float = 1.0
    team_progress_dense_floor: float = 0.5
    team_progress_dense_orchestrator: float = 0.8
    floor_saved: float = 1.0
    floor_lost: float = -1.0
    floor_invalid_action: float = -1.0
    floor_parse_error: float = -1.0
    orchestrator_parse_error: float = -0.75
    floor_parse_salvage_penalty: float = -0.15
    orchestrator_parse_salvage_penalty: float = -0.15
    total_saved_terminal: float = 1.0
    total_lost_terminal: float = -1.0
    coordination_bonus: float = -0.3
    directive_quality: float = 0.3
    priority_top_match: float = 0.4
    priority_rank_score: float = 0.2
    priority_coverage: float = 0.10
    priority_duplicate_or_unknown_penalty: float = -0.25
    priority_effect_bonus: float = 0.15
    priority_unchanged_penalty: float = -0.08


class ThreatState(EvacBaseModel):
    current_room_id: str
    target_room_id: Optional[str] = None
    steps_since_move: int
    movement_interval: int = 2


class ActionRecord(EvacBaseModel):
    step: int
    action_type: ActionType
    raw_action: dict[str, Any]
    valid: bool
    reward_total: float


class IncidentOutcomes(EvacBaseModel):
    safe: int = 0
    mild_injury: int = 0
    severe_injury: int = 0
    deaths: int = 0

    @property
    def total(self) -> int:
        return self.safe + self.mild_injury + self.severe_injury + self.deaths


class EpisodeMetrics(EvacBaseModel):
    actions_by_type: dict[str, int] = Field(default_factory=dict)
    invalid_actions: int = 0
    render_requests: int = 0
    civilians_saved_mobile: int = 0
    civilians_saved_injured: int = 0
    civilians_saved_impaired: int = 0
    civilians_lost_mobile: int = 0
    civilians_lost_injured: int = 0
    civilians_lost_impaired: int = 0
    incident_safe: int = 0
    incident_mild_injury: int = 0
    incident_severe_injury: int = 0
    incident_deaths: int = 0
    cumulative_hazard_exposure: float = 0.0
    elapsed_ms: float = 0.0


class MetricsDelta(EvacBaseModel):
    civilians_saved_delta: int = 0
    civilians_lost_delta: int = 0
    hazard_exposure_delta: float = 0.0


class EventSummary(EvacBaseModel):
    event_id: str
    event_type: EventType
    target_id: str
    description: str


class BaselineRun(EvacBaseModel):
    task_id: str
    seed: int
    episode_id: str
    steps: int
    score: ScoreValue
    total_reward: float
    termination_reason: Optional[str] = None


class TaskSpecPublic(EvacBaseModel):
    task_id: str
    name: str
    difficulty: str
    disaster_type: DisasterType
    goal: str
    description: str
    max_steps: int
    expected_score_range: list[ScoreValue]


class Occupancy(EvacBaseModel):
    mobile: int = 0
    injured: int = 0
    mobility_impaired: int = 0

    @property
    def total(self) -> int:
        return self.mobile + self.injured + self.mobility_impaired


class HazardState(EvacBaseModel):
    hazard_type: Optional[HazardType] = None
    severity: float = 0.0
    smoke: float = 0.0
    water_level: float = 0.0
    structural_integrity: float = 1.0
    passable: bool = True


class Room(EvacBaseModel):
    room_id: str
    floor_id: int
    room_type: Literal["office", "lab", "hall", "utility", "shelter"] = "office"
    geometry: Rect
    capacity: int = 20
    occupancy: Occupancy = Field(default_factory=Occupancy)
    hazard: HazardState = Field(default_factory=HazardState)
    visibility: float = 1.0
    accessible: bool = True
    adjacent_node_ids: list[str] = Field(default_factory=list)


class Corridor(EvacBaseModel):
    corridor_id: str
    from_node_id: str
    to_node_id: str
    length_cost: int = 1
    hazard: HazardState = Field(default_factory=HazardState)


class Stairwell(EvacBaseModel):
    stairwell_id: str
    floor_ids: list[int]
    capacity_per_step: int = 5
    blocked: bool = False
    entry_room_ids: dict[int, str] = Field(default_factory=dict)


class Elevator(EvacBaseModel):
    elevator_id: str
    floor_ids: list[int]
    current_floor: int = 0
    target_floor: Optional[int] = None
    operational: bool = True
    capacity: int = 6
    travel_time_per_floor: int = 2
    queue: list[ElevatorRequest] = Field(default_factory=list)


class Exit(EvacBaseModel):
    exit_id: str
    floor_id: int
    exit_type: ExitType = ExitType.ground
    adjacent_room_id: str
    blocked: bool = False
    requires_open_action: bool = False


class Floor(EvacBaseModel):
    floor_id: int
    width: int = 800
    height: int = 400
    rooms: list[Room] = Field(default_factory=list)
    corridors: list[Corridor] = Field(default_factory=list)
    stairwells: list[Stairwell] = Field(default_factory=list)
    exits: list[Exit] = Field(default_factory=list)
    elevators: list[Elevator] = Field(default_factory=list)


class Building(EvacBaseModel):
    building_id: str
    seed: int
    floors: list[Floor] = Field(default_factory=list)
    graph_edges: list[EdgeRef] = Field(default_factory=list)
    disaster_zones: list[str] = Field(default_factory=list)


class TransitGroup(EvacBaseModel):
    transit_id: str
    source_node_id: str
    target_node_id: str
    occupancy: Occupancy
    incident_outcomes: IncidentOutcomes = Field(default_factory=IncidentOutcomes)
    path_kind: Literal["corridor", "stairwell", "elevator"]
    carrier_id: Optional[str] = None
    steps_remaining: int = 1


class ScheduledEvent(EvacBaseModel):
    event_id: str
    trigger_step: int
    event_type: EventType
    target_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    triggered: bool = False


class SummaryObservation(EvacBaseModel):
    disaster_type: DisasterType
    disaster_origin: str
    goal: str
    total_civilians: int
    civilians_saved: int
    civilians_lost: int
    civilians_in_transit: int
    elevators_operational: int
    incident_outcomes: IncidentOutcomes = Field(default_factory=IncidentOutcomes)


class RoomObservation(EvacBaseModel):
    room_id: str
    floor_id: int
    occupancy: Occupancy
    hazard_type: Optional[HazardType] = None
    hazard_severity: float = 0.0
    smoke: float = 0.0
    water_level: float = 0.0
    structural_integrity: float = 1.0
    passable: bool = True
    accessible: bool = True
    connected_rooms: list[str] = Field(default_factory=list)


class EdgeObservation(EvacBaseModel):
    corridor_id: str
    from_node_id: str
    to_node_id: str
    hazard_severity: float = 0.0
    passable: bool = True


class StairwellObservation(EvacBaseModel):
    stairwell_id: str
    floor_ids: list[int]
    blocked: bool = False
    capacity_per_step: int = 5


class ExitObservation(EvacBaseModel):
    exit_id: str
    floor_id: int
    exit_type: ExitType
    blocked: bool = False
    requires_open_action: bool = False


class ElevatorObservation(EvacBaseModel):
    elevator_id: str
    current_floor: int
    target_floor: Optional[int] = None
    operational: bool = True


class TransitGroupObservation(EvacBaseModel):
    transit_id: str
    source_node_id: str
    target_node_id: str
    occupancy: Occupancy
    steps_remaining: int


class RenderObservation(EvacBaseModel):
    floor_id: int
    image_base64: str


class Observation(EvacBaseModel):
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    summary: SummaryObservation
    rooms: list[RoomObservation]
    corridors: list[EdgeObservation]
    stairwells: list[StairwellObservation]
    exits: list[ExitObservation]
    render: Optional[RenderObservation] = None


class BaseAction(EvacBaseModel):
    episode_id: str
    expected_step: int
    action_type: ActionType


class RouteCiviliansAction(BaseAction):
    action_type: Literal[ActionType.route_civilians] = ActionType.route_civilians
    from_node_id: str
    to_node_id: str
    occupancy: Occupancy
    preference: Literal["fastest", "safest", "injured_first"] = "fastest"


class EvacuateFloorAction(BaseAction):
    action_type: Literal[ActionType.evacuate_floor] = ActionType.evacuate_floor
    floor_id: int
    preferred_exit_id: Optional[str] = None


class PrioritizeRoomAction(BaseAction):
    action_type: Literal[ActionType.prioritize_room] = ActionType.prioritize_room
    room_id: str
    priority: Literal["injured_first", "all", "mobile_only"] = "all"


class BlockRouteAction(BaseAction):
    action_type: Literal[ActionType.block_route] = ActionType.block_route
    edge_id: str


class CallElevatorAction(BaseAction):
    action_type: Literal[ActionType.call_elevator] = ActionType.call_elevator
    elevator_id: str
    source_floor: int
    target_floor: int


class OpenExitAction(BaseAction):
    action_type: Literal[ActionType.open_exit] = ActionType.open_exit
    exit_id: str


class LockdownRoomAction(BaseAction):
    action_type: Literal[ActionType.lockdown_room] = ActionType.lockdown_room
    room_id: str


class RequestRenderAction(BaseAction):
    action_type: Literal[ActionType.request_render] = ActionType.request_render
    floor_id: int


class WaitAction(BaseAction):
    action_type: Literal[ActionType.wait] = ActionType.wait
    reason: Optional[str] = None


Action = Annotated[
    Union[
        RouteCiviliansAction,
        EvacuateFloorAction,
        PrioritizeRoomAction,
        BlockRouteAction,
        CallElevatorAction,
        OpenExitAction,
        LockdownRoomAction,
        RequestRenderAction,
        WaitAction,
    ],
    Field(discriminator="action_type"),
]


class Reward(EvacBaseModel):
    total: float
    civilians_saved_delta: int = 0
    civilians_lost_delta: int = 0
    hazard_avoidance_bonus: float = 0.0
    vulnerable_group_bonus: float = 0.0
    efficiency_bonus: float = 0.0
    invalid_action_penalty: float = 0.0
    idle_penalty: float = 0.0
    completion_bonus: float = 0.0

    @model_validator(mode="after")
    def validate_total(self) -> Reward:
        expected_total = (
            self.civilians_saved_delta
            + self.civilians_lost_delta
            + self.hazard_avoidance_bonus
            + self.vulnerable_group_bonus
            + self.efficiency_bonus
            + self.invalid_action_penalty
            + self.idle_penalty
            + self.completion_bonus
        )
        if not isclose(self.total, expected_total, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"Reward.total must equal the sum of components ({expected_total})"
            )
        return self


class StepInfo(EvacBaseModel):
    termination_reason: Optional[TerminationReason] = None
    invalid_action: bool = False
    invalid_reason: Optional[str] = None
    triggered_events: list[EventSummary] = Field(default_factory=list)
    metrics_delta: MetricsDelta = Field(default_factory=MetricsDelta)
    normalized_reward: float = 0.0


class ErrorResponse(EvacBaseModel):
    error_code: str
    message: str
    episode_id: Optional[str] = None


class StateView(EvacBaseModel):
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    done: bool
    termination_reason: Optional[TerminationReason] = None
    summary: SummaryObservation
    rooms: list[RoomObservation]
    corridors: list[EdgeObservation]
    stairwells: list[StairwellObservation]
    exits: list[ExitObservation]
    elevators: list[ElevatorObservation] = Field(default_factory=list)
    transit_groups: list[TransitGroupObservation] = Field(default_factory=list)
    blocked_route_ids: list[str] = Field(default_factory=list)


class TaskSpec(EvacBaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "medium_hard", "hard", "brutal"]
    disaster_type: DisasterType
    building_profile: str
    success_criteria: str
    goal: str
    max_steps: int
    evaluation_seeds: list[int] = Field(default_factory=lambda: [42, 123, 456])
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    description: str = ""
    expected_score_range: list[ScoreValue] = Field(
        default_factory=lambda: [0.001, 0.999]
    )


class EpisodeStateInternal(EvacBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    episode_id: str
    task: TaskSpec
    building: Building
    seed: int
    step: int = 0
    done: bool = False
    termination_reason: Optional[TerminationReason] = None
    civilians_saved: Occupancy = Field(default_factory=Occupancy)
    civilians_lost: Occupancy = Field(default_factory=Occupancy)
    total_civilians: Occupancy = Field(default_factory=Occupancy)
    civilians_in_transit: list[TransitGroup] = Field(default_factory=list)
    scheduled_events: list[ScheduledEvent] = Field(default_factory=list)
    panic_timers: dict[str, int] = Field(default_factory=dict)
    threat_state: Optional[ThreatState] = None
    action_history: list[ActionRecord] = Field(default_factory=list)
    metrics: EpisodeMetrics = Field(default_factory=EpisodeMetrics)
    resolved_incident_outcomes: IncidentOutcomes = Field(default_factory=IncidentOutcomes)
    room_incident_outcomes: dict[str, IncidentOutcomes] = Field(default_factory=dict)
    prioritized_rooms: set[str] = Field(default_factory=set)
    blocked_routes: set[str] = Field(default_factory=set)
    floor_visibility_state: dict[str, dict[str, Any]] = Field(default_factory=dict)
    belief_store: Any = None
    last_floor_reward_breakdowns: dict[str, dict[str, float]] = Field(default_factory=dict)
    last_prediction_score_by_agent: dict[str, float] = Field(default_factory=dict)
    belief_audit_log: list[dict[str, Any]] = Field(default_factory=list)
    rollout_metadata: dict[str, Any] = Field(default_factory=dict)
