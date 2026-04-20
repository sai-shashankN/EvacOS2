"""EvacOS-MA Multi-Agent Canonical Schema Types.

Frozen for Phase 2+. Contains all observation, action, and structural types
for the multi-agent environment. Do NOT add logic here — types only.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

REWARD_SCHEMA_VERSION: str = "v1"
PROMPT_TEMPLATE_VERSION: str = "v1"
TRACE_SCHEMA_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# Canonical string ID types  (annotated str for Pydantic validation)
# ---------------------------------------------------------------------------

EpisodeId = str
RoundId = int
BuildingId = str
AgentId = str
FloorId = str
RoomId = str
CorridorId = str
StairwellId = str
ExitId = str
CivilianGroupId = str
HazardId = str
DirectiveId = str
BeliefId = str
ActionId = str


# ---------------------------------------------------------------------------
# Base model with flexible extra handling
# ---------------------------------------------------------------------------

class _MABase(BaseModel):
    """Base for MA models. Allows extra fields by default so stubs can evolve."""
    model_config = ConfigDict(extra="forbid")


class _MABaseAllow(BaseModel):
    """Base for MA models that tolerate unknown extras (e.g. info dicts)."""
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    floor_agent = "floor_agent"
    orchestrator = "orchestrator"


class Tier(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
    brutal = "brutal"


class ActionTypeMA(str, Enum):
    """All action types for multi-agent Round 2. Superset of Round 1 ActionType."""
    # Floor-agent actions
    route_within_floor = "route_within_floor"
    prioritize_room = "prioritize_room"
    open_exit = "open_exit"
    lockdown_room = "lockdown_room"
    scout = "scout"
    predict_state = "predict_state"
    handoff_to_orchestrator = "handoff_to_orchestrator"
    # Orchestrator actions
    route_between_floors = "route_between_floors"
    call_elevator = "call_elevator"
    evacuate_floor_priority = "evacuate_floor_priority"
    broadcast_directive = "broadcast_directive"
    override_floor_agent = "override_floor_agent"
    request_explanation = "request_explanation"
    # Both roles
    wait = "wait"


class DirectivePriority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


# ---------------------------------------------------------------------------
# Nested view types  (stubs — population is Phase 4/5)
# ---------------------------------------------------------------------------

class RoomView(_MABase):
    room_id: RoomId
    floor_id: FloorId
    occupancy_mobile: int = 0
    occupancy_injured: int = 0
    occupancy_mobility_impaired: int = 0
    hazard_severity: float = 0.0
    smoke_level: float = 0.0
    accessible: bool = True
    passable: bool = True


class CorridorView(_MABase):
    corridor_id: CorridorId
    from_node_id: str
    to_node_id: str
    hazard_severity: float = 0.0
    passable: bool = True


class StairwellEntryView(_MABase):
    stairwell_id: StairwellId
    connects_floor_ids: list[int] = Field(default_factory=list)
    blocked: bool = False
    capacity_per_step: int = 5


class ExitView(_MABase):
    exit_id: ExitId
    floor_id: int
    exit_type: str = "ground"
    blocked: bool = False
    requires_open_action: bool = False


class CivilianGroupView(_MABase):
    civilian_group_id: CivilianGroupId
    location_room_id: RoomId
    mobility_profile: str = "mobile"
    count: int = 1
    status: str = "waiting"


class HazardView(_MABase):
    hazard_id: HazardId
    hazard_type: str
    severity: float = 0.0
    room_id: Optional[RoomId] = None
    projected_spread: Optional[dict[str, Any]] = None


class FloorSummary(_MABase):
    floor_id: FloorId
    known_civilian_count: int = 0
    unknown_room_count: int = 0
    hazard_severity: float = 0.0
    queue_pressure: float = 0.0
    exit_capacity_remaining: int = 0
    last_updated_round: int = 0


class StairwellAggregateView(_MABase):
    stairwell_id: StairwellId
    floor_ids: list[int] = Field(default_factory=list)
    blocked: bool = False
    current_load: int = 0
    capacity: int = 5


class ElevatorView(_MABase):
    elevator_id: str
    current_floor: int = 0
    target_floor: Optional[int] = None
    operational: bool = True
    capacity: int = 6
    queue_length: int = 0


class ExitQueueView(_MABase):
    exit_id: ExitId
    floor_id: int
    queue_depth: int = 0
    throughput_per_round: float = 0.0
    blocked: bool = False


class BeliefRollup(_MABase):
    total_beliefs: int = 0
    avg_confidence: float = 0.0
    resolved_count: int = 0
    pending_count: int = 0
    recent_highlights: list[dict[str, Any]] = Field(default_factory=list)


class ActionLogEntry(_MABase):
    agent_id: AgentId
    floor_id: Optional[FloorId] = None
    action_type: str
    round_id: int = 0
    summary: str = ""


class DirectiveOutcome(_MABase):
    directive_id: DirectiveId
    target_floor_id: FloorId
    directive_type: str
    accepted: bool = True
    outcome_summary: str = ""


class EscalationRequest(_MABase):
    agent_id: AgentId
    floor_id: FloorId
    category: str = "resource_contention"
    urgency: str = "normal"
    target_ids: list[str] = Field(default_factory=list)
    note: str = ""


# CascadeHintOrNull is a type alias
CascadeHintOrNull = Optional[dict[str, Any]]


# ---------------------------------------------------------------------------
# First-class types
# ---------------------------------------------------------------------------

class Directive(_MABase):
    directive_id: DirectiveId
    supersedes_directive_id_or_null: Optional[DirectiveId] = None
    target: str
    directive_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    priority: DirectivePriority = DirectivePriority.normal
    issued_round: int = 0
    ttl_rounds: int = 10
    human_readable_note: str = ""


class StructuredBelief(_MABase):
    belief_id: BeliefId
    predictor_agent_id: AgentId
    target_entity_ids: list[str] = Field(default_factory=list)
    horizon: int = 5
    prediction_payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    justification: str = ""
    created_round: int = 0
    resolved_round_or_null: Optional[int] = None


# ---------------------------------------------------------------------------
# Observation envelope
# ---------------------------------------------------------------------------

class ObservationEnvelopeMA(_MABase):
    episode_id: EpisodeId
    round_id: int = 0
    role: AgentRole
    agent_id: AgentId
    step: int = 0
    max_steps: int = 350
    seed: int = 42
    tier: Tier = Tier.easy
    disaster_family: str = "fire"
    action_mask: list[str] = Field(default_factory=list)
    last_reward_breakdown: dict[str, float] = Field(default_factory=dict)
    done_last_round: bool = False
    notes: list[str] = Field(default_factory=list)
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    generator_config_hash: str = ""


# ---------------------------------------------------------------------------
# Role-specific observation extensions
# ---------------------------------------------------------------------------

class FloorAgentObservationMA(_MABase):
    # Envelope fields inlined (Pydantic v2 single-inheritance composition)
    episode_id: EpisodeId
    round_id: int = 0
    role: Literal[AgentRole.floor_agent] = AgentRole.floor_agent
    agent_id: AgentId
    step: int = 0
    max_steps: int = 350
    seed: int = 42
    tier: Tier = Tier.easy
    disaster_family: str = "fire"
    action_mask: list[str] = Field(default_factory=list)
    last_reward_breakdown: dict[str, float] = Field(default_factory=dict)
    done_last_round: bool = False
    notes: list[str] = Field(default_factory=list)
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    generator_config_hash: str = ""
    # Floor-agent specific fields
    floor_id: FloorId = "floor_0"
    visible_rooms: list[RoomView] = Field(default_factory=list)
    visible_corridors: list[CorridorView] = Field(default_factory=list)
    stairwell_entries: list[StairwellEntryView] = Field(default_factory=list)
    exits_on_floor: list[ExitView] = Field(default_factory=list)
    visible_civilian_groups: list[CivilianGroupView] = Field(default_factory=list)
    local_hazards: list[HazardView] = Field(default_factory=list)
    sensor_quality: float = 1.0
    visibility_age_by_room: dict[RoomId, int] = Field(default_factory=dict)
    active_directive: Optional[Directive] = None
    override_applied_last_round: bool = False
    override_reason_last_round: Optional[str] = None
    pending_explanation_request: bool = False
    belief_horizon_limit: int = 8
    open_belief_slots: int = 1
    last_prediction_score: float = 0.0


class InterFloorView(_MABase):
    stairwells: list[StairwellAggregateView] = Field(default_factory=list)
    elevator: Optional[ElevatorView] = None
    global_exit_queue: list[ExitQueueView] = Field(default_factory=list)


class OrchestratorObservationMA(_MABase):
    # Envelope fields inlined
    episode_id: EpisodeId
    round_id: int = 0
    role: Literal[AgentRole.orchestrator] = AgentRole.orchestrator
    agent_id: AgentId
    step: int = 0
    max_steps: int = 350
    seed: int = 42
    tier: Tier = Tier.easy
    disaster_family: str = "fire"
    action_mask: list[str] = Field(default_factory=list)
    last_reward_breakdown: dict[str, float] = Field(default_factory=dict)
    done_last_round: bool = False
    notes: list[str] = Field(default_factory=list)
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    generator_config_hash: str = ""
    # Orchestrator-specific fields
    floor_summaries: list[FloorSummary] = Field(default_factory=list)
    inter_floor: InterFloorView = Field(default_factory=InterFloorView)
    belief_rollup: BeliefRollup = Field(default_factory=BeliefRollup)
    recent_floor_actions: list[ActionLogEntry] = Field(default_factory=list)
    recent_directive_outcomes: list[DirectiveOutcome] = Field(default_factory=list)
    cascade_hint: CascadeHintOrNull = None
    unresolved_escalations: list[EscalationRequest] = Field(default_factory=list)


RoleObservationMA = Union[FloorAgentObservationMA, OrchestratorObservationMA]


# ---------------------------------------------------------------------------
# Action argument models (typed per BLUEPRINT action table)
# ---------------------------------------------------------------------------

class RouteWithinFloorArgs(_MABase):
    from_room_id: RoomId
    to_room_id: RoomId
    civilian_group_id_or_null: Optional[CivilianGroupId] = None


class PrioritizeRoomArgs(_MABase):
    room_id: RoomId


class OpenExitArgsMA(_MABase):
    exit_id: ExitId


class LockdownRoomArgs(_MABase):
    room_id: RoomId


class ScoutArgs(_MABase):
    target_room_id: RoomId


class PredictStateArgs(_MABase):
    belief: StructuredBelief


class HandoffToOrchestratorArgs(_MABase):
    category: str = "resource_contention"
    target_ids: list[str] = Field(default_factory=list)
    urgency: str = "normal"
    note: str = ""


class RouteBetweenFloorsArgs(_MABase):
    source_floor_id: FloorId
    target_floor_id: FloorId
    civilian_group_count: int = 0


class CallElevatorArgs(_MABase):
    target_floor_id: FloorId
    direction: str = "up"
    reserved_for: Optional[AgentId] = None


class EvacuateFloorPriorityArgs(_MABase):
    ordered_floor_ids: list[FloorId] = Field(default_factory=list)


class BroadcastDirectiveArgs(_MABase):
    directive: Directive


class OverrideFloorAgentArgs(_MABase):
    target_floor_agent_id: AgentId
    replacement_action_type: str
    replacement_arguments: dict[str, Any] = Field(default_factory=dict)


class RequestExplanationArgs(_MABase):
    target_floor_agent_id: AgentId
    question: str = ""


class WaitArgs(_MABase):
    pass


# Discriminated union of action arguments
ActionArgsMA = Annotated[
    Union[
        RouteWithinFloorArgs,
        PrioritizeRoomArgs,
        OpenExitArgsMA,
        LockdownRoomArgs,
        ScoutArgs,
        PredictStateArgs,
        HandoffToOrchestratorArgs,
        RouteBetweenFloorsArgs,
        CallElevatorArgs,
        EvacuateFloorPriorityArgs,
        BroadcastDirectiveArgs,
        OverrideFloorAgentArgs,
        RequestExplanationArgs,
        WaitArgs,
    ],
    Field(discriminator="action_type") if False else None,  # no single discriminator
]

# Mapping from ActionTypeMA to arg model — used for dispatch in tests/validation
ACTION_TYPE_TO_ARGS: dict[ActionTypeMA, type[BaseModel]] = {
    ActionTypeMA.route_within_floor: RouteWithinFloorArgs,
    ActionTypeMA.prioritize_room: PrioritizeRoomArgs,
    ActionTypeMA.open_exit: OpenExitArgsMA,
    ActionTypeMA.lockdown_room: LockdownRoomArgs,
    ActionTypeMA.scout: ScoutArgs,
    ActionTypeMA.predict_state: PredictStateArgs,
    ActionTypeMA.handoff_to_orchestrator: HandoffToOrchestratorArgs,
    ActionTypeMA.route_between_floors: RouteBetweenFloorsArgs,
    ActionTypeMA.call_elevator: CallElevatorArgs,
    ActionTypeMA.evacuate_floor_priority: EvacuateFloorPriorityArgs,
    ActionTypeMA.broadcast_directive: BroadcastDirectiveArgs,
    ActionTypeMA.override_floor_agent: OverrideFloorAgentArgs,
    ActionTypeMA.request_explanation: RequestExplanationArgs,
    ActionTypeMA.wait: WaitArgs,
}


# ---------------------------------------------------------------------------
# Action envelope
# ---------------------------------------------------------------------------

class ActionEnvelopeMA(_MABaseAllow):
    episode_id: EpisodeId
    round_id: int = 0
    agent_id: AgentId
    action_id: ActionId
    action_type: ActionTypeMA
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    client_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action bundle (per-round submission)
# ---------------------------------------------------------------------------

class ActionBundleMA(_MABase):
    episode_id: EpisodeId
    round_id: int = 0
    orchestrator_action: Optional[ActionEnvelopeMA] = None
    floor_actions: dict[AgentId, ActionEnvelopeMA] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result info (typed-but-opaque stubs)
# ---------------------------------------------------------------------------

class StepResultInfo(_MABaseAllow):
    reservation_trace: list[dict[str, Any]] = Field(default_factory=list)
    arbitration_trace: list[dict[str, Any]] = Field(default_factory=list)
    score_snapshot: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Re-export reward types from rewards submodule (needed for StepResultMA)
# ---------------------------------------------------------------------------

from evacos_ma.schemas.rewards import RewardBreakdown, RoleReward, RewardsByRole  # noqa: E402


# ---------------------------------------------------------------------------
# Step result model
# ---------------------------------------------------------------------------

class ObservationsByRole(_MABase):
    orchestrator: OrchestratorObservationMA
    floors: dict[AgentId, FloorAgentObservationMA] = Field(default_factory=dict)


class StepResultMA(_MABaseAllow):
    observations_by_role: ObservationsByRole
    rewards_by_role: RewardsByRole
    done: bool = False
    done_reason: Optional[str] = None
    invalid_actions: list[dict[str, Any]] = Field(default_factory=list)
    round_events: list[dict[str, Any]] = Field(default_factory=list)
    info: StepResultInfo = Field(default_factory=StepResultInfo)


# ---------------------------------------------------------------------------
# Trace row common fields (for JSONL logs)
# ---------------------------------------------------------------------------

class TraceRowCommon(_MABase):
    episode_id: EpisodeId
    round_id: int = 0
    seed: int = 0
    tier: Tier = Tier.easy
    disaster_family: str = "fire"
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    generator_config_hash: str = ""
    reward_schema_version: str = REWARD_SCHEMA_VERSION
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    model_name: str = ""
    checkpoint_tag: str = ""


class EpisodeSummaryRow(TraceRowCommon):
    total_reward: float = 0.0
    termination_reason: Optional[str] = None
    civilians_saved: int = 0
    civilians_lost: int = 0
    total_steps: int = 0


class RoundTraceRow(TraceRowCommon):
    round_events: list[dict[str, Any]] = Field(default_factory=list)
    orchestrator_action_type: Optional[str] = None
    floor_action_types: dict[str, str] = Field(default_factory=dict)


class ActionTraceRow(TraceRowCommon):
    agent_id: AgentId = ""
    action_id: ActionId = ""
    action_type: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    valid: bool = True
    rejection_reason: Optional[str] = None


class RewardTraceRow(TraceRowCommon):
    agent_id: AgentId = ""
    raw_reward: float = 0.0
    normalized_reward: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class BeliefAuditRow(TraceRowCommon):
    belief_id: BeliefId = ""
    predictor_agent_id: AgentId = ""
    confidence: float = 0.0
    resolved: bool = False
    score: Optional[float] = None


class RationaleAuditRow(TraceRowCommon):
    agent_id: AgentId = ""
    action_id: ActionId = ""
    eligible_tokens: int = 0
    bonus_awarded: float = 0.0
    gates_passed: list[str] = Field(default_factory=list)
    counterfactual_delta: float = 0.0
    reason_hash: str = ""


class CurriculumEventRow(TraceRowCommon):
    event_type: str = ""
    from_tier: Optional[str] = None
    to_tier: Optional[str] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Trajectory sample (training contract)
# ---------------------------------------------------------------------------

class TrajectorySample(_MABase):
    episode_id: EpisodeId
    round_id: int = 0
    agent_id: AgentId = ""
    role: AgentRole = AgentRole.floor_agent
    prompt: list[dict[str, str]] = Field(default_factory=list)
    completion_text: str = ""
    parsed_action: dict[str, Any] = Field(default_factory=dict)
    raw_reward: float = 0.0
    normalized_reward: float = 0.0
    done: bool = False
    checkpoint_tag: str = ""
    group_id: str = ""
