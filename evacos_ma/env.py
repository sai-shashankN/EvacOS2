from __future__ import annotations

import hashlib
import math
import random
import re
import time
import uuid
from collections import deque
from typing import Any, Optional

from pydantic import BaseModel

from evacos_ma.beliefs import BeliefRegistration, BeliefStore
from evacos_ma.building_gen import generate_building
from evacos_ma.cascade import CascadeScheduler
from evacos_ma.cascade_configs import get_cascade_config
from evacos_ma.disaster import DisasterEngine, create_disaster_engine
from evacos_ma.models import (
    Action,
    ActionRecord,
    ActionType,
    Building,
    CallElevatorAction,
    DisasterType,
    ElevatorObservation,
    EpisodeStateInternal,
    EvacuateFloorAction,
    Exit,
    HazardType,
    IncidentOutcomes,
    LockdownRoomAction,
    MetricsDelta,
    Observation,
    Occupancy,
    OpenExitAction,
    PrioritizeRoomAction,
    RenderObservation,
    RequestRenderAction,
    Reward,
    Room,
    RoomObservation,
    RouteCiviliansAction,
    ScheduledEvent,
    StateView,
    StepInfo,
    SummaryObservation,
    TaskSpec,
    TerminationReason,
    TransitGroup,
    TransitGroupObservation,
    WaitAction,
)
from evacos_ma.observability import FloorVisibilityState, VisibilityConfig, build_floor_observation
from evacos_ma.reward_pipeline import RewardPipeline
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionLogEntry,
    ActionTypeMA,
    AgentRole,
    BeliefAuditRow,
    BeliefRollup,
    CivilianGroupView,
    CorridorView,
    DirectiveOutcome,
    ElevatorView,
    EscalationRequest,
    ExitQueueView,
    ExitView,
    FloorAgentObservationMA,
    FloorSummary,
    HazardView,
    InterFloorView,
    OrchestratorObservationMA,
    ObservationsByRole,
    PredictStateArgs,
    RewardBreakdown,
    RoleReward,
    RewardsByRole,
    ScoutArgs,
    StairwellAggregateView,
    StairwellEntryView,
    StepResultInfo,
    StepResultMA,
    StructuredBelief,
)
from evacos_ma.directives import DirectiveStore
from evacos_ma.permissions import action_mask_for_role
from evacos_ma.round_protocol import RoundProtocol
from evacos_ma.task_registry import get_task

_DEFAULT_ROLLOUT_REWARD_CONFIG: dict[str, float | str] = {
    "rationale_scaling": "linear_capped",
    "alpha": 0.01,
    "beta": 0.25,
    "cap": 1.0,
    "eligible_token_ceiling": 160,
    "clip_normalized_to": 1.0,
}
_RATIONALE_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _clone_occupancy(occupancy: Occupancy) -> Occupancy:
    return Occupancy(
        mobile=occupancy.mobile,
        injured=occupancy.injured,
        mobility_impaired=occupancy.mobility_impaired,
    )


def _empty_occupancy() -> Occupancy:
    return Occupancy()


def _add_occupancy(target: Occupancy, delta: Occupancy) -> None:
    target.mobile += delta.mobile
    target.injured += delta.injured
    target.mobility_impaired += delta.mobility_impaired


def _subtract_occupancy(target: Occupancy, delta: Occupancy) -> None:
    target.mobile -= delta.mobile
    target.injured -= delta.injured
    target.mobility_impaired -= delta.mobility_impaired


def _occupancy_leq(left: Occupancy, right: Occupancy) -> bool:
    return (
        left.mobile <= right.mobile
        and left.injured <= right.injured
        and left.mobility_impaired <= right.mobility_impaired
    )


def _normalize_rationale_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in _RATIONALE_TOKEN_RE.finditer(text)]


def _has_repeated_ngram(tokens: list[str], *, size: int, max_occurrences: int) -> bool:
    if len(tokens) < size:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(len(tokens) - size + 1):
        ngram = tuple(tokens[idx : idx + size])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] > max_occurrences:
            return True
    return False


def _occupancy_delta(before: Occupancy, after: Occupancy) -> Occupancy:
    return Occupancy(
        mobile=max(0, before.mobile - after.mobile),
        injured=max(0, before.injured - after.injured),
        mobility_impaired=max(0, before.mobility_impaired - after.mobility_impaired),
    )


def _remove_occupancy(target: Occupancy, count: int) -> Occupancy:
    removed = Occupancy()
    remaining = max(0, count)
    for field in ("injured", "mobility_impaired", "mobile"):
        available = getattr(target, field)
        taken = min(available, remaining)
        setattr(target, field, available - taken)
        setattr(removed, field, taken)
        remaining -= taken
        if remaining == 0:
            break
    return removed


def _clone_incident_outcomes(outcomes: IncidentOutcomes) -> IncidentOutcomes:
    return IncidentOutcomes(
        safe=outcomes.safe,
        mild_injury=outcomes.mild_injury,
        severe_injury=outcomes.severe_injury,
        deaths=outcomes.deaths,
    )


def _add_incident_outcomes(target: IncidentOutcomes, delta: IncidentOutcomes) -> None:
    target.safe += delta.safe
    target.mild_injury += delta.mild_injury
    target.severe_injury += delta.severe_injury
    target.deaths += delta.deaths


def _remove_incident_outcomes(target: IncidentOutcomes, count: int) -> IncidentOutcomes:
    removed = IncidentOutcomes()
    remaining = max(0, count)
    for field in ("severe_injury", "mild_injury", "safe"):
        available = getattr(target, field)
        taken = min(available, remaining)
        setattr(target, field, available - taken)
        setattr(removed, field, taken)
        remaining -= taken
        if remaining == 0:
            break
    return removed


def _floor_number(node_id: str) -> Optional[int]:
    if not node_id.startswith("F") or "_R" not in node_id:
        return None
    floor_fragment = node_id.split("_", maxsplit=1)[0]
    try:
        return int(floor_fragment[1:])
    except ValueError:
        return None


class EvacEnvironment:
    """Core evacuation environment managing episode state."""

    def __init__(self, reward_pipeline: RewardPipeline | None = None):
        self._episodes: dict[str, EpisodeStateInternal] = {}
        self._engines: dict[str, DisasterEngine] = {}
        self._disaster_origins: dict[str, str] = {}
        self._pending_exit_opens: dict[str, set[str]] = {}
        self._room_priority_modes: dict[str, dict[str, str]] = {}
        self._requested_render_floors: dict[str, Optional[int]] = {}
        self._elevator_progress: dict[str, dict[str, int]] = {}
        self._rngs: dict[str, random.Random] = {}
        self._cascade_schedulers: dict[str, CascadeScheduler] = {}
        self.reward_pipeline = reward_pipeline or RewardPipeline()
        self._visibility_config = VisibilityConfig()
        # Phase 5: per-episode directive, handoff, override, and round protocol stores
        self._directive_stores: dict[str, DirectiveStore] = {}
        self._handoff_stores: dict[str, list[dict[str, Any]]] = {}
        self._override_last_round: dict[str, dict[str, str]] = {}  # ep_id -> {agent_id: reason}
        self._ma_recent_floor_actions: dict[str, list[dict[str, Any]]] = {}
        self._round_protocol = RoundProtocol()

    def reset(
        self,
        task_id: str,
        seed: Optional[int] = None,
        procgen_tier: str | None = None,
        procgen_disaster_family: DisasterType | None = None,
        procgen_max_steps: int | None = None,
    ) -> tuple[str, Observation]:
        """Initialize a new episode. Returns (episode_id, initial_observation)."""
        if (
            procgen_tier is not None
            or procgen_disaster_family is not None
            or procgen_max_steps is not None
        ):
            if procgen_tier is None or procgen_disaster_family is None:
                raise ValueError(
                    "procgen_tier and procgen_disaster_family are required when using procgen reset options"
                )
            episode_id, _ = self._reset_multi_agent_procgen(
                task_id=task_id,
                seed=seed if seed is not None else random.randint(0, 2_147_483_647),
                procgen_tier=procgen_tier,
                procgen_disaster_family=procgen_disaster_family,
                procgen_max_steps=procgen_max_steps,
            )
            ep = self.get_internal_state(episode_id)
            return episode_id, self._build_observation(ep)

        task = get_task(task_id).model_copy(deep=True)
        resolved_seed = seed if seed is not None else random.randint(0, 2_147_483_647)
        building = generate_building(task_id, resolved_seed)
        origin_room_id = self._select_disaster_origin(task.disaster_type, building)
        engine = create_disaster_engine(
            task.disaster_type,
            building,
            origin_room_id,
            resolved_seed,
            disaster_zones=list(building.disaster_zones),
            scheduled_events=self._initial_scheduled_events(
                task.disaster_type, task_id=task_id, building=building, seed=resolved_seed,
            ),
        )

        episode_id = uuid.uuid4().hex
        ep = EpisodeStateInternal(
            episode_id=episode_id,
            task=task,
            building=building,
            seed=resolved_seed,
            total_civilians=self._count_total_civilians(building),
        )
        ep.floor_visibility_state = {
            self._floor_agent_key(floor.floor_id): FloorVisibilityState().model_dump(mode="json")
            for floor in building.floors
        }
        ep.last_floor_reward_breakdowns = {
            self._floor_agent_key(floor.floor_id): {}
            for floor in building.floors
        }
        ep.last_prediction_score_by_agent = {
            self._floor_agent_key(floor.floor_id): 0.0
            for floor in building.floors
        }
        ep.belief_store = BeliefStore(
            episode_id=episode_id,
            seed=resolved_seed,
            tier=self._ma_tier_value(task.difficulty),
            disaster_family=task.disaster_type.value,
            generator_config_hash=self._generator_config_hash(task.task_id, resolved_seed),
        )
        self._sync_engine_state(ep, engine)
        if task.disaster_type == DisasterType.multi_cascade:
            ep.panic_timers = {
                room.room_id: 0
                for room in self._iter_rooms(building)
                if room.occupancy.total > 0
            }
        self._sync_room_incident_outcomes(ep)

        self._episodes[episode_id] = ep
        self._engines[episode_id] = engine
        self._disaster_origins[episode_id] = origin_room_id
        self._pending_exit_opens[episode_id] = set()
        self._room_priority_modes[episode_id] = {}
        self._requested_render_floors[episode_id] = None
        self._elevator_progress[episode_id] = {}
        self._rngs[episode_id] = random.Random(resolved_seed)

        # Set up cascade scheduler for long-horizon tasks
        cascade_stages = get_cascade_config(task_id)
        if cascade_stages:
            self._cascade_schedulers[episode_id] = CascadeScheduler(
                stages=cascade_stages, seed=resolved_seed,
            )
        else:
            self._cascade_schedulers[episode_id] = CascadeScheduler(
                stages=[], seed=resolved_seed,
            )
        return episode_id, self._build_observation(ep)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, StepInfo]:
        """Process one agent action. Returns (observation, reward, done, info)."""
        if action.episode_id not in self._episodes:
            raise ValueError(f"Unknown episode_id: {action.episode_id}")

        ep = self._episodes[action.episode_id]
        if ep.done:
            raise ValueError(f"Episode {action.episode_id} is already complete")

        step_started = time.perf_counter()
        action_valid = True
        invalid_reason: Optional[str] = None
        prev_saved = _clone_occupancy(ep.civilians_saved)
        prev_lost = _clone_occupancy(ep.civilians_lost)
        next_step = ep.step + 1
        self._sync_room_incident_outcomes(ep)

        if action.expected_step != ep.step:
            action_valid = False
            invalid_reason = f"Expected step {ep.step}, received {action.expected_step}"
        else:
            action_valid, invalid_reason, _ = self._process_action(ep, action)

        pre_disaster_snapshot = self._snapshot_room_occupancy(ep.building)
        triggered_events = self._engines[ep.episode_id].advance(next_step)
        self._sync_engine_state(ep, self._engines[ep.episode_id])

        self._capture_disaster_losses(ep, pre_disaster_snapshot)
        self._advance_elevators(ep)
        self._resolve_transit(ep)
        self._apply_incident_exposure(ep)
        self._check_panic(ep)
        self._resolve_hazard_casualties(ep)
        self._apply_pending_exit_opens(ep)

        ep.step = next_step
        self._update_termination(ep)
        reward = self._compute_reward(ep, prev_saved, prev_lost, action, action_valid)

        # Compute normalized reward using the canonical episode tier.
        tier = self._episode_reward_tier(ep)
        normalized_reward = self.reward_pipeline.normalize(reward.total, tier)
        # Observe AFTER normalization (avoid self-normalization)
        self.reward_pipeline.observe(reward.total, tier)

        step_elapsed_ms = (time.perf_counter() - step_started) * 1000
        hazard_exposure = self._current_hazard_exposure(ep)
        metrics_delta = self._update_metrics(
            ep, action, action_valid, prev_saved, prev_lost, hazard_exposure, step_elapsed_ms,
        )

        ep.action_history.append(
            ActionRecord(
                step=ep.step,
                action_type=action.action_type,
                raw_action=action.model_dump(mode="json"),
                valid=action_valid,
                reward_total=reward.total,
            )
        )

        info = StepInfo(
            termination_reason=ep.termination_reason,
            invalid_action=not action_valid,
            invalid_reason=invalid_reason,
            triggered_events=triggered_events,
            metrics_delta=metrics_delta,
            normalized_reward=normalized_reward,
        )
        observation = self._build_observation(
            ep,
            render_floor=action.floor_id if isinstance(action, RequestRenderAction) else None,
        )
        return observation, reward, ep.done, info

    def state(self, episode_id: str) -> StateView:
        """Return public state view for an episode."""
        ep = self.get_internal_state(episode_id)
        observation = self._build_observation(ep)
        return StateView(
            episode_id=ep.episode_id,
            task_id=ep.task.task_id,
            step=ep.step,
            max_steps=ep.task.max_steps,
            done=ep.done,
            termination_reason=ep.termination_reason,
            summary=observation.summary,
            rooms=observation.rooms,
            corridors=observation.corridors,
            stairwells=observation.stairwells,
            exits=observation.exits,
            elevators=self._build_elevator_observations(ep),
            transit_groups=self._build_transit_group_observations(ep),
            blocked_route_ids=sorted(ep.blocked_routes),
        )

    def get_internal_state(self, episode_id: str) -> EpisodeStateInternal:
        """Return full internal state (for graders only)."""
        if episode_id not in self._episodes:
            raise ValueError(f"Unknown episode_id: {episode_id}")
        return self._episodes[episode_id]

    def cleanup_episode(self, episode_id: str) -> None:
        """Remove a completed episode from memory."""
        self._episodes.pop(episode_id, None)
        self._engines.pop(episode_id, None)
        self._disaster_origins.pop(episode_id, None)
        self._pending_exit_opens.pop(episode_id, None)
        self._room_priority_modes.pop(episode_id, None)
        self._requested_render_floors.pop(episode_id, None)
        self._elevator_progress.pop(episode_id, None)
        self._rngs.pop(episode_id, None)
        self._cascade_schedulers.pop(episode_id, None)
        self._directive_stores.pop(episode_id, None)
        self._handoff_stores.pop(episode_id, None)
        self._override_last_round.pop(episode_id, None)
        self._ma_recent_floor_actions.pop(episode_id, None)

    def reset_multi_agent(
        self,
        task_id: str,
        seed: Optional[int] = None,
        procgen_tier: Optional[str] = None,
        procgen_disaster_family: Optional[DisasterType] = None,
        procgen_max_steps: Optional[int] = None,
    ) -> tuple[str, ObservationsByRole]:
        """Initialize a multi-agent episode.

        If both procgen_tier and procgen_disaster_family are provided, uses
        the procedural generator instead of task_registry. Otherwise, falls
        back to legacy task_registry-based reset (backward compatible).
        """
        if procgen_tier is not None and procgen_disaster_family is not None:
            return self._reset_multi_agent_procgen(
                task_id=task_id,
                seed=seed if seed is not None else random.randint(0, 2_147_483_647),
                procgen_tier=procgen_tier,
                procgen_disaster_family=procgen_disaster_family,
                procgen_max_steps=procgen_max_steps,
            )
        episode_id, _ = self.reset(task_id, seed)
        ep = self.get_internal_state(episode_id)
        # Initialize Phase 5 stores
        self._directive_stores[episode_id] = DirectiveStore()
        self._handoff_stores[episode_id] = []
        self._override_last_round[episode_id] = {}
        self._ma_recent_floor_actions[episode_id] = []
        return episode_id, self._build_observations_by_role(ep)

    def _reset_multi_agent_procgen(
        self,
        task_id: str,
        seed: int,
        procgen_tier: str,
        procgen_disaster_family: DisasterType,
        procgen_max_steps: Optional[int] = None,
    ) -> tuple[str, ObservationsByRole]:
        """Reset using the procedural generator."""
        from procgen.generator import generate_instance as pg_generate
        from procgen.validator import regenerate_until_valid

        result = regenerate_until_valid(
            seed=seed,
            tier=procgen_tier,
            disaster_family=procgen_disaster_family.value,
            max_attempts=20,
        )
        if result is None:
            raise RuntimeError(
                f"Procgen could not produce a valid instance for "
                f"seed={seed} tier={procgen_tier} family={procgen_disaster_family.value}"
            )
        instance, _report = result

        building = instance.building
        task = TaskSpec(
            task_id=task_id,
            name=f"procgen_{procgen_disaster_family.value}_{procgen_tier}",
            difficulty=procgen_tier,
            disaster_type=procgen_disaster_family,
            building_profile="procgen",
            success_criteria="Evacuate civilians before routes are cut off.",
            goal=f"Evacuate all civilians from the procedurally generated {procgen_disaster_family.value} scenario.",
            max_steps=80 if procgen_max_steps is None else int(procgen_max_steps),
            description=(
                "Procedurally generated multi-agent evacuation scenario. "
                "Tier/disaster metadata are sourced from procgen state, not task_registry."
            ),
        )

        origin_room_id = self._select_disaster_origin(procgen_disaster_family, building)
        engine = create_disaster_engine(
            procgen_disaster_family,
            building,
            origin_room_id,
            seed,
            disaster_zones=list(building.disaster_zones),
            scheduled_events=instance.scheduled_events,
        )

        episode_id = uuid.uuid4().hex
        ep = EpisodeStateInternal(
            episode_id=episode_id,
            task=task,
            building=building,
            seed=seed,
            total_civilians=self._count_total_civilians(building),
        )
        ep.floor_visibility_state = {
            self._floor_agent_key(floor.floor_id): FloorVisibilityState().model_dump(mode="json")
            for floor in building.floors
        }
        ep.last_floor_reward_breakdowns = {
            self._floor_agent_key(floor.floor_id): {}
            for floor in building.floors
        }
        ep.last_prediction_score_by_agent = {
            self._floor_agent_key(floor.floor_id): 0.0
            for floor in building.floors
        }
        ep.belief_store = BeliefStore(
            episode_id=episode_id,
            seed=seed,
            tier=procgen_tier,
            disaster_family=procgen_disaster_family.value,
            generator_config_hash=instance.generator_config_hash,
        )
        self._sync_engine_state(ep, engine)
        self._sync_room_incident_outcomes(ep)
        if procgen_disaster_family == DisasterType.multi_cascade:
            ep.panic_timers = {
                room.room_id: 0
                for room in self._iter_rooms(building)
                if room.occupancy.total > 0
            }

        self._episodes[episode_id] = ep
        self._engines[episode_id] = engine
        self._disaster_origins[episode_id] = origin_room_id
        self._pending_exit_opens[episode_id] = set()
        self._room_priority_modes[episode_id] = {}
        self._requested_render_floors[episode_id] = None
        self._elevator_progress[episode_id] = {}
        self._rngs[episode_id] = random.Random(seed)
        self._cascade_schedulers[episode_id] = CascadeScheduler(stages=[], seed=seed)

        # Initialize Phase 5 stores
        self._directive_stores[episode_id] = DirectiveStore()
        self._handoff_stores[episode_id] = []
        self._override_last_round[episode_id] = {}
        self._ma_recent_floor_actions[episode_id] = []

        return episode_id, self._build_observations_by_role(ep)

    def build_floor_agent_observation(
        self,
        episode_id: str,
        floor_id: int | str,
        agent_id: Optional[str] = None,
    ) -> FloorAgentObservationMA:
        ep = self.get_internal_state(episode_id)
        floor_num = self._floor_number_from_public_id(floor_id)
        return self._build_floor_agent_observation(ep, floor_num, agent_id=agent_id)

    def build_orchestrator_observation(
        self,
        episode_id: str,
    ) -> OrchestratorObservationMA:
        ep = self.get_internal_state(episode_id)
        return self._build_orchestrator_observation(ep)

    def step_floor_agent_action(
        self,
        episode_id: str,
        agent_id: str,
        action_type: ActionTypeMA,
        arguments: dict[str, Any] | BaseModel | None = None,
    ) -> tuple[FloorAgentObservationMA, RoleReward, bool, dict[str, Any]]:
        bundle = ActionBundleMA(
            episode_id=episode_id,
            round_id=self.get_internal_state(episode_id).step,
            floor_actions={
                agent_id: ActionEnvelopeMA(
                    episode_id=episode_id,
                    round_id=self.get_internal_state(episode_id).step,
                    agent_id=agent_id,
                    action_id=uuid.uuid4().hex,
                    action_type=action_type,
                    arguments=self._model_to_dict(arguments),
                )
            },
        )
        step_result = self.step_multi_agent(bundle)
        return (
            step_result.observations_by_role.floors[agent_id],
            step_result.rewards_by_role.floors[agent_id],
            step_result.done,
            {
                "invalid_actions": step_result.invalid_actions,
                "round_events": step_result.round_events,
                "score_snapshot": step_result.info.score_snapshot,
            },
        )

    def step_multi_agent(self, bundle: ActionBundleMA) -> StepResultMA:
        ep = self.get_internal_state(bundle.episode_id)
        progress_start = ep.civilians_saved.total - ep.civilians_lost.total
        floor_components = {
            self._floor_agent_key(floor.floor_id): {}
            for floor in ep.building.floors
        }
        invalid_actions: list[dict[str, Any]] = []
        registrations: dict[str, BeliefRegistration] = {}

        # Ensure stores exist (in case step_multi_agent called without reset_multi_agent)
        if bundle.episode_id not in self._directive_stores:
            self._directive_stores[bundle.episode_id] = DirectiveStore()
        if bundle.episode_id not in self._handoff_stores:
            self._handoff_stores[bundle.episode_id] = []
        if bundle.episode_id not in self._override_last_round:
            self._override_last_round[bundle.episode_id] = {}
        if bundle.episode_id not in self._ma_recent_floor_actions:
            self._ma_recent_floor_actions[bundle.episode_id] = []

        # Pre-round Phase 4 helpers: scout, predict
        for agent_id, action in sorted(bundle.floor_actions.items()):
            parsed_type = ActionTypeMA(action.action_type)
            if parsed_type == ActionTypeMA.scout:
                try:
                    parsed = ScoutArgs.model_validate(action.arguments)
                except Exception as exc:
                    invalid_actions.append(
                        {
                            "agent_id": agent_id,
                            "action_id": action.action_id,
                            "reason": f"invalid_scout_args: {exc}",
                        }
                    )
                else:
                    self._mark_room_scouted(ep, agent_id, parsed.target_room_id)
                    floor_components[agent_id]["floor_scout_cost"] = self._scout_reward_component(ep)
            elif parsed_type == ActionTypeMA.predict_state:
                try:
                    parsed = PredictStateArgs.model_validate(action.arguments)
                except Exception as exc:
                    invalid_actions.append(
                        {
                            "agent_id": agent_id,
                            "action_id": action.action_id,
                            "reason": f"invalid_predict_state_args: {exc}",
                        }
                    )
                else:
                    registration = self._register_belief(ep, agent_id, parsed.belief)
                    registrations[agent_id] = registration
                    if not registration.accepted:
                        floor_components[agent_id]["duplicate_belief_penalty"] = self._duplicate_penalty_component(ep)
                        invalid_actions.append(
                            {
                                "agent_id": agent_id,
                                "action_id": action.action_id,
                                "reason": registration.reason,
                            }
                        )

        # --- Phase 5: delegate to RoundProtocol ---
        round_result = self._round_protocol.run_round(
            env=self,
            ep=ep,
            orchestrator_action=bundle.orchestrator_action,
            floor_actions=dict(bundle.floor_actions),
            round_id=ep.step,
            directive_store=self._directive_stores[bundle.episode_id],
            handoff_store=self._handoff_stores[bundle.episode_id],
        )

        # Advance the underlying simulation step (disaster engine, transit, etc.)
        base_obs, base_reward, done, base_info = self.step(
            WaitAction(
                episode_id=bundle.episode_id,
                expected_step=ep.step,
                action_type=ActionType.wait,
            )
        )
        del base_obs
        base_reward_scalar = float(
            getattr(base_reward, "raw", getattr(base_reward, "total", base_reward))
        )
        progress_end = ep.civilians_saved.total - ep.civilians_lost.total
        team_progress_delta = float(progress_end - progress_start)

        audit_rows = self._resolve_beliefs(ep)
        for row in audit_rows:
            predictor = row.predictor_agent_id
            floor_components.setdefault(predictor, {})
            floor_components[predictor]["floor_prediction"] = floor_components[predictor].get("floor_prediction", 0.0) + self._prediction_reward_component(ep, row.score or 0.0)
        reward_config = self._rollout_reward_config(ep)

        # Merge round protocol rejections with Phase 4 invalid actions
        all_invalid = invalid_actions + round_result.rejected_actions

        # Update override tracking
        self._override_last_round[bundle.episode_id] = round_result.override_applied
        self._record_ma_floor_actions(
            bundle.episode_id,
            {
                action.agent_id: action
                for action in round_result.accepted_actions
                if action.agent_id.startswith("floor_")
            },
            ep.step,
        )

        # Compute orchestrator reward (oversight bonus)
        weights = ep.task.reward_weights
        orchestrator_components: dict[str, float] = {}
        for agent_id, delta in round_result.counterfactual_deltas.items():
            if delta > 0:
                orchestrator_components["oversight_bonus"] = orchestrator_components.get("oversight_bonus", 0.0) + delta
        for agent_id, delta in round_result.apply_deltas.items():
            if not agent_id.startswith("floor_"):
                continue
            fc = floor_components.setdefault(agent_id, {})
            fc["floor_saved"] = fc.get("floor_saved", 0.0) + float(delta["saved"]) * weights.floor_saved
            fc["floor_lost"] = fc.get("floor_lost", 0.0) + float(delta["lost"]) * weights.floor_lost
        n_floors = max(1, len(ep.building.floors))
        orchestrator_components["base_sim_reward"] = orchestrator_components.get(
            "base_sim_reward", 0.0
        ) + base_reward_scalar
        orchestrator_components["team_progress_dense"] = orchestrator_components.get(
            "team_progress_dense", 0.0
        ) + team_progress_delta * weights.team_progress_dense_orchestrator
        base_floor_share = base_reward_scalar / n_floors
        for floor in ep.building.floors:
            agent_id = f"floor_{floor.floor_id}_agent"
            fc = floor_components.setdefault(agent_id, {})
            fc["base_sim_reward_share"] = fc.get("base_sim_reward_share", 0.0) + base_floor_share
            fc["team_progress_dense"] = fc.get("team_progress_dense", 0.0) + team_progress_delta * weights.team_progress_dense_floor

        invalid_agents: dict[str, int] = {}
        for row in invalid_actions:
            aid = row.get("agent_id", "")
            if aid.startswith("floor_"):
                invalid_agents[aid] = invalid_agents.get(aid, 0) + 1
        for rej in round_result.rejected_actions:
            aid = rej.get("agent_id", "") if isinstance(rej, dict) else getattr(rej, "agent_id", "")
            if aid.startswith("floor_"):
                invalid_agents[aid] = invalid_agents.get(aid, 0) + 1
        for aid, count in invalid_agents.items():
            fc = floor_components.setdefault(aid, {})
            fc["floor_invalid_action"] = fc.get("floor_invalid_action", 0.0) + count * weights.floor_invalid_action

        capacity_contention = 0
        for rej in round_result.rejected_actions:
            reason = rej.get("reason", "") if isinstance(rej, dict) else getattr(rej, "reason", "")
            if reason in ("stairwell_capacity", "elevator_capacity"):
                capacity_contention += 1
        orchestrator_components["coordination_bonus"] = orchestrator_components.get(
            "coordination_bonus", 0.0
        ) + float(capacity_contention) * weights.coordination_bonus

        addressed = round_result.directive_addressed_count
        issued = round_result.directive_issued_count
        churn = max(0, issued - addressed)
        orchestrator_components["directive_quality"] = orchestrator_components.get(
            "directive_quality", 0.0
        ) + (addressed - churn) * weights.directive_quality

        if done:
            orchestrator_components["total_saved_terminal"] = (
                orchestrator_components.get("total_saved_terminal", 0.0)
                + float(ep.civilians_saved.total) * weights.total_saved_terminal
            )
            orchestrator_components["total_lost_terminal"] = (
                orchestrator_components.get("total_lost_terminal", 0.0)
                + float(ep.civilians_lost.total) * weights.total_lost_terminal
            )

        orchestrator_bonus = self._orchestrator_rationale_bonus(
            action=bundle.orchestrator_action,
            counterfactual_deltas=round_result.counterfactual_deltas,
            reward_config=reward_config,
        )
        if orchestrator_bonus != 0.0:
            orchestrator_components["rationale_bonus"] = (
                orchestrator_components.get("rationale_bonus", 0.0) + orchestrator_bonus
            )

        belief_scores_by_agent: dict[str, float] = {}
        for row in audit_rows:
            if not row.resolved or row.score is None:
                continue
            belief_scores_by_agent[row.predictor_agent_id] = max(
                belief_scores_by_agent.get(row.predictor_agent_id, float("-inf")),
                float(row.score),
            )
        for agent_id, action in sorted(bundle.floor_actions.items()):
            floor_bonus = self._floor_rationale_bonus(
                action=action,
                belief_score=belief_scores_by_agent.get(agent_id, 0.0),
                reward_config=reward_config,
            )
            if floor_bonus != 0.0:
                fc = floor_components.setdefault(agent_id, {})
                fc["rationale_bonus"] = fc.get("rationale_bonus", 0.0) + floor_bonus

        # Build observations (with populated action_mask, directive, override fields)
        observations = self._build_observations_by_role(ep)
        rewards = RewardsByRole(
            orchestrator=self._build_role_reward(ep, orchestrator_components),
            floors={
                agent_id: self._build_role_reward(ep, components)
                for agent_id, components in sorted(floor_components.items())
            },
        )
        for agent_id, role_reward in rewards.floors.items():
            ep.last_floor_reward_breakdowns[agent_id] = role_reward.breakdown.get_components()

        self._clear_scout_marks(ep)
        return StepResultMA(
            observations_by_role=observations,
            rewards_by_role=rewards,
            done=done,
            done_reason=base_info.termination_reason.value if base_info.termination_reason is not None else None,
            invalid_actions=all_invalid,
            round_events=[
                event.model_dump(mode="json")
                for event in base_info.triggered_events
            ] + round_result.round_events,
            info=StepResultInfo(
                reservation_trace=round_result.reservation_trace,
                arbitration_trace=round_result.arbitration_trace,
                score_snapshot={
                    "registrations": {
                        agent_id: registration.model_dump(mode="json")
                        for agent_id, registration in registrations.items()
                    },
                    "belief_audits": [row.model_dump(mode="json") for row in audit_rows],
                    "counterfactual_deltas": round_result.counterfactual_deltas,
                },
            ),
        )

    def _initial_scheduled_events(
        self,
        disaster_type: DisasterType,
        task_id: str = "",
        building: Building | None = None,
        seed: int = 0,
    ) -> list[ScheduledEvent] | None:
        if disaster_type != DisasterType.multi_cascade:
            return None
        # For long-horizon tasks, use cascade config to build scheduled events
        cascade_stages = get_cascade_config(task_id)
        if cascade_stages and building is not None:
            from evacos_ma.cascade import build_scheduled_events_from_stages
            return build_scheduled_events_from_stages(cascade_stages, building, seed)
        return []

    def _select_disaster_origin(self, disaster_type: DisasterType, building: Building) -> str:
        if building.disaster_zones:
            return sorted(building.disaster_zones)[0]

        floor_map = {floor.floor_id: floor for floor in building.floors}

        if disaster_type == DisasterType.fire:
            floor = floor_map.get(2, building.floors[-1])
            marked = [
                room.room_id
                for room in floor.rooms
                if room.hazard.hazard_type is not None and room.hazard.severity > 0
            ]
            return sorted(marked or [room.room_id for room in floor.rooms])[0]

        if disaster_type == DisasterType.flood:
            return "F0_R0"

        if disaster_type in {DisasterType.gas, DisasterType.structural}:
            floor = floor_map.get(0, building.floors[0])
            marked = [
                room.room_id
                for room in floor.rooms
                if room.hazard.hazard_type is not None and room.hazard.severity > 0
            ]
            return sorted(marked or [room.room_id for room in floor.rooms])[0]

        if disaster_type == DisasterType.active_threat:
            floor = floor_map.get(3, building.floors[-1])
            return max(
                floor.rooms,
                key=lambda room: (room.occupancy.total, room.room_id),
            ).room_id

        floor = floor_map.get(1, building.floors[0])
        marked = [
            room.room_id
            for room in floor.rooms
            if room.hazard.hazard_type is not None and room.hazard.severity > 0
        ]
        return sorted(marked or [room.room_id for room in floor.rooms])[0]

    def _process_action(
        self,
        ep: EpisodeStateInternal,
        action: Action,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        if isinstance(action, RouteCiviliansAction):
            return self._handle_route_civilians(ep, action)
        if isinstance(action, EvacuateFloorAction):
            return self._handle_evacuate_floor(ep, action)
        if isinstance(action, PrioritizeRoomAction):
            return self._handle_prioritize_room(ep, action)
        if action.action_type == ActionType.block_route:
            return self._handle_block_route(ep, str(action.edge_id))
        if isinstance(action, CallElevatorAction):
            return self._handle_call_elevator(ep, action)
        if isinstance(action, OpenExitAction):
            return self._handle_open_exit(ep, action)
        if isinstance(action, LockdownRoomAction):
            return self._handle_lockdown_room(ep, action)
        if isinstance(action, RequestRenderAction):
            return self._handle_request_render(ep, action)
        return True, None, {"routed_rooms": set()}

    def _handle_route_civilians(
        self,
        ep: EpisodeStateInternal,
        action: RouteCiviliansAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        source = self._room_lookup(ep.building).get(action.from_node_id)
        if source is None:
            return False, f"Unknown source room: {action.from_node_id}", {"routed_rooms": set()}
        if not source.accessible or not source.hazard.passable:
            return False, f"Source room {source.room_id} is not passable", {"routed_rooms": set()}
        if source.occupancy.total == 0:
            return False, f"Source room {source.room_id} is empty", {"routed_rooms": set()}
        if not _occupancy_leq(action.occupancy, source.occupancy):
            return False, "Requested occupancy exceeds room occupancy", {"routed_rooms": set()}

        valid, reason, transits = self._build_transit_groups(
            ep,
            source,
            action.to_node_id,
            action.occupancy,
            preference=action.preference,
        )
        if not valid:
            return False, reason, {"routed_rooms": set()}

        ep.civilians_in_transit.extend(transits)
        ep.panic_timers[source.room_id] = 0
        return True, None, {"routed_rooms": {source.room_id}}

    def _handle_evacuate_floor(
        self,
        ep: EpisodeStateInternal,
        action: EvacuateFloorAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        floor = next(
            (floor for floor in ep.building.floors if floor.floor_id == action.floor_id),
            None,
        )
        if floor is None:
            return False, f"Unknown floor: {action.floor_id}", {"routed_rooms": set()}

        priority_modes = self._room_priority_modes[ep.episode_id]
        occupied_rooms = [room for room in floor.rooms if room.occupancy.total > 0]
        if not occupied_rooms:
            return False, f"Floor {action.floor_id} has no civilians to evacuate", {"routed_rooms": set()}

        occupied_rooms.sort(
            key=lambda room: (
                room.room_id not in ep.prioritized_rooms,
                room.room_id,
            )
        )

        routed_rooms: set[str] = set()
        for room in occupied_rooms:
            path = self._find_path_to_exit(
                ep,
                room.room_id,
                preferred_exit_id=action.preferred_exit_id,
                avoid_blocked_routes=True,
            )
            if not path or len(path) < 2:
                continue

            requested = _clone_occupancy(room.occupancy)
            priority = priority_modes.get(room.room_id, "all")
            if priority == "mobile_only":
                requested.injured = 0
                requested.mobility_impaired = 0

            valid, _, transits = self._build_transit_groups(
                ep,
                room,
                path[1],
                requested,
                preference="injured_first" if priority == "injured_first" else "fastest",
            )
            if not valid or not transits:
                continue
            ep.civilians_in_transit.extend(transits)
            ep.panic_timers[room.room_id] = 0
            routed_rooms.add(room.room_id)

        if not routed_rooms:
            return False, "No passable evacuation routes found for the floor", {"routed_rooms": set()}
        return True, None, {"routed_rooms": routed_rooms}

    def _handle_prioritize_room(
        self,
        ep: EpisodeStateInternal,
        action: PrioritizeRoomAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        if action.room_id not in self._room_lookup(ep.building):
            return False, f"Unknown room: {action.room_id}", {"routed_rooms": set()}
        ep.prioritized_rooms.add(action.room_id)
        self._room_priority_modes[ep.episode_id][action.room_id] = action.priority
        return True, None, {"routed_rooms": set()}

    def _handle_block_route(
        self,
        ep: EpisodeStateInternal,
        edge_id: str,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        valid_edge_ids = {
            corridor.corridor_id
            for floor in ep.building.floors
            for corridor in floor.corridors
        }
        valid_edge_ids.update(self._stairwell_lookup(ep.building))
        valid_edge_ids.update(self._elevator_lookup(ep.building))
        if edge_id not in valid_edge_ids:
            return False, f"Unknown edge: {edge_id}", {"routed_rooms": set()}
        ep.blocked_routes.add(edge_id)
        return True, None, {"routed_rooms": set()}

    def _handle_call_elevator(
        self,
        ep: EpisodeStateInternal,
        action: CallElevatorAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        elevator = self._elevator_lookup(ep.building).get(action.elevator_id)
        if elevator is None:
            return False, f"Unknown elevator: {action.elevator_id}", {"routed_rooms": set()}
        if not elevator.operational:
            return False, f"Elevator {action.elevator_id} is not operational", {"routed_rooms": set()}
        if action.source_floor not in elevator.floor_ids or action.target_floor not in elevator.floor_ids:
            return False, "Elevator does not serve the requested floors", {"routed_rooms": set()}

        self._set_elevator_target(ep, action.elevator_id, action.source_floor)
        if action.target_floor != action.source_floor:
            direction = "up" if action.target_floor > action.source_floor else "down"
            for elevator_copy in self._iter_elevator_copies(ep.building, action.elevator_id):
                elevator_copy.queue.append(
                    {
                        "floor_id": action.target_floor,
                        "direction": direction,
                    }
                )
        return True, None, {"routed_rooms": set()}

    def _handle_open_exit(
        self,
        ep: EpisodeStateInternal,
        action: OpenExitAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        exit_obj = self._exit_lookup(ep.building).get(action.exit_id)
        if exit_obj is None:
            return False, f"Unknown exit: {action.exit_id}", {"routed_rooms": set()}
        if not exit_obj.blocked:
            return False, f"Exit {action.exit_id} is already open", {"routed_rooms": set()}
        self._pending_exit_opens[ep.episode_id].add(action.exit_id)
        return True, None, {"routed_rooms": set()}

    def _handle_lockdown_room(
        self,
        ep: EpisodeStateInternal,
        action: LockdownRoomAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        room = self._room_lookup(ep.building).get(action.room_id)
        if room is None:
            return False, f"Unknown room: {action.room_id}", {"routed_rooms": set()}
        room.accessible = False
        self._engines[ep.episode_id]._room_accessibility[action.room_id] = False
        return True, None, {"routed_rooms": set()}

    def _handle_request_render(
        self,
        ep: EpisodeStateInternal,
        action: RequestRenderAction,
    ) -> tuple[bool, Optional[str], dict[str, object]]:
        if action.floor_id not in {floor.floor_id for floor in ep.building.floors}:
            return False, f"Unknown floor: {action.floor_id}", {"routed_rooms": set()}
        self._requested_render_floors[ep.episode_id] = action.floor_id
        return True, None, {"routed_rooms": set()}

    def _build_transit_groups(
        self,
        ep: EpisodeStateInternal,
        source: Room,
        target_node_id: str,
        occupancy: Occupancy,
        *,
        preference: str,
    ) -> tuple[bool, Optional[str], list[TransitGroup]]:
        connection = self._resolve_connection(ep, source.room_id, target_node_id)
        if connection is None:
            return False, f"{source.room_id} is not connected to {target_node_id}", []

        path_kind, carrier_id = connection
        room_lookup = self._room_lookup(ep.building)
        exit_lookup = self._exit_lookup(ep.building)

        if target_node_id in room_lookup:
            target_room = room_lookup[target_node_id]
            if not target_room.accessible or not target_room.hazard.passable:
                return False, f"Target room {target_room.room_id} is not passable", []
        else:
            exit_obj = exit_lookup.get(target_node_id)
            if exit_obj is None:
                return False, f"Unknown target node: {target_node_id}", []
            if exit_obj.blocked:
                return False, f"Exit {target_node_id} is blocked", []

        if occupancy.mobility_impaired > 0 and path_kind == "stairwell":
            alternate = self._resolve_connection(
                ep,
                source.room_id,
                target_node_id,
                prefer_elevator=True,
            )
            if alternate is not None and alternate[0] == "elevator":
                path_kind, carrier_id = alternate
            else:
                return False, "Mobility-impaired civilians cannot use stairwells", []

        pieces = [
            ("mobile", occupancy.mobile),
            ("injured", occupancy.injured),
            ("mobility_impaired", occupancy.mobility_impaired),
        ]
        if preference == "injured_first":
            pieces = [
                ("injured", occupancy.injured),
                ("mobile", occupancy.mobile),
                ("mobility_impaired", occupancy.mobility_impaired),
            ]

        elevator = self._elevator_lookup(ep.building).get(carrier_id or "")
        pending: list[tuple[Occupancy, int]] = []
        for cohort_name, count in pieces:
            if count <= 0:
                continue

            group_occupancy = Occupancy()
            setattr(group_occupancy, cohort_name, count)
            steps_remaining = 2 if cohort_name == "injured" else 1

            if path_kind == "elevator" and elevator is not None:
                source_floor = source.floor_id
                target_floor = _floor_number(target_node_id)
                floor_distance = abs((target_floor or source_floor) - source_floor)
                steps_remaining = max(
                    steps_remaining,
                    max(1, floor_distance) * elevator.travel_time_per_floor,
                )
                if elevator.current_floor != source_floor:
                    return False, f"Elevator {elevator.elevator_id} is not at floor {source_floor}", []
                if group_occupancy.total > elevator.capacity:
                    return False, f"Elevator {elevator.elevator_id} capacity exceeded", []

            pending.append((group_occupancy, steps_remaining))

        transits: list[TransitGroup] = []
        for group_occupancy, steps_remaining in pending:
            _subtract_occupancy(source.occupancy, group_occupancy)
            incident_outcomes = self._pull_room_incident_outcomes(ep, source.room_id, group_occupancy.total)
            transits.append(
                TransitGroup(
                    transit_id=uuid.uuid4().hex,
                    source_node_id=source.room_id,
                    target_node_id=target_node_id,
                    occupancy=group_occupancy,
                    incident_outcomes=incident_outcomes,
                    path_kind=path_kind,
                    carrier_id=carrier_id,
                    steps_remaining=steps_remaining,
                )
            )

        return True, None, transits

    def _resolve_connection(
        self,
        ep: EpisodeStateInternal,
        from_node_id: str,
        to_node_id: str,
        *,
        avoid_blocked_routes: bool = False,
        prefer_elevator: bool = False,
    ) -> Optional[tuple[str, Optional[str]]]:
        if to_node_id in self._exit_lookup(ep.building):
            exit_obj = self._exit_lookup(ep.building)[to_node_id]
            return ("corridor", None) if exit_obj.adjacent_room_id == from_node_id else None

        corridor_pairs = self._corridor_pair_lookup(ep.building)
        stair_pairs = self._stairwell_pair_lookup(ep.building)
        elevator_pairs = self._elevator_pair_lookup(ep.building)
        room_lookup = self._room_lookup(ep.building)
        pair = (from_node_id, to_node_id)

        if pair in corridor_pairs:
            corridor = corridor_pairs[pair]
            if not corridor.hazard.passable:
                return None
            if avoid_blocked_routes and corridor.corridor_id in ep.blocked_routes:
                return None
            return ("corridor", corridor.corridor_id)

        def elevator_connection() -> Optional[tuple[str, Optional[str]]]:
            if pair not in elevator_pairs:
                return None
            elevator_id = elevator_pairs[pair]
            elevator = self._elevator_lookup(ep.building)[elevator_id]
            if not elevator.operational:
                return None
            source_floor = _floor_number(from_node_id)
            if source_floor is not None and elevator.current_floor != source_floor:
                return None
            if avoid_blocked_routes and elevator_id in ep.blocked_routes:
                return None
            if to_node_id in room_lookup:
                target = room_lookup[to_node_id]
                if not target.accessible or not target.hazard.passable:
                    return None
            return ("elevator", elevator_id)

        def stair_connection() -> Optional[tuple[str, Optional[str]]]:
            if pair not in stair_pairs:
                return None
            stairwell_id = stair_pairs[pair]
            stairwell = self._stairwell_lookup(ep.building)[stairwell_id]
            if stairwell.blocked:
                return None
            if avoid_blocked_routes and stairwell_id in ep.blocked_routes:
                return None
            if to_node_id in room_lookup:
                target = room_lookup[to_node_id]
                if not target.accessible or not target.hazard.passable:
                    return None
            return ("stairwell", stairwell_id)

        preferred_connections = (
            (elevator_connection, stair_connection)
            if prefer_elevator
            else (stair_connection, elevator_connection)
        )
        for resolver in preferred_connections:
            connection = resolver()
            if connection is not None:
                return connection

        return None

    def _resolve_transit(self, ep: EpisodeStateInternal) -> None:
        room_lookup = self._room_lookup(ep.building)
        exit_lookup = self._exit_lookup(ep.building)
        remaining: list[TransitGroup] = []

        for transit in ep.civilians_in_transit:
            transit.steps_remaining -= 1

            exit_obj = exit_lookup.get(transit.target_node_id)
            if exit_obj is not None and transit.steps_remaining <= 0:
                if exit_obj.blocked:
                    source_room = room_lookup.get(transit.source_node_id)
                    source_passable = (
                        source_room is not None
                        and source_room.accessible
                        and source_room.hazard.passable
                    )
                    if source_passable:
                        _add_occupancy(source_room.occupancy, transit.occupancy)
                        self._push_room_incident_outcomes(
                            ep,
                            source_room.room_id,
                            transit.incident_outcomes,
                        )
                    else:
                        _add_occupancy(ep.civilians_lost, transit.occupancy)
                        ep.resolved_incident_outcomes.deaths += transit.incident_outcomes.total
                    continue
                _add_occupancy(ep.civilians_saved, transit.occupancy)
                _add_incident_outcomes(ep.resolved_incident_outcomes, transit.incident_outcomes)
                continue

            if transit.steps_remaining > 0:
                remaining.append(transit)
                continue

            target_room = room_lookup.get(transit.target_node_id)
            if target_room is None or not target_room.accessible or not target_room.hazard.passable:
                _add_occupancy(ep.civilians_lost, transit.occupancy)
                ep.resolved_incident_outcomes.deaths += transit.incident_outcomes.total
                continue

            _add_occupancy(target_room.occupancy, transit.occupancy)
            self._push_room_incident_outcomes(ep, target_room.room_id, transit.incident_outcomes)

        ep.civilians_in_transit = remaining

    def _check_panic(self, ep: EpisodeStateInternal) -> None:
        if ep.task.disaster_type != DisasterType.multi_cascade:
            return

        room_lookup = self._room_lookup(ep.building)
        for room in self._iter_rooms(ep.building):
            if room.occupancy.total == 0:
                ep.panic_timers.pop(room.room_id, None)
                continue

            timer = ep.panic_timers.get(room.room_id, 0) + 1
            neighbors = sorted(
                neighbor_id
                for neighbor_id in room.adjacent_node_ids
                if neighbor_id in room_lookup
                and room_lookup[neighbor_id].accessible
                and room_lookup[neighbor_id].hazard.passable
            )
            if timer < 3 or not neighbors:
                ep.panic_timers[room.room_id] = timer
                continue

            target_room_id = self._rngs[ep.episode_id].choice(neighbors)
            panic_move = Occupancy(mobile=room.occupancy.mobile, injured=room.occupancy.injured)
            if panic_move.total == 0:
                ep.panic_timers[room.room_id] = timer
                continue

            valid, _, transits = self._build_transit_groups(
                ep,
                room,
                target_room_id,
                panic_move,
                preference="fastest",
            )
            if valid and transits:
                ep.civilians_in_transit.extend(transits)
                ep.panic_timers[room.room_id] = 0
            else:
                ep.panic_timers[room.room_id] = timer

    def _resolve_hazard_casualties(self, ep: EpisodeStateInternal) -> None:
        for room in self._iter_rooms(ep.building):
            casualty_count = self._hazard_casualty_count(room)
            if casualty_count == 0:
                continue
            lost = _remove_occupancy(room.occupancy, casualty_count)
            _add_occupancy(ep.civilians_lost, lost)
            self._record_room_incident_deaths(ep, room.room_id, lost.total)

    def _hazard_casualty_count(self, room: Room) -> int:
        if room.occupancy.total == 0:
            return 0

        hazard = room.hazard
        if hazard.hazard_type == HazardType.threat:
            return 0
        if hazard.hazard_type == HazardType.structural:
            if hazard.structural_integrity <= 0.0 and not hazard.passable:
                return room.occupancy.total
            return 0
        if hazard.hazard_type == HazardType.fire:
            if hazard.severity >= 0.95:
                return min(2, room.occupancy.total)
            if hazard.severity >= 0.75:
                return 1
            return 0
        if hazard.hazard_type == HazardType.gas:
            if hazard.severity >= 0.95:
                return min(2, room.occupancy.total)
            if hazard.severity >= 0.85:
                return 1
            return 0
        if hazard.hazard_type == HazardType.flood:
            if hazard.water_level >= 1.0 and hazard.severity >= 1.0:
                return 1
        return 0

    def _capture_disaster_losses(
        self,
        ep: EpisodeStateInternal,
        snapshot: dict[str, Occupancy],
    ) -> None:
        room_lookup = self._room_lookup(ep.building)
        for room_id, before in snapshot.items():
            after = room_lookup[room_id].occupancy
            lost = _occupancy_delta(before, after)
            if lost.total > 0:
                _add_occupancy(ep.civilians_lost, lost)
                self._record_room_incident_deaths(ep, room_id, lost.total)

    def _advance_elevators(self, ep: EpisodeStateInternal) -> None:
        progress = self._elevator_progress[ep.episode_id]
        for elevator_id, elevator in self._elevator_lookup(ep.building).items():
            if not elevator.operational or elevator.target_floor is None:
                continue
            if elevator.current_floor == elevator.target_floor:
                self._set_elevator_target(ep, elevator_id, None)
                if elevator.queue:
                    next_request = elevator.queue.pop(0)
                    self._set_elevator_target(ep, elevator_id, int(next_request["floor_id"]))
                continue

            remaining = progress.get(elevator_id)
            if remaining is None:
                remaining = abs(elevator.target_floor - elevator.current_floor) * elevator.travel_time_per_floor
            remaining -= 1
            if remaining <= 0:
                self._set_elevator_floor(ep, elevator_id, elevator.target_floor)
                self._set_elevator_target(ep, elevator_id, None)
                progress.pop(elevator_id, None)
                if elevator.queue:
                    next_request = elevator.queue.pop(0)
                    self._set_elevator_target(ep, elevator_id, int(next_request["floor_id"]))
            else:
                progress[elevator_id] = remaining

    def _apply_pending_exit_opens(self, ep: EpisodeStateInternal) -> None:
        pending = self._pending_exit_opens[ep.episode_id]
        if not pending:
            return
        for exit_id in pending:
            exit_obj = self._exit_lookup(ep.building).get(exit_id)
            if exit_obj is not None:
                exit_obj.blocked = False
        pending.clear()

    def _update_termination(self, ep: EpisodeStateInternal) -> None:
        ep.done = False
        ep.termination_reason = None

        if ep.civilians_saved.total == ep.total_civilians.total and not ep.civilians_in_transit:
            ep.done = True
            ep.termination_reason = TerminationReason.all_saved
            return

        resolved_total = ep.civilians_saved.total + ep.civilians_lost.total
        if resolved_total == ep.total_civilians.total and not ep.civilians_in_transit:
            ep.done = True
            if ep.civilians_lost.total == ep.total_civilians.total:
                ep.termination_reason = TerminationReason.all_lost
            else:
                ep.termination_reason = TerminationReason.all_routes_cut
            return

        if self._all_routes_cut(ep):
            ep.done = True
            ep.termination_reason = TerminationReason.all_routes_cut
            return

        if ep.step >= ep.task.max_steps:
            ep.done = True
            ep.termination_reason = TerminationReason.max_steps
            return

    def _all_routes_cut(self, ep: EpisodeStateInternal) -> bool:
        occupied_rooms = [room for room in self._iter_rooms(ep.building) if room.occupancy.total > 0]
        if not occupied_rooms:
            return False
        return all(self._find_path_to_exit(ep, room.room_id) is None for room in occupied_rooms)

    def _build_observations_by_role(self, ep: EpisodeStateInternal) -> ObservationsByRole:
        floors = {
            self._floor_agent_key(floor.floor_id): self._build_floor_agent_observation(ep, floor.floor_id)
            for floor in ep.building.floors
        }
        return ObservationsByRole(
            orchestrator=self._build_orchestrator_observation(ep),
            floors=floors,
        )

    def _build_floor_agent_observation(
        self,
        ep: EpisodeStateInternal,
        floor_num: int,
        *,
        agent_id: Optional[str] = None,
    ) -> FloorAgentObservationMA:
        floor = next(floor for floor in ep.building.floors if floor.floor_id == floor_num)
        floor_agent_id = agent_id or self._floor_agent_key(floor_num)
        visibility_state = self._get_floor_visibility_state(ep, floor_agent_id)
        visible_rooms, visible_corridors, visibility_age_by_room, sensor_quality = build_floor_observation(
            floor_id=floor_num,
            building=ep.building,
            hazard_engine=self._engines[ep.episode_id],
            vis_state=visibility_state,
            current_round=ep.step,
            rng_seed=self._observation_seed(ep, floor_agent_id),
            config=self._visibility_config,
        )
        ep.floor_visibility_state[floor_agent_id] = visibility_state.model_dump(mode="json")

        visible_room_ids = {room.room_id for room in visible_rooms}
        visible_groups = [
            CivilianGroupView(
                civilian_group_id=f"cg_{room.room_id}",
                location_room_id=room.room_id,
                mobility_profile="mixed",
                count=room.occupancy.total,
                status="waiting",
            )
            for room in sorted(floor.rooms, key=lambda item: item.room_id)
            if room.room_id in visible_room_ids and room.occupancy.total > 0
        ]
        local_hazards = [
            HazardView(
                hazard_id=f"haz_{room.room_id}",
                hazard_type=room.hazard.hazard_type.value,
                severity=room.hazard.severity,
                room_id=room.room_id,
                projected_spread=None,
            )
            for room in sorted(floor.rooms, key=lambda item: item.room_id)
            if room.room_id in visible_room_ids and room.hazard.hazard_type is not None and (room.hazard.severity > 0 or room.hazard.smoke > 0)
        ]
        stairwell_entries = [
            StairwellEntryView(
                stairwell_id=stairwell.stairwell_id,
                connects_floor_ids=stairwell.floor_ids,
                blocked=stairwell.blocked,
                capacity_per_step=stairwell.capacity_per_step,
            )
            for stairwell in sorted(floor.stairwells, key=lambda item: item.stairwell_id)
        ]
        exits_on_floor = [
            ExitView(
                exit_id=exit_obj.exit_id,
                floor_id=exit_obj.floor_id,
                exit_type=exit_obj.exit_type.value,
                blocked=exit_obj.blocked,
                requires_open_action=exit_obj.requires_open_action,
            )
            for exit_obj in sorted(floor.exits, key=lambda item: item.exit_id)
        ]
        return FloorAgentObservationMA(
            episode_id=ep.episode_id,
            round_id=ep.step,
            agent_id=floor_agent_id,
            step=ep.step,
            max_steps=ep.task.max_steps,
            seed=ep.seed,
            tier=self._ma_tier(ep),
            disaster_family=ep.task.disaster_type.value,
            action_mask=action_mask_for_role(AgentRole.floor_agent),
            last_reward_breakdown=ep.last_floor_reward_breakdowns.get(floor_agent_id, {}),
            floor_id=self._floor_public_id(floor_num),
            visible_rooms=visible_rooms,
            visible_corridors=visible_corridors,
            stairwell_entries=stairwell_entries,
            exits_on_floor=exits_on_floor,
            visible_civilian_groups=visible_groups,
            local_hazards=local_hazards,
            sensor_quality=sensor_quality,
            visibility_age_by_room=visibility_age_by_room,
            active_directive=self._get_active_directive(ep, floor_agent_id),
            override_applied_last_round=floor_agent_id in self._override_last_round.get(ep.episode_id, {}),
            override_reason_last_round=self._override_last_round.get(ep.episode_id, {}).get(floor_agent_id),
            belief_horizon_limit=ep.belief_store.belief_horizon_limit if ep.belief_store is not None else 8,
            open_belief_slots=ep.belief_store.open_slots(floor_agent_id, 1) if ep.belief_store is not None else 1,
            last_prediction_score=ep.last_prediction_score_by_agent.get(floor_agent_id, 0.0),
            generator_config_hash=self._generator_config_hash(ep.task.task_id, ep.seed),
        )

    def _build_orchestrator_observation(self, ep: EpisodeStateInternal) -> OrchestratorObservationMA:
        floor_summaries = []
        for floor in ep.building.floors:
            floor_agent_id = self._floor_agent_key(floor.floor_id)
            visibility_state = self._get_floor_visibility_state(ep, floor_agent_id)
            visible_rooms, _, visibility_age_by_room, _ = build_floor_observation(
                floor_id=floor.floor_id,
                building=ep.building,
                hazard_engine=self._engines[ep.episode_id],
                vis_state=visibility_state,
                current_round=ep.step,
                rng_seed=self._observation_seed(ep, floor_agent_id),
                config=self._visibility_config,
            )
            floor_summaries.append(
                FloorSummary(
                    floor_id=self._floor_public_id(floor.floor_id),
                    known_civilian_count=sum(room.occupancy_mobile + room.occupancy_injured + room.occupancy_mobility_impaired for room in visible_rooms),
                    unknown_room_count=sum(1 for age in visibility_age_by_room.values() if age > 0),
                    hazard_severity=sum(room.hazard.severity for room in floor.rooms) / max(1, len(floor.rooms)),
                    queue_pressure=self._compute_queue_pressure(floor, ep),
                    exit_capacity_remaining=sum(1 for exit_obj in floor.exits if not exit_obj.blocked),
                    last_updated_round=ep.step,
                )
            )

        stairwells = [
            StairwellAggregateView(
                stairwell_id=stairwell_id,
                floor_ids=stairwell.floor_ids,
                blocked=stairwell.blocked,
                current_load=0,
                capacity=stairwell.capacity_per_step,
            )
            for stairwell_id, stairwell in sorted(self._stairwell_lookup(ep.building).items())
        ]
        elevators = sorted(self._elevator_lookup(ep.building).items())
        elevator_view = None
        if elevators:
            elevator_id, elevator = elevators[0]
            elevator_view = ElevatorView(
                elevator_id=elevator_id,
                current_floor=elevator.current_floor,
                target_floor=elevator.target_floor,
                operational=elevator.operational,
                capacity=elevator.capacity,
                queue_length=len(elevator.queue),
            )
        exit_queue = [
            ExitQueueView(
                exit_id=exit_obj.exit_id,
                floor_id=exit_obj.floor_id,
                queue_depth=0,
                throughput_per_round=1.0,
                blocked=exit_obj.blocked,
            )
            for exit_obj in sorted(self._exit_lookup(ep.building).values(), key=lambda item: item.exit_id)
        ]
        belief_snapshot = ep.belief_store.snapshot() if ep.belief_store is not None else {}
        cascade_hint = self._next_cascade_hint(ep)
        return OrchestratorObservationMA(
            episode_id=ep.episode_id,
            round_id=ep.step,
            agent_id="orchestrator",
            step=ep.step,
            max_steps=ep.task.max_steps,
            seed=ep.seed,
            tier=self._ma_tier(ep),
            disaster_family=ep.task.disaster_type.value,
            action_mask=action_mask_for_role(AgentRole.orchestrator),
            belief_rollup=BeliefRollup.model_validate(belief_snapshot),
            floor_summaries=floor_summaries,
            inter_floor=InterFloorView(
                stairwells=stairwells,
                elevator=elevator_view,
                global_exit_queue=exit_queue,
            ),
            recent_floor_actions=[
                ActionLogEntry(
                    agent_id=record["agent_id"],
                    floor_id=record["floor_id"],
                    action_type=record["action_type"],
                    round_id=record["round_id"],
                    summary=record["summary"],
                )
                for record in self._recent_floor_actions(ep)[:10]
            ],
            recent_directive_outcomes=self._get_directive_outcomes(ep),
            cascade_hint=cascade_hint,
            unresolved_escalations=self._get_unresolved_escalations(ep),
            generator_config_hash=self._generator_config_hash(ep.task.task_id, ep.seed),
        )

    def _build_observation(
        self,
        ep: EpisodeStateInternal,
        render_floor: Optional[int] = None,
    ) -> Observation:
        rooms = []
        for floor in ep.building.floors:
            for room in floor.rooms:
                rooms.append(
                    RoomObservation(
                        room_id=room.room_id,
                        floor_id=room.floor_id,
                        occupancy=_clone_occupancy(room.occupancy),
                        hazard_type=room.hazard.hazard_type,
                        hazard_severity=room.hazard.severity,
                        smoke=room.hazard.smoke,
                        water_level=getattr(room.hazard, "water_level", 0.0),
                        structural_integrity=getattr(room.hazard, "structural_integrity", 1.0),
                        passable=getattr(room.hazard, "passable", True),
                        accessible=room.accessible,
                        connected_rooms=list(room.adjacent_node_ids),
                    )
                )

        corridors = [
            {
                "corridor_id": corridor.corridor_id,
                "from_node_id": corridor.from_node_id,
                "to_node_id": corridor.to_node_id,
                "hazard_severity": corridor.hazard.severity,
                "passable": corridor.hazard.passable,
            }
            for floor in ep.building.floors
            for corridor in floor.corridors
        ]
        stairwells = [
            {
                "stairwell_id": stairwell_id,
                "floor_ids": stairwell.floor_ids,
                "blocked": stairwell.blocked,
                "capacity_per_step": stairwell.capacity_per_step,
            }
            for stairwell_id, stairwell in self._stairwell_lookup(ep.building).items()
        ]
        exits = [
            {
                "exit_id": exit_obj.exit_id,
                "floor_id": exit_obj.floor_id,
                "exit_type": exit_obj.exit_type,
                "blocked": exit_obj.blocked,
                "requires_open_action": exit_obj.requires_open_action,
            }
            for exit_obj in self._exit_lookup(ep.building).values()
        ]

        summary = SummaryObservation(
            disaster_type=ep.task.disaster_type,
            disaster_origin=self._disaster_origins[ep.episode_id],
            goal=ep.task.goal,
            total_civilians=ep.total_civilians.total,
            civilians_saved=ep.civilians_saved.total,
            civilians_lost=ep.civilians_lost.total,
            civilians_in_transit=sum(group.occupancy.total for group in ep.civilians_in_transit),
            elevators_operational=sum(
                1
                for elevator in self._elevator_lookup(ep.building).values()
                if elevator.operational
            ),
            incident_outcomes=self._resolved_incident_outcomes(ep),
        )

        observation = Observation(
            episode_id=ep.episode_id,
            task_id=ep.task.task_id,
            step=ep.step,
            max_steps=ep.task.max_steps,
            summary=summary,
            rooms=rooms,
            corridors=corridors,
            stairwells=stairwells,
            exits=exits,
        )
        if render_floor is not None:
            # Renderer not available in evacos_ma fork; return a placeholder.
            render = RenderObservation(floor_id=render_floor, image_base64="")
            return observation.model_copy(update={"render": render})
        return observation

    def _model_to_dict(self, value: dict[str, Any] | BaseModel | None) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return dict(value)

    def _floor_agent_key(self, floor_num: int) -> str:
        return f"floor_{floor_num}_agent"

    def _floor_public_id(self, floor_num: int) -> str:
        return f"floor_{floor_num}"

    def _floor_number_from_public_id(self, floor_id: int | str) -> int:
        if isinstance(floor_id, int):
            return floor_id
        if floor_id.startswith("floor_"):
            return int(floor_id.split("_", maxsplit=1)[1])
        if floor_id.startswith("F") and floor_id[1:].isdigit():
            return int(floor_id[1:])
        return int(floor_id)

    def _generator_config_hash(self, task_id: str, seed: int) -> str:
        digest = hashlib.sha256(f"{task_id}:{seed}".encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    def _ma_tier(self, ep: EpisodeStateInternal):
        from evacos_ma.schemas.multi_agent import Tier

        return Tier(self._episode_tier_value(ep))

    def _episode_tier_value(self, ep: EpisodeStateInternal) -> str:
        belief_store = getattr(ep, "belief_store", None)
        belief_tier = getattr(belief_store, "tier", None)
        if hasattr(belief_tier, "value"):
            belief_tier = getattr(belief_tier, "value")
        if belief_tier in {"easy", "medium", "hard", "brutal"}:
            return str(belief_tier)
        return self._ma_tier_value(ep.task.difficulty)

    def _episode_reward_tier(self, ep: EpisodeStateInternal) -> str:
        return self._episode_tier_value(ep)

    def _ma_tier_value(self, difficulty: str) -> str:
        if difficulty == "medium_hard":
            return "medium"
        return difficulty

    def _get_floor_visibility_state(
        self,
        ep: EpisodeStateInternal,
        agent_id: str,
    ) -> FloorVisibilityState:
        raw = ep.floor_visibility_state.get(agent_id, {})
        return FloorVisibilityState.model_validate(raw or {})

    def _observation_seed(self, ep: EpisodeStateInternal, agent_id: str) -> int:
        seed_material = f"{ep.episode_id}:{ep.step}:{agent_id}"
        return int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)

    def _mark_room_scouted(self, ep: EpisodeStateInternal, agent_id: str, room_id: str) -> None:
        visibility_state = self._get_floor_visibility_state(ep, agent_id)
        visibility_state.scouted_rooms_this_round.add(room_id)
        ep.floor_visibility_state[agent_id] = visibility_state.model_dump(mode="json")

    def _clear_scout_marks(self, ep: EpisodeStateInternal) -> None:
        for agent_id in list(ep.floor_visibility_state):
            visibility_state = self._get_floor_visibility_state(ep, agent_id)
            visibility_state.scouted_rooms_this_round.clear()
            ep.floor_visibility_state[agent_id] = visibility_state.model_dump(mode="json")

    def _register_belief(
        self,
        ep: EpisodeStateInternal,
        agent_id: str,
        belief: StructuredBelief,
    ) -> BeliefRegistration:
        belief_copy = belief.model_copy(deep=True)
        belief_copy.predictor_agent_id = agent_id
        belief_copy.created_round = ep.step
        return ep.belief_store.register(belief_copy, predictor=agent_id, slot_limit=1)

    def _belief_ground_truth(
        self,
        ep: EpisodeStateInternal,
        belief: StructuredBelief,
        current_round: int,
    ) -> dict[str, Any]:
        del current_round
        room_lookup = self._room_lookup(ep.building)
        room_id = belief.target_entity_ids[0] if belief.target_entity_ids else ""
        room = room_lookup.get(room_id)
        if room is None:
            return {}
        return {
            "expected_civilians_in_room": room.occupancy.total,
            "expected_hazard_severity_room": room.hazard.severity,
            "expected_room_passable": room.accessible and room.hazard.passable,
        }

    def _resolve_beliefs(self, ep: EpisodeStateInternal) -> list[BeliefAuditRow]:
        if ep.belief_store is None:
            return []
        audit_rows = ep.belief_store.tick(
            current_round=ep.step,
            ground_truth_provider=lambda belief, current_round: self._belief_ground_truth(ep, belief, current_round),
        )
        for row in audit_rows:
            ep.last_prediction_score_by_agent[row.predictor_agent_id] = row.score or 0.0
            ep.belief_audit_log.append(row.model_dump(mode="json"))
        return audit_rows

    def _rollout_reward_config(self, ep: EpisodeStateInternal) -> dict[str, Any]:
        payload: dict[str, Any] = dict(_DEFAULT_ROLLOUT_REWARD_CONFIG)
        reward_config = ep.rollout_metadata.get("reward_config", {})
        if isinstance(reward_config, dict):
            payload.update(reward_config)
        rationale_mode = ep.rollout_metadata.get("rationale_mode")
        if rationale_mode is not None:
            payload["rationale_scaling"] = rationale_mode
        return payload

    def _rationale_gate_passes(self, rationale: str | None) -> tuple[bool, int]:
        tokens = _normalize_rationale_tokens(rationale)
        raw_token_count = len(tokens)
        if raw_token_count < 12:
            return False, raw_token_count
        lexical_diversity = len(set(tokens)) / raw_token_count
        if lexical_diversity < 0.4:
            return False, raw_token_count
        if _has_repeated_ngram(tokens, size=4, max_occurrences=2):
            return False, raw_token_count
        return True, raw_token_count

    def _scale_rationale_bonus(
        self,
        raw_eligible_token_count: int,
        reward_config: dict[str, Any],
    ) -> float:
        rationale_scaling = str(
            reward_config.get("rationale_scaling", _DEFAULT_ROLLOUT_REWARD_CONFIG["rationale_scaling"])
        )
        if rationale_scaling == "off":
            return 0.0
        eligible_tokens = min(
            raw_eligible_token_count,
            int(
                reward_config.get(
                    "eligible_token_ceiling",
                    _DEFAULT_ROLLOUT_REWARD_CONFIG["eligible_token_ceiling"],
                )
            ),
        )
        if rationale_scaling == "linear_capped":
            return min(
                float(reward_config.get("alpha", _DEFAULT_ROLLOUT_REWARD_CONFIG["alpha"]))
                * eligible_tokens,
                float(reward_config.get("cap", _DEFAULT_ROLLOUT_REWARD_CONFIG["cap"])),
            )
        if rationale_scaling == "log_uncapped":
            return float(reward_config.get("beta", _DEFAULT_ROLLOUT_REWARD_CONFIG["beta"])) * math.log(
                1 + eligible_tokens
            )
        return 0.0

    def _orchestrator_rationale_bonus(
        self,
        *,
        action: ActionEnvelopeMA | None,
        counterfactual_deltas: dict[str, float],
        reward_config: dict[str, Any],
    ) -> float:
        if action is None:
            return 0.0
        if action.action_type != ActionTypeMA.override_floor_agent:
            return 0.0
        passes, raw_token_count = self._rationale_gate_passes(action.rationale)
        if not passes:
            return 0.0
        target_agent_id = str(action.arguments.get("target_floor_agent_id", ""))
        if counterfactual_deltas.get(target_agent_id, 0.0) <= 0.0:
            return 0.0
        return self._scale_rationale_bonus(raw_token_count, reward_config)

    def _floor_rationale_bonus(
        self,
        *,
        action: ActionEnvelopeMA,
        belief_score: float,
        reward_config: dict[str, Any],
    ) -> float:
        if action.action_type != ActionTypeMA.predict_state:
            return 0.0
        passes, raw_token_count = self._rationale_gate_passes(action.rationale)
        if not passes or belief_score < 0.5:
            return 0.0
        return self._scale_rationale_bonus(raw_token_count, reward_config)

    def _build_role_reward(
        self,
        ep: EpisodeStateInternal,
        components: dict[str, float],
    ) -> RoleReward:
        """Build a per-role reward envelope.

        ``RoleReward.normalized`` is deprecated and intentionally identical to
        ``raw``. Training-time advantages are computed from raw rewards via
        group-mean-std in ``MultiAgentGRPOTrainer._compute_group_advantages``.
        The field remains for trace/backwards-compatibility.
        """
        del ep
        raw = float(sum(components.values()))
        return RoleReward(
            raw=raw,
            normalized=raw,
            breakdown=RewardBreakdown(**components),
        )

    def _prediction_reward_component(self, ep: EpisodeStateInternal, score: float) -> float:
        return score * ep.task.reward_weights.floor_prediction

    def _scout_reward_component(self, ep: EpisodeStateInternal) -> float:
        return -0.1 * ep.task.reward_weights.floor_scout_cost

    def _duplicate_penalty_component(self, ep: EpisodeStateInternal) -> float:
        return -1.0 * ep.task.reward_weights.duplicate_belief_penalty

    def _recent_floor_actions(self, ep: EpisodeStateInternal) -> list[dict[str, Any]]:
        recent = self._ma_recent_floor_actions.get(ep.episode_id, [])
        return list(reversed(recent[-10:]))

    def _record_ma_floor_actions(
        self,
        episode_id: str,
        floor_actions: dict[str, ActionEnvelopeMA],
        round_id: int,
    ) -> None:
        history = self._ma_recent_floor_actions.setdefault(episode_id, [])
        for agent_id, action in sorted(floor_actions.items()):
            history.append(
                {
                    "agent_id": agent_id,
                    "floor_id": agent_id.rsplit("_agent", 1)[0] if agent_id.endswith("_agent") else None,
                    "action_type": action.action_type.value,
                    "round_id": round_id,
                    "summary": self._summarize_ma_action(action),
                }
            )
        if len(history) > 10:
            del history[:-10]

    def _compute_queue_pressure(self, floor: Floor, ep: EpisodeStateInternal) -> float:
        del ep
        total_civilians = sum(room.occupancy.total for room in floor.rooms)
        open_exits = sum(1 for exit_obj in floor.exits if not exit_obj.blocked)
        open_stairwells = [
            stairwell
            for stairwell in floor.stairwells
            if not getattr(stairwell, "blocked", False)
        ]
        outflow = open_exits * 10 + sum(
            stairwell.capacity_per_step for stairwell in open_stairwells
        )
        if outflow <= 0:
            return 1.0 if total_civilians > 0 else 0.0
        return min(1.0, float(total_civilians) / float(outflow))

    def _next_cascade_hint(self, ep: EpisodeStateInternal) -> dict[str, Any] | None:
        upcoming_events = sorted(
            (
                event
                for event in ep.scheduled_events
                if not event.triggered and event.trigger_step >= ep.step
            ),
            key=lambda event: (event.trigger_step, event.event_id),
        )
        if not upcoming_events:
            return None

        event = upcoming_events[0]
        room_lookup = self._room_lookup(ep.building)
        floor_id = None
        room_id = (
            event.payload.get("origin_room_id")
            or event.payload.get("room_id")
            or event.target_id
        )
        room = room_lookup.get(room_id)
        if room is not None:
            floor_id = room.floor_id

        hint = {
            "event_id": event.event_id,
            "next_cascade_round": event.trigger_step,
            "type": event.event_type.value,
            "target_id": event.target_id,
        }
        if floor_id is not None:
            hint["floor"] = floor_id
        return hint

    def _summarize_ma_action(self, action: ActionEnvelopeMA) -> str:
        target = (
            action.arguments.get("room_id")
            or action.arguments.get("to_room_id")
            or action.arguments.get("stairwell_id")
            or action.arguments.get("exit_id")
            or action.arguments.get("target_floor_agent_id")
        )
        if target:
            return f"{action.action_type.value} -> {target}"
        return action.action_type.value

    def _get_active_directive(
        self,
        ep: EpisodeStateInternal,
        floor_agent_id: str,
    ):
        """Return the highest-priority live directive for a floor agent."""
        ds = self._directive_stores.get(ep.episode_id)
        if ds is None:
            return None
        return ds.active_directive_for_target(floor_agent_id, ep.step)

    def _get_directive_outcomes(self, ep: EpisodeStateInternal) -> list[DirectiveOutcome]:
        """Return all directive outcomes for this episode."""
        ds = self._directive_stores.get(ep.episode_id)
        if ds is None:
            return []
        return ds.directive_outcomes()

    def _get_unresolved_escalations(self, ep: EpisodeStateInternal) -> list[EscalationRequest]:
        """Return unresolved handoff escalations."""
        handoffs = self._handoff_stores.get(ep.episode_id, [])
        return [
            EscalationRequest(
                agent_id=h["agent_id"],
                floor_id=h.get("floor_id", ""),
                category=h.get("category", "resource_contention"),
                urgency=h.get("urgency", "normal"),
                target_ids=h.get("target_ids", []),
                note=h.get("note", ""),
            )
            for h in handoffs
            if not h.get("addressed", False)
        ]

    def _build_elevator_observations(
        self,
        ep: EpisodeStateInternal,
    ) -> list[ElevatorObservation]:
        return [
            ElevatorObservation(
                elevator_id=elevator_id,
                current_floor=elevator.current_floor,
                target_floor=elevator.target_floor,
                operational=elevator.operational,
            )
            for elevator_id, elevator in sorted(self._elevator_lookup(ep.building).items())
        ]

    def _build_transit_group_observations(
        self,
        ep: EpisodeStateInternal,
    ) -> list[TransitGroupObservation]:
        return [
            TransitGroupObservation(
                transit_id=transit.transit_id,
                source_node_id=transit.source_node_id,
                target_node_id=transit.target_node_id,
                occupancy=_clone_occupancy(transit.occupancy),
                steps_remaining=transit.steps_remaining,
            )
            for transit in ep.civilians_in_transit
        ]

    def _compute_reward(
        self,
        ep: EpisodeStateInternal,
        prev_saved: Occupancy,
        prev_lost: Occupancy,
        action: Action,
        action_valid: bool,
    ) -> Reward:
        weights = ep.task.reward_weights
        saved_delta = ep.civilians_saved.total - prev_saved.total
        lost_delta = ep.civilians_lost.total - prev_lost.total

        saved_component = int(saved_delta * weights.civilian_saved)
        lost_component = int(lost_delta * weights.civilian_lost)
        hazard_avoidance_bonus = (
            weights.hazard_avoidance
            if action_valid and self._routed_away_from_hazard(action, ep)
            else 0.0
        )

        vulnerable_group_bonus = 0.0
        if saved_delta > 0:
            injured_saved = ep.civilians_saved.injured - prev_saved.injured
            impaired_saved = ep.civilians_saved.mobility_impaired - prev_saved.mobility_impaired
            vulnerable_group_bonus = (injured_saved + impaired_saved) * weights.vulnerable_bonus

        efficiency_bonus = (
            weights.efficiency
            if action_valid and saved_delta > 0 and self._used_optimal_route(action, ep)
            else 0.0
        )
        invalid_action_penalty = weights.invalid_action if not action_valid else 0.0
        idle_penalty = weights.time_step
        if action.action_type == ActionType.wait and self._civilians_in_danger(ep):
            idle_penalty += weights.idle

        completion_bonus = 0.0
        if ep.done and ep.termination_reason == TerminationReason.all_saved:
            completion_bonus = weights.completion

        total = (
            saved_component
            + lost_component
            + hazard_avoidance_bonus
            + vulnerable_group_bonus
            + efficiency_bonus
            + invalid_action_penalty
            + idle_penalty
            + completion_bonus
        )
        return Reward(
            total=total,
            civilians_saved_delta=saved_component,
            civilians_lost_delta=lost_component,
            hazard_avoidance_bonus=hazard_avoidance_bonus,
            vulnerable_group_bonus=vulnerable_group_bonus,
            efficiency_bonus=efficiency_bonus,
            invalid_action_penalty=invalid_action_penalty,
            idle_penalty=idle_penalty,
            completion_bonus=completion_bonus,
        )

    def _routed_away_from_hazard(self, action: Action, ep: EpisodeStateInternal) -> bool:
        if not isinstance(action, RouteCiviliansAction):
            return False
        source = self._room_lookup(ep.building).get(action.from_node_id)
        if source is None:
            return False
        if action.to_node_id in self._room_lookup(ep.building):
            target_severity = self._room_lookup(ep.building)[action.to_node_id].hazard.severity
        else:
            target_severity = 0.0 if action.to_node_id in self._exit_lookup(ep.building) else 1.0
        return source.hazard.severity > 0.3 and target_severity < source.hazard.severity

    def _used_optimal_route(self, action: Action, ep: EpisodeStateInternal) -> bool:
        if not isinstance(action, RouteCiviliansAction):
            return False
        source_distance = self._distance_to_exit(ep, action.from_node_id)
        target_distance = self._distance_to_exit(ep, action.to_node_id)
        return target_distance is not None and source_distance is not None and target_distance < source_distance

    def _civilians_in_danger(self, ep: EpisodeStateInternal) -> bool:
        return any(
            room.occupancy.total > 0 and room.hazard.severity > 0.3
            for room in self._iter_rooms(ep.building)
        )

    def _update_metrics(
        self,
        ep: EpisodeStateInternal,
        action: Action,
        action_valid: bool,
        prev_saved: Occupancy,
        prev_lost: Occupancy,
        hazard_exposure: float,
        elapsed_ms: float,
    ) -> MetricsDelta:
        action_key = action.action_type.value
        ep.metrics.actions_by_type[action_key] = ep.metrics.actions_by_type.get(action_key, 0) + 1
        if not action_valid:
            ep.metrics.invalid_actions += 1
        if action.action_type == ActionType.request_render:
            ep.metrics.render_requests += 1

        ep.metrics.civilians_saved_mobile = ep.civilians_saved.mobile
        ep.metrics.civilians_saved_injured = ep.civilians_saved.injured
        ep.metrics.civilians_saved_impaired = ep.civilians_saved.mobility_impaired
        ep.metrics.civilians_lost_mobile = ep.civilians_lost.mobile
        ep.metrics.civilians_lost_injured = ep.civilians_lost.injured
        ep.metrics.civilians_lost_impaired = ep.civilians_lost.mobility_impaired
        incident_outcomes = self._resolved_incident_outcomes(ep)
        ep.metrics.incident_safe = incident_outcomes.safe
        ep.metrics.incident_mild_injury = incident_outcomes.mild_injury
        ep.metrics.incident_severe_injury = incident_outcomes.severe_injury
        ep.metrics.incident_deaths = incident_outcomes.deaths
        ep.metrics.cumulative_hazard_exposure += hazard_exposure
        ep.metrics.elapsed_ms += elapsed_ms

        return MetricsDelta(
            civilians_saved_delta=ep.civilians_saved.total - prev_saved.total,
            civilians_lost_delta=ep.civilians_lost.total - prev_lost.total,
            hazard_exposure_delta=hazard_exposure,
        )

    def _current_hazard_exposure(self, ep: EpisodeStateInternal) -> float:
        return sum(
            room.hazard.severity
            for room in self._iter_rooms(ep.building)
            if room.occupancy.total > 0
        )

    def _snapshot_room_occupancy(self, building: Building) -> dict[str, Occupancy]:
        return {
            room.room_id: _clone_occupancy(room.occupancy)
            for room in self._iter_rooms(building)
        }

    def _count_total_civilians(self, building: Building) -> Occupancy:
        total = Occupancy()
        for room in self._iter_rooms(building):
            _add_occupancy(total, room.occupancy)
        return total

    def _sync_room_incident_outcomes(self, ep: EpisodeStateInternal) -> None:
        known_room_ids = set()
        for room in self._iter_rooms(ep.building):
            known_room_ids.add(room.room_id)
            outcomes = ep.room_incident_outcomes.setdefault(room.room_id, IncidentOutcomes())
            delta = room.occupancy.total - outcomes.total
            if delta > 0:
                outcomes.safe += delta
            elif delta < 0:
                _remove_incident_outcomes(outcomes, -delta)

        for room_id in list(ep.room_incident_outcomes):
            if room_id not in known_room_ids:
                ep.room_incident_outcomes.pop(room_id, None)

    def _resolved_incident_outcomes(self, ep: EpisodeStateInternal) -> IncidentOutcomes:
        expected_total = ep.civilians_saved.total + ep.civilians_lost.total
        if ep.resolved_incident_outcomes.total == expected_total:
            return _clone_incident_outcomes(ep.resolved_incident_outcomes)
        return IncidentOutcomes(
            safe=ep.civilians_saved.total,
            deaths=ep.civilians_lost.total,
        )

    def _pull_room_incident_outcomes(
        self,
        ep: EpisodeStateInternal,
        room_id: str,
        count: int,
    ) -> IncidentOutcomes:
        outcomes = ep.room_incident_outcomes.setdefault(room_id, IncidentOutcomes())
        return _remove_incident_outcomes(outcomes, count)

    def _push_room_incident_outcomes(
        self,
        ep: EpisodeStateInternal,
        room_id: str,
        outcomes: IncidentOutcomes,
    ) -> None:
        room_outcomes = ep.room_incident_outcomes.setdefault(room_id, IncidentOutcomes())
        _add_incident_outcomes(room_outcomes, outcomes)

    def _record_room_incident_deaths(
        self,
        ep: EpisodeStateInternal,
        room_id: str,
        count: int,
    ) -> None:
        if count <= 0:
            return
        removed = self._pull_room_incident_outcomes(ep, room_id, count)
        ep.resolved_incident_outcomes.deaths += removed.total

    def _apply_incident_exposure(self, ep: EpisodeStateInternal) -> None:
        for room in self._iter_rooms(ep.building):
            if room.occupancy.total <= 0:
                continue

            outcomes = ep.room_incident_outcomes.setdefault(room.room_id, IncidentOutcomes())
            if outcomes.total <= 0:
                continue

            severe_condition = (
                room.hazard.severity >= 0.7
                or room.hazard.smoke >= 0.6
                or room.hazard.water_level >= 0.7
                or room.hazard.structural_integrity <= 0.4
            )
            mild_condition = (
                room.hazard.severity >= 0.3
                or room.hazard.smoke >= 0.3
                or room.hazard.water_level >= 0.3
                or room.hazard.structural_integrity <= 0.7
            )

            if severe_condition:
                promote = min(1, outcomes.safe + outcomes.mild_injury)
                mild_taken = min(outcomes.mild_injury, promote)
                outcomes.mild_injury -= mild_taken
                outcomes.severe_injury += mild_taken
                promote -= mild_taken
                if promote > 0:
                    safe_taken = min(outcomes.safe, promote)
                    outcomes.safe -= safe_taken
                    outcomes.severe_injury += safe_taken
            elif mild_condition and outcomes.safe > 0:
                outcomes.safe -= 1
                outcomes.mild_injury += 1

    def _find_path_to_exit(
        self,
        ep: EpisodeStateInternal,
        start_room_id: str,
        *,
        preferred_exit_id: Optional[str] = None,
        avoid_blocked_routes: bool = False,
    ) -> Optional[list[str]]:
        room_lookup = self._room_lookup(ep.building)
        if start_room_id not in room_lookup:
            return None
        start_room = room_lookup[start_room_id]
        if not start_room.accessible or not start_room.hazard.passable:
            return None

        exit_lookup = self._exit_lookup(ep.building)
        preferred_exit = exit_lookup.get(preferred_exit_id) if preferred_exit_id else None
        if preferred_exit is not None and preferred_exit.blocked:
            return None

        queue: deque[list[str]] = deque([[start_room_id]])
        visited = {start_room_id}
        while queue:
            path = queue.popleft()
            current_room_id = path[-1]
            current_room = room_lookup[current_room_id]

            exits_here = [
                exit_obj
                for exit_obj in exit_lookup.values()
                if exit_obj.adjacent_room_id == current_room.room_id and not exit_obj.blocked
            ]
            if preferred_exit is not None:
                exits_here = [exit_obj for exit_obj in exits_here if exit_obj.exit_id == preferred_exit.exit_id]
            if exits_here:
                return [*path, sorted(exits_here, key=lambda exit_obj: exit_obj.exit_id)[0].exit_id]

            for neighbor_id in sorted(self._room_neighbors(ep, current_room_id, avoid_blocked_routes)):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append([*path, neighbor_id])
        return None

    def _distance_to_exit(self, ep: EpisodeStateInternal, node_id: str) -> Optional[int]:
        if node_id in self._exit_lookup(ep.building):
            return 0
        path = self._find_path_to_exit(ep, node_id)
        if path is None:
            return None
        return len(path) - 1

    def _room_neighbors(
        self,
        ep: EpisodeStateInternal,
        room_id: str,
        avoid_blocked_routes: bool,
    ) -> list[str]:
        neighbors: list[str] = []
        room_lookup = self._room_lookup(ep.building)
        for edge in ep.building.graph_edges:
            if edge.from_id != room_id or edge.to_id not in room_lookup:
                continue
            if self._resolve_connection(
                ep,
                edge.from_id,
                edge.to_id,
                avoid_blocked_routes=avoid_blocked_routes,
            ) is not None:
                neighbors.append(edge.to_id)
        return neighbors

    def _sync_engine_state(self, ep: EpisodeStateInternal, engine: DisasterEngine) -> None:
        threat_state = getattr(engine, "threat_state", None)
        ep.threat_state = threat_state.model_copy(deep=True) if threat_state is not None else None

        scheduled_events = getattr(engine, "scheduled_events", None)
        ep.scheduled_events = []
        if scheduled_events is not None:
            ep.scheduled_events = [event.model_copy(deep=True) for event in scheduled_events]

    def _iter_rooms(self, building: Building) -> list[Room]:
        return [room for floor in building.floors for room in floor.rooms]

    def _room_lookup(self, building: Building) -> dict[str, Room]:
        return {room.room_id: room for room in self._iter_rooms(building)}

    def _exit_lookup(self, building: Building) -> dict[str, Exit]:
        exits: dict[str, Exit] = {}
        for floor in building.floors:
            for exit_obj in floor.exits:
                exits.setdefault(exit_obj.exit_id, exit_obj)
        return exits

    def _stairwell_lookup(self, building: Building) -> dict[str, object]:
        stairwells = {}
        for floor in building.floors:
            for stairwell in floor.stairwells:
                stairwells.setdefault(stairwell.stairwell_id, stairwell)
        return stairwells

    def _elevator_lookup(self, building: Building) -> dict[str, object]:
        elevators = {}
        for floor in building.floors:
            for elevator in floor.elevators:
                elevators.setdefault(elevator.elevator_id, elevator)
        return elevators

    def _corridor_pair_lookup(self, building: Building) -> dict[tuple[str, str], object]:
        lookup = {}
        for floor in building.floors:
            for corridor in floor.corridors:
                lookup[(corridor.from_node_id, corridor.to_node_id)] = corridor
                lookup[(corridor.to_node_id, corridor.from_node_id)] = corridor
        return lookup

    def _stairwell_pair_lookup(self, building: Building) -> dict[tuple[str, str], str]:
        lookup: dict[tuple[str, str], str] = {}
        for stairwell_id, stairwell in self._stairwell_lookup(building).items():
            ordered_floors = sorted(stairwell.floor_ids)
            for lower_floor, upper_floor in zip(ordered_floors, ordered_floors[1:], strict=False):
                lower_room = stairwell.entry_room_ids[lower_floor]
                upper_room = stairwell.entry_room_ids[upper_floor]
                lookup[(lower_room, upper_room)] = stairwell_id
                lookup[(upper_room, lower_room)] = stairwell_id
        return lookup

    def _elevator_pair_lookup(self, building: Building) -> dict[tuple[str, str], str]:
        lookup: dict[tuple[str, str], str] = {}
        room_lookup = self._room_lookup(building)
        for elevator_id in self._elevator_lookup(building):
            served_rooms = sorted(
                (room.floor_id, room.room_id)
                for room in room_lookup.values()
                if elevator_id in room.adjacent_node_ids
            )
            for (_, lower_room), (_, upper_room) in zip(served_rooms, served_rooms[1:], strict=False):
                lookup[(lower_room, upper_room)] = elevator_id
                lookup[(upper_room, lower_room)] = elevator_id
        return lookup

    def _iter_elevator_copies(self, building: Building, elevator_id: str) -> list[object]:
        return [
            elevator
            for floor in building.floors
            for elevator in floor.elevators
            if elevator.elevator_id == elevator_id
        ]

    def _set_elevator_target(
        self,
        ep: EpisodeStateInternal,
        elevator_id: str,
        target_floor: Optional[int],
    ) -> None:
        for elevator in self._iter_elevator_copies(ep.building, elevator_id):
            elevator.target_floor = target_floor
        if target_floor is None:
            self._elevator_progress[ep.episode_id].pop(elevator_id, None)

    def _set_elevator_floor(
        self,
        ep: EpisodeStateInternal,
        elevator_id: str,
        floor_id: int,
    ) -> None:
        for elevator in self._iter_elevator_copies(ep.building, elevator_id):
            elevator.current_floor = floor_id
