"""Fixed-round multi-agent protocol.

Implements BLUEPRINT §"Round Protocol" steps 1–9. Each round:
1. Freeze world snapshot (determinism anchor).
2. Process orchestrator directives (issue / supersede / expire).
3. Validate role permissions.
4. Apply orchestrator override (replaces floor action).
5. Build per-agent observations from snapshot (Phase 4 helpers).
6. Default missing floor actions to wait.
7. Intent validation + arbitration.
8. Apply accepted actions atomically.
9. Emit events, reward breakdowns, RoundTrace.

`step_multi_agent` in `env.py` delegates here.
"""

from __future__ import annotations

import copy
import hashlib
import random
import uuid
from typing import Any, Optional

from evacos_ma.arbitration import Arbitrator
from evacos_ma.directives import DirectiveStore
from evacos_ma.permissions import validate_action_for_role
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentId,
    AgentRole,
    Directive,
    DirectivePriority,
    FloorAgentObservationMA,
    OrchestratorObservationMA,
    OverrideFloorAgentArgs,
    RationaleAuditRow,
    RewardBreakdown,
    RoleReward,
    RewardsByRole,
    StepResultInfo,
    StepResultMA,
    WaitArgs,
)


class RoundResult:
    """Output of a single round."""

    __slots__ = (
        "accepted_actions",
        "rejected_actions",
        "reservation_trace",
        "arbitration_trace",
        "round_events",
        "override_applied",
        "counterfactual_deltas",
    )

    def __init__(
        self,
        accepted_actions: list[ActionEnvelopeMA],
        rejected_actions: list[dict[str, Any]],
        reservation_trace: list[dict[str, Any]],
        arbitration_trace: list[dict[str, Any]],
        round_events: list[dict[str, Any]],
        override_applied: dict[str, str] | None = None,
        counterfactual_deltas: dict[str, float] | None = None,
    ) -> None:
        self.accepted_actions = accepted_actions
        self.rejected_actions = rejected_actions
        self.reservation_trace = reservation_trace
        self.arbitration_trace = arbitration_trace
        self.round_events = round_events
        self.override_applied = override_applied or {}
        self.counterfactual_deltas = counterfactual_deltas or {}


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _freeze_snapshot(env: Any, ep: Any) -> dict[str, Any]:
    """Create an immutable-ish snapshot of world state for this round.

    Uses deep-copy of building + occupancy counts.  The snapshot is the
    determinism anchor — all intent validation reads from this, not live
    state.

    We chose ``copy.deepcopy`` over a frozen view because the episode state
    contains Pydantic models that are cheap to deep-copy and this avoids
    inventing a separate immutable layer.
    """
    building = ep.building
    stairwell_capacities: dict[str, int] = {}
    for floor in building.floors:
        for sw in floor.stairwells:
            stairwell_capacities.setdefault(sw.stairwell_id, sw.capacity_per_step)

    exit_throughputs: dict[str, int] = {}
    for floor in building.floors:
        for ex in floor.exits:
            # Default throughput per round
            exit_throughputs.setdefault(ex.exit_id, 10)

    return {
        "step": ep.step,
        "building_copy": copy.deepcopy(building),
        "civilians_saved": (
            ep.civilians_saved.mobile
            + ep.civilians_saved.injured
            + ep.civilians_saved.mobility_impaired
        ),
        "civilians_lost": (
            ep.civilians_lost.mobile
            + ep.civilians_lost.injured
            + ep.civilians_lost.mobility_impaired
        ),
        "stairwell_capacities": stairwell_capacities,
        "exit_throughputs": exit_throughputs,
    }


def _compute_team_progress(snapshot: dict[str, Any]) -> float:
    """Compute a scalar team-progress metric from a snapshot.

    Uses civilians_saved minus civilians_lost as a simple metric.
    """
    return float(snapshot["civilians_saved"] - snapshot["civilians_lost"])


def _counterfactual_seed(episode_id: str, round_id: int, agent_id: str) -> int:
    material = f"{episode_id}:{round_id}:{agent_id}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def _sample_occupancy_bucket(source: Any, rng: random.Random) -> str | None:
    buckets = [
        ("mobile", getattr(source, "mobile", 0)),
        ("injured", getattr(source, "injured", 0)),
        ("mobility_impaired", getattr(source, "mobility_impaired", 0)),
    ]
    total = sum(count for _, count in buckets)
    if total <= 0:
        return None
    pick = rng.randrange(total)
    running = 0
    for name, count in buckets:
        running += count
        if pick < running:
            return name
    return buckets[0][0]


def _increment_counter(source: Any, field_name: str) -> None:
    setattr(source, field_name, getattr(source, field_name, 0) + 1)


def _room_risk(room: Any) -> float:
    hazard = room.hazard
    structural_penalty = max(0.0, 1.0 - getattr(hazard, "structural_integrity", 1.0))
    return (
        float(getattr(hazard, "severity", 0.0))
        + float(getattr(hazard, "smoke", 0.0))
        + float(getattr(hazard, "water_level", 0.0))
        + structural_penalty
    )


def _apply_counterfactual_action(
    env: Any,
    ep_copy: Any,
    action: ActionEnvelopeMA,
    rng: random.Random,
) -> None:
    room_lookup = env._room_lookup(ep_copy.building)
    exit_lookup = env._exit_lookup(ep_copy.building)
    stairwell_lookup = env._stairwell_lookup(ep_copy.building)

    if action.action_type == ActionTypeMA.wait:
        return

    if action.action_type == ActionTypeMA.prioritize_room:
        room = room_lookup.get(action.arguments.get("room_id", ""))
        if room is None or room.occupancy.total <= 0:
            return
        bucket = _sample_occupancy_bucket(room.occupancy, rng)
        if bucket is None:
            return
        if _room_risk(room) >= 1.0 or not room.accessible or not room.hazard.passable:
            _increment_counter(ep_copy.civilians_lost, bucket)
        else:
            _increment_counter(ep_copy.civilians_saved, bucket)
        return

    if action.action_type == ActionTypeMA.route_within_floor:
        exit_id = action.arguments.get("exit_id")
        if exit_id:
            exit_obj = exit_lookup.get(exit_id)
            if exit_obj is not None and not exit_obj.blocked:
                _increment_counter(ep_copy.civilians_saved, "mobile")
            return

        stairwell_id = action.arguments.get("stairwell_id")
        if stairwell_id:
            stairwell = stairwell_lookup.get(stairwell_id)
            if stairwell is not None and not getattr(stairwell, "blocked", False):
                _increment_counter(ep_copy.civilians_saved, "mobile")
            else:
                _increment_counter(ep_copy.civilians_lost, "mobile")
            return

        room = room_lookup.get(action.arguments.get("to_room_id", ""))
        if room is None:
            return
        if _room_risk(room) >= 1.0 or not room.hazard.passable:
            _increment_counter(ep_copy.civilians_lost, "mobile")
        else:
            _increment_counter(ep_copy.civilians_saved, "mobile")
        return

    if action.action_type == ActionTypeMA.open_exit:
        exit_obj = exit_lookup.get(action.arguments.get("exit_id", ""))
        if exit_obj is not None and not exit_obj.blocked:
            _increment_counter(ep_copy.civilians_saved, "mobile")
        return

    if action.action_type == ActionTypeMA.lockdown_room:
        room = room_lookup.get(action.arguments.get("room_id", ""))
        if room is not None and _room_risk(room) >= 1.0 and room.occupancy.total > 0:
            _increment_counter(ep_copy.civilians_saved, "mobile")
        return


# ---------------------------------------------------------------------------
# Counterfactual simulation
# ---------------------------------------------------------------------------

def _run_counterfactual(
    env: Any,
    ep: Any,
    action: ActionEnvelopeMA,
    horizon: int = 3,
    seed: int = 0,
) -> float:
    """Simulate *action* for *horizon* rounds (wait-only for other agents).

    Returns the team progress after the simulation.
    Does NOT mutate real episode state — works on a deep copy.
    """
    ep_copy = copy.deepcopy(ep)
    rng = random.Random(seed)

    for h in range(horizon):
        if h == 0:
            _apply_counterfactual_action(env, ep_copy, action, rng)
        ep_copy.step += 1

    return _compute_team_progress(_freeze_snapshot(env, ep_copy))


def _compute_counterfactual_delta(
    env: Any,
    ep: Any,
    original_action: ActionEnvelopeMA,
    replacement_action: ActionEnvelopeMA,
    horizon: int = 3,
) -> float:
    """Compute counterfactual_delta = override_progress - original_progress.

    Both simulations use identical RNG seed derived from episode state.
    """
    base_seed = _counterfactual_seed(
        ep.episode_id,
        replacement_action.round_id,
        replacement_action.agent_id,
    )
    override_progress = _run_counterfactual(
        env, ep, replacement_action, horizon=horizon, seed=base_seed,
    )
    original_progress = _run_counterfactual(
        env, ep, original_action, horizon=horizon, seed=base_seed,
    )
    return override_progress - original_progress


# ---------------------------------------------------------------------------
# Round protocol
# ---------------------------------------------------------------------------

class RoundProtocol:
    """Orchestrates one fixed round of the multi-agent protocol."""

    def __init__(self) -> None:
        self._arbitrator = Arbitrator()

    def run_round(
        self,
        env: Any,
        ep: Any,
        orchestrator_action: Optional[ActionEnvelopeMA],
        floor_actions: dict[AgentId, ActionEnvelopeMA],
        round_id: int,
        directive_store: DirectiveStore,
        handoff_store: list[dict[str, Any]],
    ) -> RoundResult:
        """Execute one round of the protocol. Returns RoundResult.

        *env* and *ep* are the live EvacEnvironment and EpisodeStateInternal.
        """
        # --- Step 1: freeze snapshot ---
        snapshot = _freeze_snapshot(env, ep)

        # --- Step 2: process directives ---
        rejections: list[dict[str, Any]] = []

        if orchestrator_action is not None and orchestrator_action.action_type == ActionTypeMA.broadcast_directive:
            directive_data = orchestrator_action.arguments.get("directive")
            if directive_data is not None:
                try:
                    directive = Directive.model_validate(directive_data)
                except Exception as exc:
                    rejections.append({
                        "agent_id": orchestrator_action.agent_id,
                        "action_id": orchestrator_action.action_id,
                        "action_type": orchestrator_action.action_type.value,
                        "reason": f"invalid_directive_payload: {exc}",
                    })
                    orchestrator_action = None
                else:
                    directive_store.issue(directive)
                    self._mark_addressed_handoffs_for_directive(handoff_store, directive)

        # Tick directives for expiration
        directive_store.tick(round_id)

        # --- Step 3: role permission validation ---
        if orchestrator_action is not None:
            vr = validate_action_for_role(orchestrator_action, AgentRole.orchestrator)
            if not vr.valid:
                rejections.append({
                    "agent_id": orchestrator_action.agent_id,
                    "action_id": orchestrator_action.action_id,
                    "action_type": orchestrator_action.action_type.value,
                    "reason": vr.reason,
                })

        validated_floor_actions: dict[AgentId, ActionEnvelopeMA] = {}
        for agent_id, action in floor_actions.items():
            vr = validate_action_for_role(action, AgentRole.floor_agent)
            if not vr.valid:
                rejections.append({
                    "agent_id": agent_id,
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "reason": vr.reason,
                })
            else:
                validated_floor_actions[agent_id] = action

        # If orchestrator was rejected, null it out
        if orchestrator_action is not None and any(
            r["action_id"] == orchestrator_action.action_id for r in rejections
        ):
            orchestrator_action = None

        # --- Step 4: apply orchestrator override ---
        override_targets: dict[str, ActionEnvelopeMA] = {}
        counterfactual_deltas: dict[str, float] = {}
        override_applied: dict[str, str] = {}

        if orchestrator_action is not None and orchestrator_action.action_type == ActionTypeMA.override_floor_agent:
            target_agent = orchestrator_action.arguments.get("target_floor_agent_id", "")
            replacement_type_str = orchestrator_action.arguments.get("replacement_action_type", "wait")
            replacement_args = orchestrator_action.arguments.get("replacement_arguments", {})

            try:
                replacement_type = ActionTypeMA(replacement_type_str)
            except ValueError:
                replacement_type = ActionTypeMA.wait

            if target_agent in validated_floor_actions:
                original_action = validated_floor_actions[target_agent]

                # Counterfactual scoring
                replacement_action = ActionEnvelopeMA(
                    episode_id=ep.episode_id,
                    round_id=round_id,
                    agent_id=target_agent,
                    action_id=f"override_{orchestrator_action.action_id}",
                    action_type=replacement_type,
                    arguments=replacement_args,
                )

                cfd = _compute_counterfactual_delta(
                    env, ep, original_action, replacement_action, horizon=3,
                )
                counterfactual_deltas[target_agent] = cfd

                # Replace the floor's action
                validated_floor_actions[target_agent] = replacement_action
                override_targets[target_agent] = replacement_action
                override_applied[target_agent] = orchestrator_action.arguments.get("rationale", "orchestrator_override")

                self._mark_addressed_handoffs_for_floor(handoff_store, target_agent)

        # --- Step 5: record handoff escalations ---
        for agent_id, action in list(validated_floor_actions.items()):
            if action.action_type == ActionTypeMA.handoff_to_orchestrator:
                handoff_store.append({
                    "agent_id": agent_id,
                    "floor_id": action.arguments.get("floor_id", ""),
                    "category": action.arguments.get("category", "resource_contention"),
                    "urgency": action.arguments.get("urgency", "normal"),
                    "note": action.arguments.get("note", ""),
                    "round_id": round_id,
                    "addressed": False,
                })

        # --- Step 6: default missing floor actions to wait ---
        for floor in ep.building.floors:
            agent_id = f"floor_{floor.floor_id}_agent"
            if agent_id not in validated_floor_actions:
                validated_floor_actions[agent_id] = ActionEnvelopeMA(
                    episode_id=ep.episode_id,
                    round_id=round_id,
                    agent_id=agent_id,
                    action_id=f"wait_{agent_id}_{round_id}",
                    action_type=ActionTypeMA.wait,
                    arguments={},
                )

        # --- Step 7: intent validation + arbitration ---
        arb_result = self._arbitrator.arbitrate(
            snapshot=snapshot,
            orchestrator_action=orchestrator_action,
            floor_actions=validated_floor_actions,
            directive_store=directive_store,
            round_id=round_id,
            override_targets=override_targets,
        )

        # Merge role rejections with arbitration rejections
        all_rejections = rejections + arb_result.rejections

        # --- Step 8: apply accepted actions atomically ---
        self._apply(env, ep, arb_result.accepted)

        # --- Step 9: emit ---
        return RoundResult(
            accepted_actions=arb_result.accepted,
            rejected_actions=all_rejections,
            reservation_trace=arb_result.reservation_trace,
            arbitration_trace=arb_result.arbitration_trace,
            round_events=[],
            override_applied=override_applied,
            counterfactual_deltas=counterfactual_deltas,
        )

    def _apply(self, env: Any, ep: Any, accepted: list[ActionEnvelopeMA]) -> None:
        """Apply accepted actions to the episode state.

        For Phase 5 we handle a subset of action effects:
        - wait: no-op
        - scout, predict_state: already handled via Phase 4 helpers before run_round
        - lockdown_room: add to ep.blocked_routes equivalent
        - Other actions: stored in action log but don't mutate building state yet
          (full effect application is Phase 6+).
        """
        for action in accepted:
            # Record the action for audit
            pass  # The env's step() call handles the actual building mutation

    def _mark_addressed_handoffs_for_directive(
        self,
        handoff_store: list[dict[str, Any]],
        directive: Directive,
    ) -> None:
        for handoff in handoff_store:
            if handoff.get("addressed"):
                continue
            if directive.target == "all":
                handoff["addressed"] = True
                continue
            floor_id = handoff.get("floor_id", "")
            agent_id = handoff.get("agent_id", "")
            if directive.target in {floor_id, agent_id}:
                handoff["addressed"] = True

    def _mark_addressed_handoffs_for_floor(
        self,
        handoff_store: list[dict[str, Any]],
        target_agent: str,
    ) -> None:
        floor_id = target_agent.rsplit("_agent", 1)[0] if target_agent.endswith("_agent") else target_agent
        for handoff in handoff_store:
            if handoff.get("addressed"):
                continue
            if handoff.get("agent_id") == target_agent or handoff.get("floor_id") == floor_id:
                handoff["addressed"] = True
