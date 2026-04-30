"""Multi-role rollout collector."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from curriculum.controller import EVAL_SEEDS
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
    FloorAgentObservationMA,
)
from evacos_ma.schemas.multi_agent import TRACE_SCHEMA_VERSION as _TRACE_V
from evacos_ma.schemas.rewards import REWARD_SCHEMA_VERSION as _REWARD_V

from training.metrics import write_trace_row
from training.policy_adapter import Policy, StubPolicy, parse_completion_to_action
from training.prompts import (
    PROMPT_TEMPLATE_VERSION as _PROMPT_TEMPLATE_VERSION,
    build_floor_prompt,
    build_orchestrator_prompt,
)
from training.reward import RewardNormalizer, normalize_per_role

PROMPT_TEMPLATE_VERSION = _PROMPT_TEMPLATE_VERSION
logger = logging.getLogger(__name__)
_ROUND_TRACE_WARNED_FIELDS: set[str] = set()
_DEFAULT_REWARD_CONFIG: dict[str, object] = {
    "rationale_scaling": "linear_capped",
    "alpha": 0.01,
    "beta": 0.25,
    "cap": 1.0,
    "eligible_token_ceiling": 160,
    "clip_normalized_to": 1.0,
}
_PRIORITY_REWARD_KEYS: tuple[str, ...] = (
    "priority_top_match",
    "priority_rank_score",
    "priority_coverage",
    "priority_duplicate_or_unknown_penalty",
    "priority_effect_bonus",
    "priority_unchanged_penalty",
)


@dataclass
class TrajectorySample:
    episode_id: str
    round_id: int
    agent_id: str
    role: str
    seed: int
    tier: str
    disaster_family: str
    generator_config_hash: str
    prompt: list[dict[str, str]]
    completion_text: str
    parsed_action: dict
    raw_reward: float
    normalized_reward: float
    done: bool
    checkpoint_tag: str
    group_id: str
    trace_schema_version: str
    reward_schema_version: str
    prompt_template_version: str
    model_name: str
    completion_token_ids: list[int] | None = None


@dataclass
class EpisodeRolloutResult:
    episode_id: str
    seed: int
    tier: str
    disaster_family: str
    generator_config_hash: str
    samples: list[TrajectorySample]
    total_raw_reward_by_role: dict[str, float]
    total_normalized_reward_by_role: dict[str, float]
    done_reason: str | None
    num_rounds: int
    wall_clock_seconds: float
    rationale_bonus_total: float = 0.0
    rationale_bonus_count: int = 0
    priority_component_totals: dict[str, float] = field(default_factory=dict)
    priority_component_counts: dict[str, int] = field(default_factory=dict)
    priority_behavior_totals: dict[str, float] = field(default_factory=dict)
    priority_behavior_counts: dict[str, int] = field(default_factory=dict)
    priority_directive_issue_count: int = 0


def _extract_tier_str(tier_obj: object) -> str:
    if isinstance(tier_obj, str):
        return tier_obj
    return str(getattr(tier_obj, "value", tier_obj))


def _extract_disaster_family_str(disaster_family: object) -> str:
    value = getattr(disaster_family, "value", None)
    if value is not None:
        return str(value)
    text = str(disaster_family)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _fallback_wait_action(episode_id: str, round_id: int, agent_id: str) -> ActionEnvelopeMA:
    return ActionEnvelopeMA(
        episode_id=episode_id,
        round_id=round_id,
        agent_id=agent_id,
        action_id=uuid.uuid4().hex[:8],
        action_type=ActionTypeMA.wait,
        arguments={},
    )


def _trace_common(
    *,
    episode_id: str,
    round_id: int,
    seed: int,
    tier: str,
    disaster_family: str,
    generator_config_hash: str,
    checkpoint_tag: str,
    model_name: str,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "round_id": round_id,
        "seed": seed,
        "tier": tier,
        "disaster_family": disaster_family,
        "trace_schema_version": _TRACE_V,
        "generator_config_hash": generator_config_hash,
        "reward_schema_version": _REWARD_V,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "model_name": model_name,
        "checkpoint_tag": checkpoint_tag,
    }


def _runtime_reward_config(
    reward_config: Mapping[str, object] | None,
    rationale_mode: str | None,
) -> dict[str, object]:
    payload = dict(_DEFAULT_REWARD_CONFIG)
    if reward_config is not None:
        payload.update(dict(reward_config))
    if rationale_mode is not None:
        payload["rationale_scaling"] = rationale_mode
    return payload


def _coerce_policy_result(result: object) -> tuple[str, list[int]]:
    if isinstance(result, tuple) and len(result) == 2:
        text, token_ids = result
        text_str = text if isinstance(text, str) else str(text)
        if isinstance(token_ids, list):
            return text_str, [int(token_id) for token_id in token_ids]
        return text_str, []
    if isinstance(result, str):
        return result, []
    return str(result), []


def _prompt_scoped_group_id(
    *,
    episode_id: str,
    round_id: int,
    role: str,
    agent_id: str,
    prompt: list[dict[str, str]],
) -> str:
    """Stable group id for multiple completions sampled from one prompt."""

    prompt_blob = json.dumps(
        prompt,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{episode_id}:{round_id}:{role}:{agent_id}:{prompt_blob}",
    ).hex[:16]
    return f"prompt_{digest}"


def _action_type_value(action: ActionEnvelopeMA) -> str:
    return str(getattr(action.action_type, "value", action.action_type))


def _bounded_reward(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _floor_candidate_reward(
    floor_obs: FloorAgentObservationMA,
    action: ActionEnvelopeMA | None,
    parse_status: str,
) -> float:
    """Cheap verifier reward for same-prompt candidate ranking.

    Only the first candidate is executed in the environment. Extra candidates
    still need a reward signal for GRPO, so this verifier scores whether the
    proposed action is parse-valid, allowed, references visible entities, and
    moves known civilians toward exits/stairs instead of waiting or scouting.
    """

    if action is None or parse_status != "ok":
        return -1.0

    action_type = _action_type_value(action)
    allowed_actions = {
        str(getattr(item, "value", item)) for item in (floor_obs.action_mask or [])
    }
    if allowed_actions and action_type not in allowed_actions:
        return -0.9

    args = action.arguments if isinstance(action.arguments, dict) else {}
    visible_rooms = {room.room_id for room in (floor_obs.visible_rooms or [])}
    civilian_rooms = {
        group.location_room_id
        for group in (floor_obs.visible_civilian_groups or [])
        if int(getattr(group, "count", 0) or 0) > 0
    }
    unsafe_rooms = {
        room.room_id
        for room in (floor_obs.visible_rooms or [])
        if float(getattr(room, "hazard_severity", 0.0) or 0.0) > 0.0
        or float(getattr(room, "smoke_level", 0.0) or 0.0) > 0.0
    }
    unsafe_rooms.update(
        hazard.room_id
        for hazard in (floor_obs.local_hazards or [])
        if getattr(hazard, "room_id", None)
    )
    unsafe_corridor_nodes: set[str] = set()
    for corridor in (floor_obs.visible_corridors or []):
        if (
            not bool(getattr(corridor, "passable", True))
            or float(getattr(corridor, "hazard_severity", 0.0) or 0.0) > 0.0
        ):
            unsafe_corridor_nodes.add(str(getattr(corridor, "from_node_id", "")))
            unsafe_corridor_nodes.add(str(getattr(corridor, "to_node_id", "")))
    exit_by_id = {exit_view.exit_id: exit_view for exit_view in (floor_obs.exits_on_floor or [])}
    stairwell_ids = {
        stairwell.stairwell_id
        for stairwell in (floor_obs.stairwell_entries or [])
        if not bool(getattr(stairwell, "blocked", False))
    }
    has_available_egress = bool(
        any(not bool(getattr(exit_view, "blocked", False)) for exit_view in exit_by_id.values())
        or stairwell_ids
    )
    has_known_people = bool(civilian_rooms)

    if action_type == ActionTypeMA.route_within_floor.value:
        from_room = args.get("from_room_id")
        to_room = args.get("to_room_id")
        exit_id = args.get("exit_id")
        stairwell_id = args.get("stairwell_id")
        target_valid = (
            (to_room in visible_rooms if to_room else False)
            or (
                exit_id in exit_by_id
                and not bool(getattr(exit_by_id[exit_id], "blocked", False))
                if exit_id
                else False
            )
            or (stairwell_id in stairwell_ids if stairwell_id else False)
        )
        if not target_valid:
            return -0.95
        if from_room and from_room not in visible_rooms:
            return -0.85
        if to_room and has_known_people and has_available_egress:
            score = 0.05
            if from_room in civilian_rooms:
                score += 0.1
            if from_room in unsafe_rooms and to_room not in unsafe_rooms:
                score += 0.1
            if to_room in unsafe_corridor_nodes:
                score -= 0.2
            return _bounded_reward(score)
        score = 0.35
        if from_room in civilian_rooms:
            score += 0.35
        if exit_id or stairwell_id:
            score += 0.35
        if from_room in unsafe_rooms:
            score += 0.1
        if from_room in unsafe_corridor_nodes:
            score += 0.05
        return _bounded_reward(score)

    if action_type == ActionTypeMA.prioritize_room.value:
        room_id = args.get("room_id")
        if room_id not in visible_rooms:
            return -0.45
        score = 0.25
        if room_id in civilian_rooms:
            score += 0.35
        if room_id in unsafe_rooms:
            score += 0.15
        return _bounded_reward(score)

    if action_type == ActionTypeMA.open_exit.value:
        exit_id = args.get("exit_id")
        exit_view = exit_by_id.get(exit_id)
        if exit_view is None:
            return -0.5
        if bool(getattr(exit_view, "blocked", False)):
            return -0.4
        if bool(getattr(exit_view, "requires_open_action", False)):
            return 0.65
        return -0.8 if has_known_people and has_available_egress else -0.2

    if action_type == ActionTypeMA.scout.value:
        target_room = args.get("target_room_id")
        if target_room and target_room not in visible_rooms:
            return -0.25
        return -0.15 if has_known_people else 0.15

    if action_type == ActionTypeMA.lockdown_room.value:
        room_id = args.get("room_id")
        if room_id not in visible_rooms:
            return -0.45
        if room_id in civilian_rooms:
            return -0.4
        return 0.2 if room_id in unsafe_rooms else -0.05

    if action_type == ActionTypeMA.handoff_to_orchestrator.value:
        return 0.05 if has_known_people or unsafe_rooms else -0.05

    if action_type == ActionTypeMA.predict_state.value:
        return 0.0

    if action_type == ActionTypeMA.wait.value:
        return -0.45 if has_known_people or unsafe_rooms else -0.05

    return -0.2


def _select_floor_candidate(
    floor_obs: FloorAgentObservationMA,
    candidates: Sequence[ActionEnvelopeMA | None],
    parse_statuses: Sequence[str],
) -> tuple[int, ActionEnvelopeMA | None, str]:
    """Pick the most useful parsed candidate for environment execution.

    GRPO samples multiple completions for the same floor prompt. Executing the
    first sample wastes that signal when later candidates contain the useful
    route/open action, so use the same cheap verifier that scores non-executed
    candidates to choose the environment action.
    """

    if not candidates:
        return 0, None, "missing_candidate"

    best_index = 0
    best_score = float("-inf")
    for index, (candidate, parse_status) in enumerate(zip(candidates, parse_statuses, strict=False)):
        score = _floor_candidate_reward(floor_obs, candidate, parse_status)
        if score > best_score:
            best_index = index
            best_score = score

    selected_action = candidates[best_index]
    selected_status = parse_statuses[best_index] if best_index < len(parse_statuses) else "missing_status"
    return best_index, selected_action, selected_status


def _floor_oracle_candidate_payload(floor_obs: FloorAgentObservationMA) -> tuple[str, list[int]]:
    """Build one exact-ID expert floor action for bootstrap candidate groups."""

    allowed = {str(getattr(item, "value", item)) for item in (floor_obs.action_mask or [])}
    visible_rooms = [room.room_id for room in (floor_obs.visible_rooms or [])]
    civilian_rooms = [
        group.location_room_id
        for group in (floor_obs.visible_civilian_groups or [])
        if int(getattr(group, "count", 0) or 0) > 0 and group.location_room_id
    ]
    from_room = civilian_rooms[0] if civilian_rooms else (visible_rooms[0] if visible_rooms else None)
    base: dict[str, object] = {
        "episode_id": floor_obs.episode_id,
        "round_id": floor_obs.round_id,
        "agent_id": floor_obs.agent_id,
    }

    if "route_within_floor" in allowed and from_room:
        exit_view = next(
            (item for item in (floor_obs.exits_on_floor or []) if not bool(getattr(item, "blocked", False))),
            None,
        )
        if exit_view is not None:
            return (
                json.dumps(
                    {
                        **base,
                        "action_id": "oracle_route_exit",
                        "action_type": ActionTypeMA.route_within_floor.value,
                        "arguments": {"from_room_id": from_room, "exit_id": exit_view.exit_id},
                    },
                    separators=(",", ":"),
                ),
                [],
            )
        stair = next(
            (
                item
                for item in (floor_obs.stairwell_entries or [])
                if not bool(getattr(item, "blocked", False))
            ),
            None,
        )
        if stair is not None:
            return (
                json.dumps(
                    {
                        **base,
                        "action_id": "oracle_route_stair",
                        "action_type": ActionTypeMA.route_within_floor.value,
                        "arguments": {"from_room_id": from_room, "stairwell_id": stair.stairwell_id},
                    },
                    separators=(",", ":"),
                ),
                [],
            )

    if "open_exit" in allowed:
        exit_view = next(
            (
                item
                for item in (floor_obs.exits_on_floor or [])
                if not bool(getattr(item, "blocked", False))
                and bool(getattr(item, "requires_open_action", False))
            ),
            None,
        )
        if exit_view is not None:
            return (
                json.dumps(
                    {
                        **base,
                        "action_id": "oracle_open_exit",
                        "action_type": ActionTypeMA.open_exit.value,
                        "arguments": {"exit_id": exit_view.exit_id},
                    },
                    separators=(",", ":"),
                ),
                [],
            )

    if "scout" in allowed and visible_rooms:
        return (
            json.dumps(
                {
                    **base,
                    "action_id": "oracle_scout",
                    "action_type": ActionTypeMA.scout.value,
                    "arguments": {"target_room_id": visible_rooms[0]},
                },
                separators=(",", ":"),
            ),
            [],
        )

    return (
        json.dumps(
            {
                **base,
                "action_id": "oracle_wait",
                "action_type": ActionTypeMA.wait.value,
                "arguments": {},
            },
            separators=(",", ":"),
        ),
        [],
    )


def _rejection_reason_by_action_id(result: object) -> dict[str, str]:
    rejected: dict[str, str] = {}
    for row in getattr(result, "invalid_actions", []) or []:
        if not isinstance(row, dict):
            continue
        action_id = str(row.get("action_id") or "")
        if not action_id:
            continue
        rejected[action_id] = str(row.get("reason") or "env_rejected")
    return rejected


def _mark_action_rejection(action: ActionEnvelopeMA, reason: str | None) -> None:
    if not reason:
        return
    action.fallback_reason = "env_rejected"
    action.rejection_reason = reason


def _candidate_parsed_action(
    action: ActionEnvelopeMA | None,
    parse_status: str,
    candidate_index: int,
    *,
    selected_for_execution: bool = False,
) -> dict:
    if action is None:
        return {
            "fallback_reason": "parse_error",
            "parse_status": parse_status,
            "candidate_index": candidate_index,
            "selected_for_execution": selected_for_execution,
        }
    parsed = action.model_dump(mode="json")
    parsed["parse_status"] = parse_status
    parsed["candidate_index"] = candidate_index
    parsed["selected_for_execution"] = selected_for_execution
    if selected_for_execution and getattr(action, "fallback_reason", None):
        parsed["fallback_reason"] = getattr(action, "fallback_reason")
        parsed["rejection_reason"] = getattr(action, "rejection_reason", None)
    return parsed


def _warn_missing_round_payload_field_once(field_name: str, reason: str) -> None:
    if field_name in _ROUND_TRACE_WARNED_FIELDS:
        return
    _ROUND_TRACE_WARNED_FIELDS.add(field_name)
    logger.warning("round_trace missing authoritative %s; emitting empty default (%s)", field_name, reason)


def _build_round_state_payload(
    env: EvacEnvironment,
    episode_id: str,
) -> tuple[dict[str, int], dict[str, float]]:
    try:
        building = env.get_internal_state(episode_id).building
    except Exception as exc:  # pragma: no cover - defensive fallback
        reason = f"state_unavailable: {exc}"
        _warn_missing_round_payload_field_once("per_floor_civilians", reason)
        _warn_missing_round_payload_field_once("per_floor_hazard_severity", reason)
        return {}, {}

    per_floor_civilians: dict[str, int] = {}
    per_floor_hazard_severity: dict[str, float] = {}
    for floor in building.floors:
        floor_key = f"floor_{floor.floor_id}"
        per_floor_civilians[floor_key] = sum(room.occupancy.total for room in floor.rooms)
        hazard_values = [
            float(room.hazard.severity)
            for room in floor.rooms
            if room.hazard.hazard_type is not None or room.hazard.severity > 0.0
        ]
        per_floor_hazard_severity[floor_key] = (
            sum(hazard_values) / len(hazard_values) if hazard_values else 0.0
        )
    return per_floor_civilians, per_floor_hazard_severity


def _build_round_action_feed(
    orchestrator_action: ActionEnvelopeMA,
    *,
    expected_action_type: ActionTypeMA,
    round_id: int,
) -> list[dict[str, object]]:
    if orchestrator_action.action_type != expected_action_type:
        return []
    return [
        {
            "agent_id": "orchestrator",
            "action_id": orchestrator_action.action_id,
            "action_type": expected_action_type.value,
            "arguments": orchestrator_action.arguments,
            "round_id": round_id,
        }
    ]


def _emit_round_artifacts(
    *,
    jsonl_dir: Path,
    env: EvacEnvironment,
    episode_id: str,
    common: dict[str, object],
    round_events: list[dict],
    orchestrator_action: ActionEnvelopeMA,
    floor_actions: dict[str, ActionEnvelopeMA],
    reward_rows: list[tuple[str, float, float, dict]],
    belief_rows: list[dict],
    rationale_rows: list[dict],
    completion_rows: dict[str, dict[str, object]],
) -> None:
    per_floor_civilians, per_floor_hazard_severity = _build_round_state_payload(env, episode_id)
    round_id = int(common.get("round_id", 0))
    write_trace_row(
        jsonl_dir / "round_trace.jsonl",
        {
            **common,
            "round_events": round_events,
            "orchestrator_action_type": orchestrator_action.action_type.value,
            "floor_action_types": {
                agent_id: action.action_type.value
                for agent_id, action in sorted(floor_actions.items())
            },
            "per_floor_civilians": per_floor_civilians,
            "per_floor_hazard_severity": per_floor_hazard_severity,
            "directive_feed": _build_round_action_feed(
                orchestrator_action,
                expected_action_type=ActionTypeMA.broadcast_directive,
                round_id=round_id,
            ),
            "override_feed": _build_round_action_feed(
                orchestrator_action,
                expected_action_type=ActionTypeMA.override_floor_agent,
                round_id=round_id,
            ),
            "reward_ticker": {
                agent_id: normalized_reward
                for agent_id, _raw_reward, normalized_reward, _breakdown in reward_rows
            },
        },
    )

    for agent_id, action in [("orchestrator", orchestrator_action), *sorted(floor_actions.items())]:
        parsed = action.model_dump(mode="json")
        valid = "fallback_reason" not in parsed
        completion_row = completion_rows.get(agent_id, {})
        write_trace_row(
            jsonl_dir / "action_trace.jsonl",
            {
                **common,
                "agent_id": agent_id,
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "arguments": action.arguments,
                "valid": valid,
                "rejection_reason": parsed.get("fallback_reason"),
                "parse_status": completion_row.get("parse_status"),
                "completion_text": completion_row.get("completion_text"),
            },
        )

    for agent_id, raw_reward, normalized_reward, breakdown in reward_rows:
        write_trace_row(
            jsonl_dir / "reward_trace.jsonl",
            {
                **common,
                "agent_id": agent_id,
                "raw_reward": raw_reward,
                "normalized_reward": normalized_reward,
                "breakdown": breakdown,
            },
        )

    for belief_row in belief_rows:
        write_trace_row(jsonl_dir / "belief_audit.jsonl", {**common, **belief_row})

    for rationale_row in rationale_rows:
        write_trace_row(jsonl_dir / "rationale_audit.jsonl", {**common, **rationale_row})


def collect_episode(
    env: EvacEnvironment,
    policy: Policy,
    *,
    seed: int,
    tier: str,
    disaster_family: DisasterType,
    max_rounds: int = 80,
    checkpoint_tag: str = "baseline",
    model_name: str = "stub",
    normalizer: RewardNormalizer | None = None,
    update_normalizer: bool = True,
    jsonl_dir: Path | None = None,
    cleanup_env_episode: bool = False,
    rationale_mode: str | None = None,
    reward_config: Mapping[str, object] | None = None,
    candidates_per_floor_prompt: int = 1,
    include_oracle_floor_candidate: bool = False,
) -> EpisodeRolloutResult:
    if normalizer is None:
        normalizer = RewardNormalizer()
    if jsonl_dir is None:
        jsonl_dir = Path("outputs/logs")

    started = time.monotonic()
    disaster_family_str = _extract_disaster_family_str(disaster_family)
    episode_id: str | None = None
    try:
        reward_runtime = _runtime_reward_config(reward_config, rationale_mode)
        episode_id, obs_by_role = env.reset_multi_agent(
            task_id=f"procgen_{tier}_{disaster_family_str}",
            seed=seed,
            procgen_tier=tier,
            procgen_disaster_family=disaster_family,
            procgen_max_steps=max_rounds,
        )
        state = env.get_internal_state(episode_id)
        state.rollout_metadata["rationale_mode"] = str(
            reward_runtime.get("rationale_scaling", "linear_capped")
        )
        state.rollout_metadata["reward_config"] = dict(reward_runtime)

        generator_config_hash = obs_by_role.orchestrator.generator_config_hash
        samples: list[TrajectorySample] = []
        total_raw: dict[str, float] = {"orchestrator": 0.0}
        total_norm: dict[str, float] = {"orchestrator": 0.0}
        rationale_bonus_total = 0.0
        rationale_bonus_count = 0
        priority_component_totals = {key: 0.0 for key in _PRIORITY_REWARD_KEYS}
        priority_component_counts = {key: 0 for key in _PRIORITY_REWARD_KEYS}
        priority_behavior_totals = {
            "priority_top_match_rate": 0.0,
            "priority_rank_fraction_mean": 0.0,
            "priority_coverage_fraction_mean": 0.0,
            "priority_effect_bonus_rate": 0.0,
            "priority_unchanged_rate": 0.0,
        }
        priority_behavior_counts = {key: 0 for key in priority_behavior_totals}
        priority_directive_issue_count = 0
        previous_priority_order: tuple[str, ...] | None = None
        for agent_id in obs_by_role.floors:
            total_raw[agent_id] = 0.0
            total_norm[agent_id] = 0.0

        round_id = 0
        done = False
        done_reason: str | None = None

        while not done and round_id < max_rounds:
            orch_prompt = build_orchestrator_prompt(obs_by_role.orchestrator)
            floor_prompts: dict[str, list[dict[str, str]]] = {}
            for agent_id, floor_obs in obs_by_role.floors.items():
                floor_prompts[agent_id] = build_floor_prompt(floor_obs)
            floor_ids = list(floor_prompts.keys())
            candidate_count = max(1, int(candidates_per_floor_prompt))

            # Batched fast path: policies that expose act_batch collapse the 6
            # per-round calls (1 orchestrator + 5 floors) into a single generate
            # call. Detection is by hasattr, so StubPolicy / hf_policy_factory
            # fall back to the per-agent loop transparently.
            floor_candidate_payloads: dict[str, list[tuple[str, list[int]]]] = {}
            if hasattr(policy, "act_batch"):
                batch_prompts: list[list[dict[str, str]]] = [orch_prompt]
                batch_agent_ids: list[str] = ["orchestrator"]
                batch_roles: list[str] = ["orchestrator"]
                for aid in floor_ids:
                    for _ in range(candidate_count):
                        batch_prompts.append(floor_prompts[aid])
                        batch_agent_ids.append(aid)
                        batch_roles.append("floor_agent")
                completions = policy.act_batch(batch_prompts, batch_agent_ids, batch_roles)
                batch_results = [_coerce_policy_result(item) for item in completions]
                orch_completion, orch_completion_ids = batch_results[0]
                cursor = 1
                for aid in floor_ids:
                    payloads = batch_results[cursor : cursor + candidate_count]
                    cursor += candidate_count
                    if include_oracle_floor_candidate:
                        payloads = [
                            *payloads,
                            _floor_oracle_candidate_payload(obs_by_role.floors[aid]),
                        ]
                    floor_candidate_payloads[aid] = payloads
                floor_completions = {
                    aid: payloads[0][0]
                    for aid, payloads in floor_candidate_payloads.items()
                }
                floor_completion_ids = {
                    aid: payloads[0][1]
                    for aid, payloads in floor_candidate_payloads.items()
                }
            else:
                orch_completion, orch_completion_ids = _coerce_policy_result(
                    policy.act(orch_prompt, "orchestrator", "orchestrator")
                )
                for aid in floor_ids:
                    floor_candidate_payloads[aid] = [
                        _coerce_policy_result(
                            policy.act(floor_prompts[aid], aid, "floor_agent")
                        )
                        for _ in range(candidate_count)
                    ]
                    if include_oracle_floor_candidate:
                        floor_candidate_payloads[aid].append(
                            _floor_oracle_candidate_payload(obs_by_role.floors[aid])
                        )
                floor_completions = {
                    aid: payloads[0][0]
                    for aid, payloads in floor_candidate_payloads.items()
                }
                floor_completion_ids = {
                    aid: payloads[0][1]
                    for aid, payloads in floor_candidate_payloads.items()
                }

            orch_parse_status = "ok"
            orch_action, orch_parse_status = parse_completion_to_action(
                orch_completion,
                "orchestrator",
                "orchestrator",
                obs_by_role.orchestrator.episode_id,
                obs_by_role.orchestrator.round_id,
            )
            if orch_action is None:
                orch_action = _fallback_wait_action(episode_id, round_id, "orchestrator")
                orch_action.fallback_reason = "parse_error"

            floor_actions: dict[str, ActionEnvelopeMA] = {}
            floor_payloads: dict[str, tuple[str, list[dict[str, str]]]] = {}
            floor_candidate_actions: dict[str, list[ActionEnvelopeMA | None]] = {}
            floor_candidate_parse_statuses: dict[str, list[str]] = {}
            floor_selected_candidate_indices: dict[str, int] = {}
            floor_parse_statuses: dict[str, str] = {}
            for agent_id in floor_ids:
                floor_prompt = floor_prompts[agent_id]
                parsed_candidates: list[ActionEnvelopeMA | None] = []
                parsed_statuses: list[str] = []
                for floor_completion, _token_ids in floor_candidate_payloads[agent_id]:
                    floor_action, floor_parse_status = parse_completion_to_action(
                        floor_completion,
                        agent_id,
                        "floor_agent",
                        obs_by_role.floors[agent_id].episode_id,
                        obs_by_role.floors[agent_id].round_id,
                    )
                    parsed_candidates.append(floor_action)
                    parsed_statuses.append(floor_parse_status)
                selected_index, floor_action, floor_parse_status = _select_floor_candidate(
                    obs_by_role.floors[agent_id],
                    parsed_candidates,
                    parsed_statuses,
                )
                if floor_action is None:
                    floor_action = _fallback_wait_action(episode_id, round_id, agent_id)
                    floor_action.fallback_reason = "parse_error"
                floor_actions[agent_id] = floor_action
                selected_payload = floor_candidate_payloads[agent_id][selected_index]
                floor_payloads[agent_id] = (selected_payload[0], floor_prompt)
                floor_candidate_actions[agent_id] = parsed_candidates
                floor_candidate_parse_statuses[agent_id] = parsed_statuses
                floor_selected_candidate_indices[agent_id] = selected_index
                floor_parse_statuses[agent_id] = floor_parse_status

            result = env.step_multi_agent(
                ActionBundleMA(
                    episode_id=episode_id,
                    round_id=round_id,
                    orchestrator_action=orch_action,
                    floor_actions=floor_actions,
                )
            )
            done = result.done
            done_reason = result.done_reason if done else None
            rejected_by_action_id = _rejection_reason_by_action_id(result)
            _mark_action_rejection(
                orch_action,
                rejected_by_action_id.get(orch_action.action_id),
            )
            for floor_action in floor_actions.values():
                _mark_action_rejection(
                    floor_action,
                    rejected_by_action_id.get(floor_action.action_id),
                )

            norm_rewards = normalize_per_role(
                result.rewards_by_role,
                tier,
                normalizer,
                update=update_normalizer,
                clip=float(reward_runtime.get("clip_normalized_to", 1.0)),
            )

            common = _trace_common(
                episode_id=episode_id,
                round_id=round_id,
                seed=seed,
                tier=tier,
                disaster_family=disaster_family_str,
                generator_config_hash=generator_config_hash,
                checkpoint_tag=checkpoint_tag,
                model_name=model_name,
            )
            reward_rows: list[tuple[str, float, float, dict]] = []
            belief_rows = list(result.info.score_snapshot.get("belief_audits", []))
            rationale_rows: list[dict] = []

            orch_parsed = orch_action.model_dump(mode="json")
            orch_raw = result.rewards_by_role.orchestrator.raw
            orch_norm = norm_rewards.get("orchestrator", 0.0)
            orch_breakdown = result.rewards_by_role.orchestrator.breakdown.get_components()
            total_raw["orchestrator"] += orch_raw
            total_norm["orchestrator"] += orch_norm
            reward_rows.append(
                (
                    "orchestrator",
                    orch_raw,
                    orch_norm,
                    orch_breakdown,
                )
            )
            for key in _PRIORITY_REWARD_KEYS:
                if key in orch_breakdown:
                    priority_component_totals[key] += float(orch_breakdown.get(key, 0.0))
                    priority_component_counts[key] += 1
            priority_snapshot = result.info.score_snapshot.get("priority", {})
            if isinstance(priority_snapshot, dict):
                priority_directive_issue_count += int(
                    priority_snapshot.get("priority_directive_issued_count", 0) or 0
                )
                if priority_snapshot.get("priority_action"):
                    used_order = tuple(str(floor_id) for floor_id in priority_snapshot.get("priority_order_used", []))
                    oracle_order = [
                        str(floor_id)
                        for floor_id in priority_snapshot.get("priority_oracle_order", [])
                    ]
                    top_match = 0.0
                    if used_order and oracle_order and used_order[0] == oracle_order[0]:
                        top_match = 1.0
                    priority_behavior_totals["priority_top_match_rate"] += top_match
                    priority_behavior_counts["priority_top_match_rate"] += 1

                    priority_behavior_totals["priority_rank_fraction_mean"] += float(
                        priority_snapshot.get("priority_rank_fraction", 0.0) or 0.0
                    )
                    priority_behavior_counts["priority_rank_fraction_mean"] += 1
                    priority_behavior_totals["priority_coverage_fraction_mean"] += float(
                        priority_snapshot.get("priority_coverage_fraction", 0.0) or 0.0
                    )
                    priority_behavior_counts["priority_coverage_fraction_mean"] += 1
                    priority_behavior_totals["priority_effect_bonus_rate"] += (
                        1.0 if priority_snapshot.get("priority_effect_bonus_applied") else 0.0
                    )
                    priority_behavior_counts["priority_effect_bonus_rate"] += 1
                    if previous_priority_order is not None:
                        priority_behavior_totals["priority_unchanged_rate"] += (
                            1.0 if used_order == previous_priority_order else 0.0
                        )
                        priority_behavior_counts["priority_unchanged_rate"] += 1
                    previous_priority_order = used_order
            orch_bonus = float(orch_breakdown.get("rationale_bonus", 0.0))
            if orch_bonus > 0.0:
                rationale_bonus_total += orch_bonus
                rationale_bonus_count += 1
            orch_rationale = orch_parsed.get("rationale")
            if orch_rationale:
                # Resolve counterfactual_delta by the override target, not by "orchestrator" (H1).
                _cf_deltas = result.info.score_snapshot.get("counterfactual_deltas", {})
                _target_agent_id = orch_parsed.get("arguments", {}).get("target_floor_agent_id", "")
                orch_cf_delta = float(_cf_deltas.get(_target_agent_id, 0.0)) if _target_agent_id else 0.0
                rationale_rows.append(
                    {
                        "agent_id": "orchestrator",
                        "action_id": orch_action.action_id,
                        "eligible_tokens": len(str(orch_rationale).split()),
                        "bonus_awarded": result.rewards_by_role.orchestrator.breakdown.get_components().get(
                            "rationale_bonus", 0.0
                        ),
                        "gates_passed": ["has_rationale"],
                        "counterfactual_delta": orch_cf_delta,
                        "reason_hash": uuid.uuid5(uuid.NAMESPACE_OID, str(orch_rationale)).hex[:12],
                    }
                )
            samples.append(
                TrajectorySample(
                    episode_id=episode_id,
                    round_id=round_id,
                    agent_id="orchestrator",
                    role="orchestrator",
                    seed=seed,
                    tier=tier,
                    disaster_family=disaster_family_str,
                    generator_config_hash=generator_config_hash,
                    prompt=orch_prompt,
                    completion_text=orch_completion,
                    completion_token_ids=orch_completion_ids,
                    parsed_action=orch_parsed,
                    raw_reward=orch_raw,
                    normalized_reward=orch_norm,
                    done=done,
                    checkpoint_tag=checkpoint_tag,
                    group_id="rollout_orchestrator",
                    trace_schema_version=_TRACE_V,
                    reward_schema_version=_REWARD_V,
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    model_name=model_name,
                )
            )

            for agent_id, floor_obs in obs_by_role.floors.items():
                floor_completion, floor_prompt = floor_payloads[agent_id]
                floor_action = floor_actions[agent_id]
                floor_parsed = floor_action.model_dump(mode="json")
                floor_reward = result.rewards_by_role.floors.get(agent_id)
                floor_raw = 0.0 if floor_reward is None else floor_reward.raw
                floor_norm = norm_rewards.get(agent_id, 0.0)
                total_raw[agent_id] += floor_raw
                total_norm[agent_id] += floor_norm
                breakdown = {} if floor_reward is None else floor_reward.breakdown.get_components()
                reward_rows.append((agent_id, floor_raw, floor_norm, breakdown))
                floor_bonus = float(breakdown.get("rationale_bonus", 0.0))
                if floor_bonus > 0.0:
                    rationale_bonus_total += floor_bonus
                    rationale_bonus_count += 1
                rationale = floor_parsed.get("rationale")
                if rationale:
                    rationale_rows.append(
                        {
                            "agent_id": agent_id,
                            "action_id": floor_action.action_id,
                            "eligible_tokens": len(str(rationale).split()),
                            "bonus_awarded": breakdown.get("rationale_bonus", 0.0),
                            "gates_passed": ["has_rationale"],
                            "counterfactual_delta": result.info.score_snapshot.get("counterfactual_deltas", {}).get(
                                agent_id, 0.0
                            ),
                            "reason_hash": uuid.uuid5(uuid.NAMESPACE_OID, str(rationale)).hex[:12],
                        }
                    )
                if candidate_count > 1:
                    candidate_group_id = _prompt_scoped_group_id(
                        episode_id=episode_id,
                        round_id=round_id,
                        role="floor_agent",
                        agent_id=agent_id,
                        prompt=floor_prompt,
                    )
                    for candidate_index, (
                        (candidate_text, candidate_token_ids),
                        candidate_action,
                        candidate_parse_status,
                    ) in enumerate(
                        zip(
                            floor_candidate_payloads[agent_id],
                            floor_candidate_actions[agent_id],
                            floor_candidate_parse_statuses[agent_id],
                            strict=True,
                        )
                    ):
                        selected_for_execution = candidate_index == floor_selected_candidate_indices[agent_id]
                        if selected_for_execution:
                            candidate_raw_reward = floor_raw
                            candidate_normalized_reward = floor_norm
                        else:
                            candidate_raw_reward = _floor_candidate_reward(
                                floor_obs,
                                candidate_action,
                                candidate_parse_status,
                            )
                            candidate_normalized_reward = candidate_raw_reward
                        samples.append(
                            TrajectorySample(
                                episode_id=episode_id,
                                round_id=round_id,
                                agent_id=agent_id,
                                role="floor_agent",
                                seed=seed,
                                tier=tier,
                                disaster_family=disaster_family_str,
                                generator_config_hash=generator_config_hash,
                                prompt=floor_prompt,
                                completion_text=candidate_text,
                                completion_token_ids=candidate_token_ids,
                                parsed_action=_candidate_parsed_action(
                                    candidate_action,
                                    candidate_parse_status,
                                    candidate_index,
                                    selected_for_execution=selected_for_execution,
                                ),
                                raw_reward=candidate_raw_reward,
                                normalized_reward=candidate_normalized_reward,
                                done=done,
                                checkpoint_tag=checkpoint_tag,
                                group_id=candidate_group_id,
                                trace_schema_version=_TRACE_V,
                                reward_schema_version=_REWARD_V,
                                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                                model_name=model_name,
                            )
                        )
                else:
                    samples.append(
                        TrajectorySample(
                            episode_id=episode_id,
                            round_id=round_id,
                            agent_id=agent_id,
                            role="floor_agent",
                            seed=seed,
                            tier=tier,
                            disaster_family=disaster_family_str,
                            generator_config_hash=generator_config_hash,
                            prompt=floor_prompt,
                            completion_text=floor_completion,
                            completion_token_ids=floor_completion_ids.get(agent_id, []),
                            parsed_action=floor_parsed,
                            raw_reward=floor_raw,
                            normalized_reward=floor_norm,
                            done=done,
                            checkpoint_tag=checkpoint_tag,
                            group_id=f"ep_{episode_id}_r_{round_id}_floor",
                            trace_schema_version=_TRACE_V,
                            reward_schema_version=_REWARD_V,
                            prompt_template_version=PROMPT_TEMPLATE_VERSION,
                            model_name=model_name,
                        )
                    )

            _emit_round_artifacts(
                jsonl_dir=jsonl_dir,
                env=env,
                episode_id=episode_id,
                common=common,
                round_events=result.round_events,
                orchestrator_action=orch_action,
                floor_actions=floor_actions,
                reward_rows=reward_rows,
                belief_rows=belief_rows,
                rationale_rows=rationale_rows,
                completion_rows={
                    "orchestrator": {
                        "completion_text": orch_completion,
                        "parse_status": orch_parse_status,
                    },
                    **{
                        agent_id: {
                            "completion_text": floor_payloads[agent_id][0],
                            "parse_status": floor_parse_statuses[agent_id],
                        }
                        for agent_id in floor_ids
                    },
                },
            )

            obs_by_role = result.observations_by_role
            round_id += 1

        elapsed = time.monotonic() - started
        final_state = env.get_internal_state(episode_id)
        episode_result = EpisodeRolloutResult(
            episode_id=episode_id,
            seed=seed,
            tier=tier,
            disaster_family=disaster_family_str,
            generator_config_hash=generator_config_hash,
            samples=samples,
            total_raw_reward_by_role=total_raw,
            total_normalized_reward_by_role=total_norm,
            done_reason=done_reason,
            num_rounds=round_id,
            wall_clock_seconds=elapsed,
            rationale_bonus_total=rationale_bonus_total,
            rationale_bonus_count=rationale_bonus_count,
            priority_component_totals=priority_component_totals,
            priority_component_counts=priority_component_counts,
            priority_behavior_totals=priority_behavior_totals,
            priority_behavior_counts=priority_behavior_counts,
            priority_directive_issue_count=priority_directive_issue_count,
        )
        write_trace_row(
            jsonl_dir / "episode_summary.jsonl",
            {
                **_trace_common(
                    episode_id=episode_id,
                    round_id=max(round_id - 1, 0),
                    seed=seed,
                    tier=tier,
                    disaster_family=disaster_family_str,
                    generator_config_hash=generator_config_hash,
                    checkpoint_tag=checkpoint_tag,
                    model_name=model_name,
                ),
                "total_reward": total_norm.get("orchestrator", 0.0),
                "termination_reason": done_reason,
                "civilians_saved": final_state.civilians_saved.total,
                "civilians_lost": final_state.civilians_lost.total,
                "total_steps": round_id,
            },
        )
        return episode_result
    finally:
        if cleanup_env_episode and episode_id is not None:
            try:
                env.cleanup_episode(episode_id)
            except Exception as exc:
                logger.warning("cleanup_episode(%s) failed: %s", episode_id, exc)


def collect_batch(
    env: EvacEnvironment,
    policy: Policy,
    curriculum: "CurriculumController",
    *,
    num_episodes: int,
    seed_generator: Callable[[], int],
    disaster_families: Sequence[DisasterType],
    max_rounds: int = 80,
    checkpoint_tag: str = "baseline",
    model_name: str = "stub",
    is_eval: bool = False,
    normalizer: RewardNormalizer | None = None,
    jsonl_dir: Path | None = None,
    seed_collision_retry_limit: int = 1000,
    rationale_mode: str | None = None,
    cleanup_env_episodes: bool = True,
    reward_config: Mapping[str, object] | None = None,
    candidates_per_floor_prompt: int = 1,
    include_oracle_floor_candidate: bool = False,
) -> list[EpisodeRolloutResult]:
    if normalizer is None:
        normalizer = RewardNormalizer()
    if jsonl_dir is None:
        jsonl_dir = Path("outputs/logs")

    results: list[EpisodeRolloutResult] = []
    family_cycle_idx = 0
    for _ in range(num_episodes):
        family = disaster_families[family_cycle_idx % len(disaster_families)]
        family_cycle_idx += 1
        family_str = _extract_disaster_family_str(family)
        tier = _extract_tier_str(curriculum.suggest_next_tier(family_str))

        seed = seed_generator()
        if not is_eval:
            attempts = 0
            while seed in EVAL_SEEDS:
                attempts += 1
                if attempts >= seed_collision_retry_limit:
                    raise RuntimeError(
                        f"Unable to draw a non-eval training seed after {seed_collision_retry_limit} attempts; "
                        "seed collided with EVAL_SEEDS."
                    )
                seed = seed_generator()

        episode = collect_episode(
            env,
            policy,
            seed=seed,
            tier=tier,
            disaster_family=family,
            max_rounds=max_rounds,
            checkpoint_tag=checkpoint_tag,
            model_name=model_name,
            normalizer=normalizer,
            update_normalizer=not is_eval,
            jsonl_dir=jsonl_dir,
            cleanup_env_episode=cleanup_env_episodes,
            rationale_mode=rationale_mode,
            reward_config=reward_config,
            candidates_per_floor_prompt=candidates_per_floor_prompt,
            include_oracle_floor_candidate=include_oracle_floor_candidate,
        )
        if not is_eval:
            curriculum.record_outcome(
                tier=tier,
                disaster_family=family_str,
                normalized_reward=episode.total_normalized_reward_by_role.get("orchestrator", 0.0),
                seed=seed,
                is_eval=False,
            )
        results.append(episode)

    return results
