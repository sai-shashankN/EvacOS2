"""Multi-role rollout collector."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from curriculum.controller import EVAL_SEEDS
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionEnvelopeMA,
    ActionTypeMA,
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


def _extract_tier_str(tier_obj: object) -> str:
    if isinstance(tier_obj, str):
        return tier_obj
    return str(getattr(tier_obj, "value", tier_obj))


def _extract_disaster_family_str(disaster_family: object) -> str:
    if isinstance(disaster_family, str):
        return disaster_family
    return str(getattr(disaster_family, "value", disaster_family))


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


def _emit_round_artifacts(
    *,
    jsonl_dir: Path,
    common: dict[str, object],
    round_events: list[dict],
    orchestrator_action: ActionEnvelopeMA,
    floor_actions: dict[str, ActionEnvelopeMA],
    reward_rows: list[tuple[str, float, float, dict]],
    belief_rows: list[dict],
    rationale_rows: list[dict],
) -> None:
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
        },
    )

    for agent_id, action in [("orchestrator", orchestrator_action), *sorted(floor_actions.items())]:
        parsed = action.model_dump(mode="json")
        valid = "fallback_reason" not in parsed
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
) -> EpisodeRolloutResult:
    if normalizer is None:
        normalizer = RewardNormalizer()
    if jsonl_dir is None:
        jsonl_dir = Path("outputs/logs")

    started = time.monotonic()
    disaster_family_str = _extract_disaster_family_str(disaster_family)
    episode_id, obs_by_role = env.reset_multi_agent(
        task_id=f"procgen_{tier}_{disaster_family_str}",
        seed=seed,
        procgen_tier=tier,
        procgen_disaster_family=disaster_family,
    )

    generator_config_hash = obs_by_role.orchestrator.generator_config_hash
    samples: list[TrajectorySample] = []
    total_raw: dict[str, float] = {"orchestrator": 0.0}
    total_norm: dict[str, float] = {"orchestrator": 0.0}
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

        # Batched fast path: policies that expose act_batch collapse the 6
        # per-round calls (1 orchestrator + 5 floors) into a single generate
        # call. Detection is by hasattr, so StubPolicy / hf_policy_factory
        # fall back to the per-agent loop transparently.
        if hasattr(policy, "act_batch"):
            batch_prompts: list[list[dict[str, str]]] = [orch_prompt]
            batch_agent_ids: list[str] = ["orchestrator"]
            batch_roles: list[str] = ["orchestrator"]
            for aid in floor_ids:
                batch_prompts.append(floor_prompts[aid])
                batch_agent_ids.append(aid)
                batch_roles.append("floor_agent")
            completions = policy.act_batch(batch_prompts, batch_agent_ids, batch_roles)
            orch_completion = completions[0]
            floor_completions = {
                aid: completions[i + 1] for i, aid in enumerate(floor_ids)
            }
        else:
            orch_completion = policy.act(orch_prompt, "orchestrator", "orchestrator")
            floor_completions = {
                aid: policy.act(floor_prompts[aid], aid, "floor_agent")
                for aid in floor_ids
            }

        orch_action, _ = parse_completion_to_action(orch_completion, "orchestrator", "orchestrator")
        if orch_action is None:
            orch_action = _fallback_wait_action(episode_id, round_id, "orchestrator")
            orch_action.fallback_reason = "parse_error"

        floor_actions: dict[str, ActionEnvelopeMA] = {}
        floor_payloads: dict[str, tuple[str, list[dict[str, str]]]] = {}
        for agent_id in floor_ids:
            floor_prompt = floor_prompts[agent_id]
            floor_completion = floor_completions[agent_id]
            floor_action, _ = parse_completion_to_action(floor_completion, agent_id, "floor_agent")
            if floor_action is None:
                floor_action = _fallback_wait_action(episode_id, round_id, agent_id)
                floor_action.fallback_reason = "parse_error"
            floor_actions[agent_id] = floor_action
            floor_payloads[agent_id] = (floor_completion, floor_prompt)

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

        norm_rewards = normalize_per_role(
            result.rewards_by_role,
            tier,
            normalizer,
            update=update_normalizer,
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
        total_raw["orchestrator"] += orch_raw
        total_norm["orchestrator"] += orch_norm
        reward_rows.append(
            (
                "orchestrator",
                orch_raw,
                orch_norm,
                result.rewards_by_role.orchestrator.breakdown.get_components(),
            )
        )
        orch_rationale = orch_parsed.get("rationale")
        if orch_rationale:
            rationale_rows.append(
                {
                    "agent_id": "orchestrator",
                    "action_id": orch_action.action_id,
                    "eligible_tokens": len(str(orch_rationale).split()),
                    "bonus_awarded": result.rewards_by_role.orchestrator.breakdown.get_components().get(
                        "rationale_bonus", 0.0
                    ),
                    "gates_passed": ["has_rationale"],
                    "counterfactual_delta": result.info.score_snapshot.get("counterfactual_deltas", {}).get(
                        "orchestrator", 0.0
                    ),
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
                parsed_action=orch_parsed,
                raw_reward=orch_raw,
                normalized_reward=orch_norm,
                done=done,
                checkpoint_tag=checkpoint_tag,
                group_id=f"tier_{tier}_orchestrator",
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
                    parsed_action=floor_parsed,
                    raw_reward=floor_raw,
                    normalized_reward=floor_norm,
                    done=done,
                    checkpoint_tag=checkpoint_tag,
                    group_id=f"tier_{tier}_floor_agent",
                    trace_schema_version=_TRACE_V,
                    reward_schema_version=_REWARD_V,
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    model_name=model_name,
                )
            )

        _emit_round_artifacts(
            jsonl_dir=jsonl_dir,
            common=common,
            round_events=result.round_events,
            orchestrator_action=orch_action,
            floor_actions=floor_actions,
            reward_rows=reward_rows,
            belief_rows=belief_rows,
            rationale_rows=rationale_rows,
        )

        obs_by_role = result.observations_by_role
        round_id += 1

    elapsed = time.monotonic() - started
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
            "civilians_saved": 0,
            "civilians_lost": 0,
            "total_steps": round_id,
        },
    )
    return episode_result


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
