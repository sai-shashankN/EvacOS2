"""Deterministic fixed-suite evaluation harness."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from curriculum.controller import EVAL_SEEDS
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType

from evaluation._training_contract import (
    Policy,
    RewardNormalizer,
    ScopeDecision,
    collect_batch,
    route_scope,
)

SCHEMA_VERSION = "2026.04.24"
DEFAULT_DISASTER_FAMILIES: tuple[DisasterType, ...] = tuple(DisasterType)
_MIN_EVAL_ZSCORE_SAMPLES = 30
_DEFAULT_REWARD_CONFIG: dict[str, float | str] = {
    "rationale_scaling": "linear_capped",
    "alpha": 0.01,
    "beta": 0.25,
    "cap": 1.0,
    "eligible_token_ceiling": 160,
    "clip_normalized_to": 1.0,
}


class SummaryStats(BaseModel):
    mean: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0


class AggregateStats(BaseModel):
    eval_score_pct: SummaryStats = Field(default_factory=SummaryStats)
    raw_reward: dict[str, SummaryStats] = Field(default_factory=dict)
    normalized_reward: dict[str, SummaryStats] = Field(default_factory=dict)
    save_rate: SummaryStats = Field(default_factory=SummaryStats)
    invalid_action_rate: SummaryStats = Field(default_factory=SummaryStats)
    override_win_rate: SummaryStats = Field(default_factory=SummaryStats)
    wall_clock_s: SummaryStats = Field(default_factory=SummaryStats)


class EpisodeResult(BaseModel):
    episode_id: str
    label: str
    tier: str
    seed: int
    disaster_family: str
    rationale_mode: str
    checkpoint_tag: str
    model_name: str
    trace_schema_version: str
    generator_config_hash: str
    reward_schema_version: str
    prompt_template_version: str
    total_rounds: int
    wall_clock_s: float
    termination_reason: str | None = None
    total_civilians: int = 0
    civilians_saved: int = 0
    civilians_lost: int = 0
    civilians_remaining: int = 0
    save_rate: float = 0.0
    eval_score_pct: float = 0.0
    invalid_action_rate: float = 0.0
    override_win_rate: float = 0.0
    action_type_counts: dict[str, int] = Field(default_factory=dict)
    action_type_counts_by_role: dict[str, dict[str, int]] = Field(default_factory=dict)
    wait_rate: float = 0.0
    scout_rate: float = 0.0
    route_rate: float = 0.0
    evacuate_rate: float = 0.0
    raw_reward_by_role: dict[str, float] = Field(default_factory=dict)
    normalized_reward_by_role: dict[str, float] = Field(default_factory=dict)
    scope_policy_key: str = "generalist"
    scope_route_reason: str = "not_routed"
    schema_version: str = SCHEMA_VERSION


class FixedSuiteResult(BaseModel):
    label: str
    tiers: list[str]
    seeds: list[int]
    disaster_families: list[str]
    rationale_mode: str
    env_side_rationale_wired: bool = False
    normalizer_z_scored: bool = False
    episodes: list[EpisodeResult]
    aggregate: AggregateStats
    schema_version: str = SCHEMA_VERSION


class FixedTierCurriculum:
    """Minimal tier pinning shim for eval."""

    def __init__(self, tier: str) -> None:
        self._tier = tier

    def suggest_next_tier(self, disaster_family: str) -> str:
        del disaster_family
        return self._tier

    def record_outcome(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        return None


def _coerce_family(family: DisasterType | str) -> DisasterType:
    if isinstance(family, DisasterType):
        return family
    return DisasterType(str(family))


def _deterministic_episode_id(label: str, rationale_mode: str, tier: str, seed: int, family: str) -> str:
    return f"{label}:{rationale_mode}:{tier}:{seed}:{family}"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _summary(values: Sequence[float]) -> SummaryStats:
    if not values:
        return SummaryStats()
    arr = np.asarray(values, dtype=float)
    return SummaryStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _count_actions(counts: Counter[str], names: Sequence[str]) -> int:
    return sum(counts.get(name, 0) for name in names)


def _eval_score_pct(*, save_rate: float, invalid_action_rate: float) -> float:
    """Bounded human-facing score; training reward remains separate."""
    valid_multiplier = 1.0 - (0.5 * _clamp01(invalid_action_rate))
    return 100.0 * _clamp01(save_rate) * _clamp01(valid_multiplier)


def _resolve_reward_config(
    reward_config: Mapping[str, object] | None,
    rationale_mode: str,
) -> dict[str, object]:
    payload: dict[str, object] = dict(_DEFAULT_REWARD_CONFIG)
    if reward_config is not None:
        payload.update(dict(reward_config))
    payload["rationale_scaling"] = rationale_mode
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_episode_result(
    *,
    label: str,
    rationale_mode: str,
    tier: str,
    seed: int,
    family: str,
    log_dir: Path,
    batch_result: object,
    env: EvacEnvironment,
    scope_decision: ScopeDecision,
) -> EpisodeResult:
    actual_episode_id = batch_result.episode_id
    action_rows = [
        row for row in _read_jsonl(log_dir / "action_trace.jsonl")
        if row.get("episode_id") == actual_episode_id
    ]
    rationale_rows = [
        row for row in _read_jsonl(log_dir / "rationale_audit.jsonl")
        if row.get("episode_id") == actual_episode_id
    ]
    summary_rows = [
        row for row in _read_jsonl(log_dir / "episode_summary.jsonl")
        if row.get("episode_id") == actual_episode_id
    ]

    action_count = len(action_rows)
    invalid_count = sum(1 for row in action_rows if not row.get("valid", True))
    action_counts = Counter(
        str(row.get("action_type", "unknown")) for row in action_rows
    )
    action_counts_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    for row in action_rows:
        agent_id = str(row.get("agent_id", ""))
        role = "orchestrator" if agent_id == "orchestrator" else "floor_agent"
        action_counts_by_role[role][str(row.get("action_type", "unknown"))] += 1
    override_ids = {
        row["action_id"]
        for row in action_rows
        if row.get("action_type") == "override_floor_agent"
    }
    override_wins = sum(
        1
        for row in rationale_rows
        if row.get("action_id") in override_ids and float(row.get("counterfactual_delta", 0.0)) > 0
    )
    summary = summary_rows[-1] if summary_rows else {}

    state = env.get_internal_state(actual_episode_id)
    total_civilians = max(state.total_civilians.total, 1)
    saved = state.civilians_saved.total
    lost = state.civilians_lost.total
    remaining = max(total_civilians - saved - lost, 0)

    floor_raw = _mean(
        value for key, value in batch_result.total_raw_reward_by_role.items() if key != "orchestrator"
    )
    floor_norm = _mean(
        value for key, value in batch_result.total_normalized_reward_by_role.items() if key != "orchestrator"
    )

    return EpisodeResult(
        episode_id=_deterministic_episode_id(label, rationale_mode, tier, seed, family),
        label=label,
        tier=tier,
        seed=seed,
        disaster_family=family,
        rationale_mode=rationale_mode,
        checkpoint_tag=str(summary.get("checkpoint_tag", label)),
        model_name=str(summary.get("model_name", "stub")),
        trace_schema_version=str(summary.get("trace_schema_version", "v1")),
        generator_config_hash=str(summary.get("generator_config_hash", "")),
        reward_schema_version=str(summary.get("reward_schema_version", "v1")),
        prompt_template_version=str(summary.get("prompt_template_version", "2026.04.20")),
        total_rounds=int(batch_result.num_rounds),
        wall_clock_s=float(batch_result.wall_clock_seconds),
        termination_reason=batch_result.done_reason,
        total_civilians=int(total_civilians),
        civilians_saved=saved,
        civilians_lost=lost,
        civilians_remaining=int(remaining),
        save_rate=float(saved / total_civilians),
        eval_score_pct=_eval_score_pct(
            save_rate=float(saved / total_civilians),
            invalid_action_rate=float(invalid_count / action_count) if action_count else 0.0,
        ),
        invalid_action_rate=float(invalid_count / action_count) if action_count else 0.0,
        override_win_rate=float(override_wins / len(override_ids)) if override_ids else 0.0,
        action_type_counts={key: int(value) for key, value in sorted(action_counts.items())},
        action_type_counts_by_role={
            role: {key: int(value) for key, value in sorted(counts.items())}
            for role, counts in sorted(action_counts_by_role.items())
        },
        wait_rate=float(action_counts.get("wait", 0) / action_count) if action_count else 0.0,
        scout_rate=float(_count_actions(action_counts, ("scout", "scout_status")) / action_count)
        if action_count
        else 0.0,
        route_rate=float(
            _count_actions(
                action_counts,
                ("route_within_floor", "route_between_floors", "route_civilians"),
            )
            / action_count
        )
        if action_count
        else 0.0,
        evacuate_rate=float(
            _count_actions(
                action_counts,
                ("evacuate_floor_priority", "evacuate_floor", "evacuate_room"),
            )
            / action_count
        )
        if action_count
        else 0.0,
        raw_reward_by_role={
            "orchestrator": float(batch_result.total_raw_reward_by_role.get("orchestrator", 0.0)),
            "floor_agent": float(floor_raw),
        },
        normalized_reward_by_role={
            "orchestrator": float(batch_result.total_normalized_reward_by_role.get("orchestrator", 0.0)),
            "floor_agent": float(floor_norm),
        },
        scope_policy_key=scope_decision.policy_key,
        scope_route_reason=scope_decision.reason,
    )


def _build_aggregate(episodes: Sequence[EpisodeResult]) -> AggregateStats:
    roles = ("orchestrator", "floor_agent")
    raw_reward = {
        role: _summary([episode.raw_reward_by_role.get(role, 0.0) for episode in episodes])
        for role in roles
    }
    normalized_reward = {
        role: _summary([episode.normalized_reward_by_role.get(role, 0.0) for episode in episodes])
        for role in roles
    }
    return AggregateStats(
        eval_score_pct=_summary([episode.eval_score_pct for episode in episodes]),
        raw_reward=raw_reward,
        normalized_reward=normalized_reward,
        save_rate=_summary([episode.save_rate for episode in episodes]),
        invalid_action_rate=_summary([episode.invalid_action_rate for episode in episodes]),
        override_win_rate=_summary([episode.override_win_rate for episode in episodes]),
        wall_clock_s=_summary([episode.wall_clock_s for episode in episodes]),
    )


def _write_result(result: FixedSuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _seed_eval_normalizer(
    tier: str,
    snapshot: dict | None = None,
) -> RewardNormalizer:
    """Return an eval normalizer, optionally preloaded from checkpoint stats."""
    del tier
    normalizer = RewardNormalizer()
    if snapshot is not None:
        normalizer.load_snapshot(snapshot)
    return normalizer


def _snapshot_has_zscore_state(snapshot: dict | None) -> bool:
    if snapshot is None:
        return False
    for state in snapshot.values():
        if int(state.get("count", 0)) >= _MIN_EVAL_ZSCORE_SAMPLES:
            return True
    return False


def _policy_for_scope(
    policy_factory: Callable[[], Policy],
    scope_decision: ScopeDecision,
) -> Policy:
    scoped_factory = getattr(policy_factory, "for_scope", None)
    if callable(scoped_factory):
        return scoped_factory(scope_decision)
    return policy_factory()


def _cached_policy_for_scope(
    policy_factory: Callable[[], Policy],
    scope_decision: ScopeDecision,
    policy_cache: dict[str, Policy],
) -> Policy:
    """Reuse loaded policies across eval episodes.

    Trained HF/LoRA policy factories can load multi-GB checkpoints. The fixed
    suite evaluates many seeds/tiers per scope, so calling the factory inside
    the episode loop reloads the model per episode and burns GPU time.
    """
    scoped_factory = getattr(policy_factory, "for_scope", None)
    cache_key = scope_decision.policy_key if callable(scoped_factory) else "__default__"
    if cache_key not in policy_cache:
        policy_cache[cache_key] = _policy_for_scope(policy_factory, scope_decision)
    return policy_cache[cache_key]


def run_fixed_suite(
    policy_factory: Callable[[], Policy],
    *,
    tiers: Sequence[str] = ("easy", "medium"),
    seeds: Sequence[int] = EVAL_SEEDS,
    disaster_families: Sequence[DisasterType | str] = DEFAULT_DISASTER_FAMILIES,
    max_rounds: int = 500,
    rationale_mode: str = "linear_capped",
    label: str = "trained",
    output_dir: Path = Path("outputs/evals"),
    normalizer_snapshot: dict | None = None,
    reward_config: Mapping[str, object] | None = None,
) -> FixedSuiteResult:
    families = [_coerce_family(family) for family in disaster_families]
    episodes: list[EpisodeResult] = []
    policy_cache: dict[str, Policy] = {}
    resolved_reward_config = _resolve_reward_config(reward_config, rationale_mode)

    for tier in tiers:
        for seed in seeds:
            for family in families:
                env = EvacEnvironment()
                curriculum = FixedTierCurriculum(tier)
                scope_decision = route_scope(
                    {
                        "disaster_family": family.value,
                        "tier": tier,
                    }
                )
                policy = _cached_policy_for_scope(policy_factory, scope_decision, policy_cache)
                normalizer = _seed_eval_normalizer(tier, snapshot=normalizer_snapshot)
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                results = collect_batch(
                    env,
                    policy,
                    curriculum,
                    num_episodes=1,
                    seed_generator=lambda seed=seed: seed,
                    disaster_families=[family],
                    max_rounds=max_rounds,
                    checkpoint_tag=label,
                    model_name=type(policy).__name__,
                    is_eval=True,
                    normalizer=normalizer,
                    jsonl_dir=log_dir,
                    rationale_mode=rationale_mode,
                    reward_config=resolved_reward_config,
                    cleanup_env_episodes=False,
                )
                # Eval-purity contract is enforced by `is_eval=True` flowing into
                # `collect_batch -> collect_episode -> normalize_per_role(update=False)`.
                # A monkeypatch-based test asserts `.update()` is never called.
                # Snapshot-equality would be too strict because `_get_state` lazy-
                # creates empty state entries on first `.normalize(...)` call.
                episode = _build_episode_result(
                    label=label,
                    rationale_mode=rationale_mode,
                    tier=tier,
                    seed=int(seed),
                    family=family.value,
                    log_dir=log_dir,
                    batch_result=results[0],
                    env=env,
                    scope_decision=scope_decision,
                )
                episodes.append(episode)

    episodes.sort(key=lambda item: (item.tier, item.seed, item.disaster_family))
    result = FixedSuiteResult(
        label=label,
        tiers=list(tiers),
        seeds=[int(seed) for seed in seeds],
        disaster_families=[family.value for family in families],
        rationale_mode=rationale_mode,
        env_side_rationale_wired=True,
        normalizer_z_scored=_snapshot_has_zscore_state(normalizer_snapshot),
        episodes=episodes,
        aggregate=_build_aggregate(episodes),
    )
    _write_result(result, output_dir / f"fixed_suite_{label}_{rationale_mode}.json")
    return result
