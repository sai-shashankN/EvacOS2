"""Deterministic fixed-suite evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from curriculum.controller import EVAL_SEEDS
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from training.policy_adapter import Policy
from training.reward import RewardNormalizer
from training.rollout import collect_batch

SCHEMA_VERSION = "2026.04.20"
DEFAULT_DISASTER_FAMILIES: tuple[DisasterType, ...] = tuple(DisasterType)


class SummaryStats(BaseModel):
    mean: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0


class AggregateStats(BaseModel):
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
    civilians_saved: int = 0
    civilians_lost: int = 0
    save_rate: float = 0.0
    invalid_action_rate: float = 0.0
    override_win_rate: float = 0.0
    raw_reward_by_role: dict[str, float] = Field(default_factory=dict)
    normalized_reward_by_role: dict[str, float] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


class FixedSuiteResult(BaseModel):
    label: str
    tiers: list[str]
    seeds: list[int]
    disaster_families: list[str]
    rationale_mode: str
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
        civilians_saved=saved,
        civilians_lost=lost,
        save_rate=float(saved / total_civilians),
        invalid_action_rate=float(invalid_count / action_count) if action_count else 0.0,
        override_win_rate=float(override_wins / len(override_ids)) if override_ids else 0.0,
        raw_reward_by_role={
            "orchestrator": float(batch_result.total_raw_reward_by_role.get("orchestrator", 0.0)),
            "floor_agent": float(floor_raw),
        },
        normalized_reward_by_role={
            "orchestrator": float(batch_result.total_normalized_reward_by_role.get("orchestrator", 0.0)),
            "floor_agent": float(floor_norm),
        },
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
def _seed_eval_normalizer(tier: str) -> RewardNormalizer:
    # Phase 8 eval purity contract: no .update() calls in eval. Return a cold
    # normalizer; collect_batch(is_eval=True, update_normalizer=False) will
    # normalize through whatever state the normalizer is in, matching the
    # Phase 7 notebook eval cell's pattern (cold RewardNormalizer + is_eval=True).
    return RewardNormalizer()


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
) -> FixedSuiteResult:
    families = [_coerce_family(family) for family in disaster_families]
    episodes: list[EpisodeResult] = []

    for tier in tiers:
        for seed in seeds:
            for family in families:
                env = EvacEnvironment()
                curriculum = FixedTierCurriculum(tier)
                policy = policy_factory()
                normalizer = _seed_eval_normalizer(tier)
                log_dir = Path("outputs/logs")
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
                )
                episodes.append(episode)

    episodes.sort(key=lambda item: (item.tier, item.seed, item.disaster_family))
    result = FixedSuiteResult(
        label=label,
        tiers=list(tiers),
        seeds=[int(seed) for seed in seeds],
        disaster_families=[family.value for family in families],
        rationale_mode=rationale_mode,
        episodes=episodes,
        aggregate=_build_aggregate(episodes),
    )
    _write_result(result, output_dir / f"fixed_suite_{label}_{rationale_mode}.json")
    return result
