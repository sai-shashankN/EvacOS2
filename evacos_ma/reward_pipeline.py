"""Raw vs normalized reward pipeline.

Provides a clean separation where the environment emits a raw_reward
(task-interpretable, unbounded) and a normalized_reward in [-1, 1] derived
via per-tier Welford running statistics.

- raw_reward: the interpretable per-round reward; used for reporting/eval.
- normalized_reward: purely a training signal, bounded in [-clip, clip].

The raw_reward preserves the Phase 1 meaning. normalized_reward is new.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from evacos_ma.schemas.rewards import REWARD_SCHEMA_VERSION


@dataclass
class RunningRewardStats:
    """Welford incremental moments for reward normalization per tier."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    tier: str = ""

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict with safe JSON values."""
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min": self.min if math.isfinite(self.min) else None,
            "max": self.max if math.isfinite(self.max) else None,
            "tier": self.tier,
        }


def normalize_reward(
    raw: float,
    stats: RunningRewardStats,
    clip: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Normalize a raw reward using running stats.

    Returns clip((raw - mean) / (std + eps), -clip, clip).
    For count < 2, returns clip(raw, -clip, clip) (direction-preserving, unnormalized).

    Ordering guarantee: for any two raw values a < b at the same stats snapshot,
    normalize_reward(a, stats) <= normalize_reward(b, stats).
    """
    if stats.count < 2:
        return max(-clip, min(clip, raw))
    std = stats.std
    z = (raw - stats.mean) / (std + eps)
    return max(-clip, min(clip, z))


class RewardPipeline:
    """Orchestrates per-tier running reward stats and normalization.

    Holds one RunningRewardStats per tier. Usage pattern:
      1. Call normalize() to get normalized_reward for the current sample.
      2. Call observe() AFTER using the current snapshot (avoids self-normalization).
    """

    def __init__(self) -> None:
        self._stats: dict[str, RunningRewardStats] = {}
        self._reward_schema_version: str = REWARD_SCHEMA_VERSION

    def _get_stats(self, tier: str) -> RunningRewardStats:
        if tier not in self._stats:
            self._stats[tier] = RunningRewardStats(tier=tier)
        return self._stats[tier]

    def observe(self, raw: float, tier: str) -> None:
        """Update running stats with a new raw reward observation.

        Call this AFTER using the current snapshot for normalization
        so the current sample isn't self-normalizing.
        """
        stats = self._get_stats(tier)
        stats.count += 1
        delta = raw - stats.mean
        stats.mean += delta / stats.count
        delta2 = raw - stats.mean
        stats.m2 += delta * delta2
        if raw < stats.min:
            stats.min = raw
        if raw > stats.max:
            stats.max = raw

    def normalize(self, raw: float, tier: str, clip: float = 1.0) -> float:
        """Normalize a raw reward using current per-tier stats."""
        stats = self._get_stats(tier)
        return normalize_reward(raw, stats, clip=clip)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all tier stats."""
        result: dict[str, Any] = {}
        for tier, stats in sorted(self._stats.items()):
            result[tier] = {
                **stats.to_dict(),
                "reward_schema_version": self._reward_schema_version,
            }
        return result

    def snapshot_json(self) -> str:
        """Return JSON string of snapshot."""
        return json.dumps(self.snapshot())

    @property
    def reward_schema_version(self) -> str:
        return self._reward_schema_version


def compute_raw_reward(
    reward_total: float,
    components: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Compute the raw reward from a Reward object's total and breakdown.

    Returns (raw_reward, breakdown_dict) where breakdown_dict has stable keys
    matching BLUEPRINT's Reward Components.
    """
    return reward_total, dict(components)
