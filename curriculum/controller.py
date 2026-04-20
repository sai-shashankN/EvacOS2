"""Curriculum controller that drives training-time tier selection.

Maintains rolling statistics per (tier, disaster_family) pair and promotes
or demotes the suggested tier based on performance thresholds.  Eval seeds
are held out and never influence curriculum state.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Tier = Literal["easy", "medium", "hard", "brutal"]

EVAL_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)

ROLLING_WINDOW: int = 50
MIN_SAMPLES_FOR_DECISION: int = 30
PROMOTION_THRESHOLD: float = 0.65
DEMOTION_THRESHOLD: float = 0.25
TIER_ORDER: tuple[Tier, ...] = ("easy", "medium", "hard", "brutal")


@dataclass
class TierStats:
    tier: Tier
    disaster_family: str
    sample_count: int = 0
    rolling_rewards: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    last_promoted_round: int | None = None
    last_demoted_round: int | None = None


class CurriculumController:
    """Drives training-time tier selection with rolling statistics."""

    ROLLING_WINDOW: int = ROLLING_WINDOW
    MIN_SAMPLES_FOR_DECISION: int = MIN_SAMPLES_FOR_DECISION
    PROMOTION_THRESHOLD: float = PROMOTION_THRESHOLD
    DEMOTION_THRESHOLD: float = DEMOTION_THRESHOLD
    TIER_ORDER: tuple[Tier, ...] = TIER_ORDER

    def __init__(self, log_path: Path | None = None) -> None:
        # key = (disaster_family, tier)
        self._stats: dict[tuple[str, Tier], TierStats] = {}
        # key = disaster_family -> current suggested tier
        self._current_tier: dict[str, Tier] = {}
        self._log_path = log_path or Path("outputs/logs/curriculum_events.jsonl")
        self._total_outcomes: int = 0

    @staticmethod
    def _canonicalize_disaster_family(disaster_family: object) -> str:
        if hasattr(disaster_family, "value"):
            return str(getattr(disaster_family, "value"))
        text = str(disaster_family)
        if text.startswith("DisasterType."):
            return text.split(".", 1)[1]
        return text

    def _get_stats(self, tier: Tier, disaster_family: str) -> TierStats:
        disaster_family = self._canonicalize_disaster_family(disaster_family)
        key = (disaster_family, tier)
        if key not in self._stats:
            self._stats[key] = TierStats(
                tier=tier,
                disaster_family=disaster_family,
            )
        return self._stats[key]

    def _get_current_tier(self, disaster_family: str) -> Tier:
        disaster_family = self._canonicalize_disaster_family(disaster_family)
        if disaster_family not in self._current_tier:
            self._current_tier[disaster_family] = "easy"
        return self._current_tier[disaster_family]

    def _rolling_mean(self, stats: TierStats) -> float:
        if not stats.rolling_rewards:
            return 0.0
        return sum(stats.rolling_rewards) / len(stats.rolling_rewards)

    def _tier_index(self, tier: Tier) -> int:
        return self.TIER_ORDER.index(tier)

    def record_outcome(
        self,
        tier: Tier,
        disaster_family: str,
        normalized_reward: float,
        seed: int,
        is_eval: bool = False,
    ) -> None:
        """Record a training outcome. Ignores eval seeds and is_eval=True."""
        disaster_family = self._canonicalize_disaster_family(disaster_family)
        if is_eval:
            return
        if seed in EVAL_SEEDS:
            return

        stats = self._get_stats(tier, disaster_family)
        stats.sample_count += 1
        stats.rolling_rewards.append(normalized_reward)
        self._total_outcomes += 1

        # Check promotion/demotion for the current tier
        current = self._get_current_tier(disaster_family)
        if tier != current:
            return

        current_stats = self._get_stats(current, disaster_family)
        if current_stats.sample_count < self.MIN_SAMPLES_FOR_DECISION:
            return

        mean = self._rolling_mean(current_stats)
        current_idx = self._tier_index(current)

        if mean >= self.PROMOTION_THRESHOLD and current_idx < len(self.TIER_ORDER) - 1:
            new_tier = self.TIER_ORDER[current_idx + 1]
            self._current_tier[disaster_family] = new_tier
            current_stats.last_promoted_round = self._total_outcomes
            self._log_event("promote", disaster_family, current, new_tier, mean, current_stats.sample_count)
        elif mean <= self.DEMOTION_THRESHOLD and current_idx > 0:
            new_tier = self.TIER_ORDER[current_idx - 1]
            self._current_tier[disaster_family] = new_tier
            current_stats.last_demoted_round = self._total_outcomes
            self._log_event("demote", disaster_family, current, new_tier, mean, current_stats.sample_count)

    def suggest_next_tier(self, disaster_family: str) -> Tier:
        """Return the tier the trainer should sample next for this disaster family."""
        disaster_family = self._canonicalize_disaster_family(disaster_family)
        return self._get_current_tier(disaster_family)

    def _log_event(
        self,
        event: str,
        disaster_family: str,
        from_tier: Tier,
        to_tier: Tier,
        rolling_mean: float,
        sample_count: int,
    ) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            "disaster_family": disaster_family,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "rolling_mean": round(rolling_mean, 4),
            "sample_count": sample_count,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Curriculum %s: %s %s -> %s (mean=%.3f, n=%d)",
                     event, disaster_family, from_tier, to_tier, rolling_mean, sample_count)

    def snapshot(self) -> dict:
        """JSON-serializable state for checkpointing."""
        stats_snap = {}
        for (df, tier), stats in sorted(self._stats.items()):
            stats_snap[f"{df}:{tier}"] = {
                "tier": stats.tier,
                "disaster_family": stats.disaster_family,
                "sample_count": stats.sample_count,
                "rolling_rewards": list(stats.rolling_rewards),
                "last_promoted_round": stats.last_promoted_round,
                "last_demoted_round": stats.last_demoted_round,
            }
        return {
            "current_tier": {
                self._canonicalize_disaster_family(df): tier
                for df, tier in sorted(self._current_tier.items())
            },
            "stats": stats_snap,
            "total_outcomes": self._total_outcomes,
        }

    def load_snapshot(self, data: dict) -> None:
        """Load state from a snapshot dict."""
        self._current_tier = {
            self._canonicalize_disaster_family(df): tier
            for df, tier in data.get("current_tier", {}).items()
        }
        self._total_outcomes = data.get("total_outcomes", 0)
        self._stats.clear()
        for key_str, stats_data in data.get("stats", {}).items():
            parts = key_str.split(":", 1)
            if len(parts) != 2:
                continue
            df, tier = parts[0], parts[1]
            df = self._canonicalize_disaster_family(df)
            stats = TierStats(
                tier=tier,
                disaster_family=df,
                sample_count=stats_data["sample_count"],
                rolling_rewards=deque(stats_data.get("rolling_rewards", []), maxlen=ROLLING_WINDOW),
                last_promoted_round=stats_data.get("last_promoted_round"),
                last_demoted_round=stats_data.get("last_demoted_round"),
            )
            self._stats[(df, tier)] = stats
