"""Reward normalization tests (Phase 3 deliverable #5).

Tests cover:
- Ordering preservation: raw_a < raw_b => normalized_a <= normalized_b for all pairs.
- Bounded: normalized_reward ∈ [-1, 1] for every sample.
- Snapshot stability: json round-trip of pipeline.snapshot().
- Schema version: REWARD_SCHEMA_VERSION present in snapshot.
"""

from __future__ import annotations

import json
import random
from itertools import combinations

from evacos_ma.reward_pipeline import RewardPipeline, RunningRewardStats, normalize_reward
from evacos_ma.schemas.rewards import REWARD_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _generate_sample(n: int = 50, seed: int = 42) -> list[tuple[str, float]]:
    """Generate fixed synthetic (tier, raw_reward) pairs."""
    rng = random.Random(seed)
    tiers = ["easy", "medium", "hard", "brutal"]
    return [(rng.choice(tiers), rng.uniform(-1000, 1000)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ordering_preservation() -> None:
    """For all C(50,2) same-tier pairs, raw_a < raw_b => normalized_a <= normalized_b."""
    samples = _generate_sample(50)
    pipeline = RewardPipeline()

    # First pass: observe all samples to build up stats
    tier_samples: dict[str, list[float]] = {}
    for tier, raw in samples:
        tier_samples.setdefault(tier, []).append(raw)
        pipeline.observe(raw, tier)

    # Second pass: check ordering within each tier
    for tier, raws in tier_samples.items():
        if len(raws) < 2:
            continue
        normalized = [pipeline.normalize(r, tier) for r in raws]
        for (i, raw_a), (j, raw_b) in combinations(enumerate(raws), 2):
            if raw_a < raw_b:
                assert normalized[i] <= normalized[j], (
                    f"Ordering violated: tier={tier}, raw_a={raw_a} < raw_b={raw_b} "
                    f"but norm_a={normalized[i]} > norm_b={normalized[j]}"
                )
            elif raw_b < raw_a:
                assert normalized[j] <= normalized[i], (
                    f"Ordering violated: tier={tier}, raw_b={raw_b} < raw_a={raw_a} "
                    f"but norm_b={normalized[j]} > norm_a={normalized[i]}"
                )


def test_bounded_normalized_rewards() -> None:
    """Every normalized reward is within [-1, 1]."""
    samples = _generate_sample(50)
    pipeline = RewardPipeline()

    for tier, raw in samples:
        pipeline.observe(raw, tier)

    for tier, raw in samples:
        norm = pipeline.normalize(raw, tier)
        assert -1.0 <= norm <= 1.0, (
            f"normalized_reward={norm} out of [-1, 1] for tier={tier}, raw={raw}"
        )


def test_snapshot_stability() -> None:
    """pipeline.snapshot() round-trips via json.dumps/loads and equals itself."""
    samples = _generate_sample(50)
    pipeline = RewardPipeline()

    for tier, raw in samples:
        pipeline.observe(raw, tier)

    snap = pipeline.snapshot()
    json_str = json.dumps(snap)
    snap_rt = json.loads(json_str)
    assert snap == snap_rt, "Snapshot round-trip mismatch"


def test_schema_version_in_snapshot() -> None:
    """REWARD_SCHEMA_VERSION appears in every snapshot tier entry."""
    samples = _generate_sample(50)
    pipeline = RewardPipeline()

    for tier, raw in samples:
        pipeline.observe(raw, tier)

    snap = pipeline.snapshot()
    for tier, entry in snap.items():
        assert "reward_schema_version" in entry, (
            f"Missing reward_schema_version in tier {tier}"
        )
        assert entry["reward_schema_version"] == REWARD_SCHEMA_VERSION


def test_normalize_reward_ordering_unit() -> None:
    """Unit test: normalize_reward is order-preserving on a simple case."""
    stats = RunningRewardStats(count=100, mean=0.0, m2=100.0, min=-50.0, max=50.0, tier="easy")
    vals = [-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0]
    normalized = [normalize_reward(v, stats) for v in vals]
    for i in range(len(normalized) - 1):
        assert normalized[i] <= normalized[i + 1], (
            f"Ordering violated: vals[{i}]={vals[i]} => {normalized[i]} "
            f"vs vals[{i+1}]={vals[i+1]} => {normalized[i+1]}"
        )
