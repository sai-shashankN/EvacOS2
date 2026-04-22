"""Tests for per-role reward normalization (Welford).

Heavy-dep-free.
"""

import json
import math

from evacos_ma.reward_pipeline import RewardPipeline
from training.reward import RewardNormalizer, normalize_per_role
from evacos_ma.schemas.rewards import RoleReward, RewardsByRole


class TestWelfordNormalization:
    def test_after_40_updates_output_is_clipped_and_centered(self):
        """After 40 updates at tier=easy role=orchestrator, normalize() output is
        within [-clip, clip] and has |mean|<0.2 when the input distribution has variance."""
        n = RewardNormalizer()
        # Use varied inputs so stddev > 0 and z-score normalization kicks in
        for i in range(40):
            n.update("orchestrator", "easy", 0.1 * (i % 5))

        # Query the mean value — z-score should be near 0
        mean_val = sum(0.1 * (i % 5) for i in range(40)) / 40
        result = n.normalize("orchestrator", "easy", mean_val)
        assert -1.0 <= result <= 1.0
        # z-score of the mean should be close to 0
        assert abs(result) < 0.2

    def test_constant_input_uses_tanh_fallback(self):
        """When all inputs are the same (stddev=0), tanh fallback is used and result is bounded."""
        n = RewardNormalizer()
        for _ in range(40):
            n.update("orchestrator", "easy", 0.3)
        result = n.normalize("orchestrator", "easy", 0.3)
        assert -1.0 <= result <= 1.0
        assert abs(result - math.tanh(0.3)) < 1e-10

    def test_snapshot_load_roundtrip(self):
        """snapshot() / load_snapshot() JSON round-trip preserves normalize() output."""
        n = RewardNormalizer()
        for i in range(50):
            n.update("orchestrator", "easy", float(i) * 0.1)

        # Get normalized value before snapshot
        before = n.normalize("orchestrator", "easy", 2.5)

        # Round-trip through JSON
        snap = n.snapshot()
        json_str = json.dumps(snap)
        loaded = json.loads(json_str)

        n2 = RewardNormalizer()
        n2.load_snapshot(loaded)
        after = n2.normalize("orchestrator", "easy", 2.5)

        assert before == after


class TestNormalizePerRole:
    def test_normalize_per_role_normalizes_before_update(self):
        n = RewardNormalizer()
        n.load_snapshot(
            {
                "orchestrator:easy": {"count": 40, "mean": 0.5, "m2": 20.0},
                "floor_agent:easy": {"count": 40, "mean": 0.0, "m2": 20.0},
            }
        )
        rewards = RewardsByRole(
            orchestrator=RoleReward(raw=1.0, normalized=0.0),
            floors={"floor_0_agent": RoleReward(raw=0.0, normalized=0.0)},
        )

        normalized = normalize_per_role(rewards, "easy", n, update=True)

        expected = max(-1.0, min(1.0, (1.0 - 0.5) / math.sqrt(20.0 / 40.0)))
        assert normalized["orchestrator"] == expected
        snapshot = n.snapshot()
        assert snapshot["orchestrator:easy"]["count"] == 41
        assert snapshot["orchestrator:easy"]["mean"] != 0.5

    def test_normalize_per_role_eval_mode_does_not_update(self):
        n = RewardNormalizer()
        n.load_snapshot(
            {
                "orchestrator:easy": {"count": 40, "mean": 0.5, "m2": 20.0},
                "floor_agent:easy": {"count": 40, "mean": 0.0, "m2": 20.0},
            }
        )
        rewards = RewardsByRole(
            orchestrator=RoleReward(raw=1.0, normalized=0.0),
            floors={"floor_0_agent": RoleReward(raw=0.0, normalized=0.0)},
        )

        snapshot_before = n.snapshot()
        normalized = normalize_per_role(rewards, "easy", n, update=False)

        expected = max(-1.0, min(1.0, (1.0 - 0.5) / math.sqrt(20.0 / 40.0)))
        assert normalized["orchestrator"] == expected
        assert n.snapshot() == snapshot_before

    def test_update_false_does_not_change_state(self):
        """update=False in normalize_per_role does not change internal state."""
        n = RewardNormalizer()
        # Seed with some data
        for i in range(10):
            n.update("orchestrator", "easy", 1.0)
            n.update("floor_agent", "easy", 0.5)

        snap_before = n.snapshot()

        # Call with update=False
        rewards = RewardsByRole(
            orchestrator=RoleReward(raw=0.9, normalized=0.0),
            floors={
                "floor_0_agent": RoleReward(raw=0.4, normalized=0.0),
                "floor_1_agent": RoleReward(raw=0.3, normalized=0.0),
            },
        )
        normalize_per_role(rewards, "easy", n, update=False)

        snap_after = n.snapshot()
        assert snap_before == snap_after

    def test_training_normalizer_scope_differs_from_env_reward_pipeline(self):
        pipeline = RewardPipeline()
        pipeline.observe(1.0, "easy")
        pipeline.observe(2.0, "hard")

        n = RewardNormalizer()
        rewards = RewardsByRole(
            orchestrator=RoleReward(raw=1.0, normalized=0.0),
            floors={
                "floor_0_agent": RoleReward(raw=0.4, normalized=0.0),
                "floor_1_agent": RoleReward(raw=0.2, normalized=0.0),
            },
        )
        normalize_per_role(rewards, "easy", n, update=True)

        assert set(pipeline.snapshot()) == {"easy", "hard"}
        snapshot = n.snapshot()
        assert set(snapshot) == {"floor_agent:easy", "orchestrator:easy"}
        assert snapshot["floor_agent:easy"]["count"] == 2
        assert "floor_0_agent:easy" not in snapshot
