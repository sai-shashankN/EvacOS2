"""Tests for the multi-role rollout collector."""

import json
import os
import random
import shutil
import tempfile
from pathlib import Path

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from curriculum.controller import CurriculumController, EVAL_SEEDS

from training.policy_adapter import StubPolicy
from training.reward import RewardNormalizer
from training.rollout import collect_batch, collect_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env() -> EvacEnvironment:
    return EvacEnvironment()


def _make_policy(seed: int = 0) -> StubPolicy:
    return StubPolicy(seed=seed)


def _tmp_logs_dir() -> Path:
    root = Path(tempfile.gettempdir()) / f"evacos_rollout_logs_{os.getpid()}_{random.randint(0, 99999)}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectEpisode:
    def test_stub_policy_easy_fire_produces_nonempty_result(self):
        """One StubPolicy episode on easy/fire produces a non-empty EpisodeRolloutResult
        with 6 samples per round (5 floors + 1 orch)."""
        env = _make_env()
        policy = _make_policy(seed=42)
        result = collect_episode(
            env,
            policy,
            seed=10,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=3,
        )
        assert result.episode_id
        assert result.num_rounds > 0
        assert result.num_rounds <= 3
        # 6 samples per round (5 floors + 1 orchestrator)
        assert len(result.samples) == result.num_rounds * 6
        assert result.seed == 10
        assert result.tier == "easy"
        assert result.disaster_family == "fire"
        assert result.wall_clock_seconds > 0

    def test_group_id_format(self):
        """group_id uses role-specific semantics: orchestrator is batch-wide, floor is per-(episode, round)."""
        env = _make_env()
        policy = _make_policy(seed=1)
        result = collect_episode(
            env,
            policy,
            seed=20,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=2,
        )
        for sample in result.samples:
            if sample.role == "orchestrator":
                assert sample.group_id == "rollout_orchestrator"
            else:
                assert sample.group_id.startswith("ep_")
                assert "_r_" in sample.group_id
                assert sample.group_id.endswith("_floor")


class TestCollectBatchEvalSeeds:
    def test_eval_seeds_collision_skipped_in_training(self):
        """Training mode raises once retry budget is exhausted on EVAL_SEEDS collisions."""
        env = _make_env()
        policy = _make_policy(seed=99)
        curriculum = CurriculumController()
        try:
            collect_batch(
                env,
                policy,
                curriculum,
                num_episodes=1,
                seed_generator=lambda: 42,
                disaster_families=[DisasterType.fire],
                max_rounds=1,
                is_eval=False,
                seed_collision_retry_limit=3,
            )
        except RuntimeError as exc:
            assert "EVAL_SEEDS" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError on training/eval seed collision exhaustion")

    def test_eval_batch_does_not_mutate_reward_normalizer(self):
        env = _make_env()
        policy = _make_policy(seed=5)
        curriculum = CurriculumController()
        normalizer = RewardNormalizer()
        for idx in range(40):
            normalizer.update("orchestrator", "easy", 0.1 * (idx % 4))
            normalizer.update("floor_agent", "easy", 0.05 * (idx % 3))

        snapshot_before = normalizer.snapshot()
        logs_dir = _tmp_logs_dir()
        try:
            seed_values = iter([42, 123, 456])
            collect_batch(
                env,
                policy,
                curriculum,
                num_episodes=3,
                seed_generator=lambda: next(seed_values),
                disaster_families=[DisasterType.fire],
                max_rounds=1,
                is_eval=True,
                normalizer=normalizer,
                jsonl_dir=logs_dir,
            )
            assert snapshot_before == normalizer.snapshot()
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)

    def test_stub_rollout_emits_required_trace_artifacts_with_common_fields(self):
        env = _make_env()
        policy = _make_policy(seed=13)
        logs_dir = _tmp_logs_dir()
        try:
            result = collect_episode(
                env,
                policy,
                seed=99,
                tier="easy",
                disaster_family=DisasterType.fire,
                max_rounds=1,
                jsonl_dir=logs_dir,
            )
            assert result.num_rounds == 1

            required = [
                "round_trace.jsonl",
                "action_trace.jsonl",
                "reward_trace.jsonl",
                "episode_summary.jsonl",
            ]
            common_fields = {
                "episode_id",
                "round_id",
                "seed",
                "tier",
                "disaster_family",
                "trace_schema_version",
                "generator_config_hash",
                "reward_schema_version",
                "prompt_template_version",
                "model_name",
                "checkpoint_tag",
            }
            for name in required:
                path = logs_dir / name
                assert path.exists()
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                assert rows
                for row in rows:
                    assert common_fields.issubset(row.keys())
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)


class TestParseFallback:
    def test_invalid_completion_produces_fallback(self):
        """Invalid completion → sample emitted with parsed_action['fallback_reason'] == 'parse_error'
        and episode does not crash."""

        class BadPolicy:
            def act(self, prompt, agent_id, role):
                return "THIS IS NOT VALID JSON !!!"

        env = _make_env()
        result = collect_episode(
            env,
            BadPolicy(),
            seed=30,
            tier="easy",
            disaster_family=DisasterType.fire,
            max_rounds=2,
        )
        assert result.num_rounds > 0
        assert len(result.samples) > 0
        # All samples should have fallback_reason since all completions are bad
        fallback_count = sum(
            1 for s in result.samples
            if s.parsed_action.get("fallback_reason") == "parse_error"
        )
        assert fallback_count > 0
        # Every sample should be present (no crash)
        assert len(result.samples) == result.num_rounds * 6
