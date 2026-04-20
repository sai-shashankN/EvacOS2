"""Tests for the Phase 7 Unsloth training backend.

All tests MUST pass with the current Windows environment where neither
``unsloth`` nor ``vllm`` is installed. The backend is additive and
strictly opt-in via ``TrainingConfig.backend = "unsloth"``.
"""

from __future__ import annotations

from unittest import mock

import pytest

from training.config_schema import TrainingConfig
from training.policy_adapter import StubPolicy, unsloth_policy_factory


# ---------------------------------------------------------------------------
# 1. Config schema
# ---------------------------------------------------------------------------


class TestConfigSchemaBackend:
    def test_default_backend_is_hf(self):
        """Default backend is 'hf' — no silent regression for existing users."""
        config = TrainingConfig()
        assert config.backend == "hf"
        assert config.rollout.use_vllm is False
        assert config.unsloth_max_seq_length == 4096
        assert config.load_in_4bit is True

    def test_backend_accepts_unsloth(self):
        """backend='unsloth' validates and round-trips through model_dump."""
        config = TrainingConfig(backend="unsloth")
        assert config.backend == "unsloth"
        dumped = config.model_dump(mode="json")
        assert dumped["backend"] == "unsloth"

    def test_backend_rejects_unknown(self):
        """Unknown backend strings are rejected at construction time."""
        with pytest.raises(Exception):
            TrainingConfig(backend="llamafile")


# ---------------------------------------------------------------------------
# 2. Factory raises RuntimeError when unsloth is absent
# ---------------------------------------------------------------------------


class TestUnslothFactoryGuards:
    def test_factory_raises_when_unsloth_missing(self):
        """On Windows / bare Linux without unsloth, the factory must raise a
        RuntimeError whose message contains the string 'unsloth'."""
        with pytest.raises(RuntimeError) as excinfo:
            unsloth_policy_factory("Qwen/Qwen2.5-1.5B-Instruct")
        assert "unsloth" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 3. Policy protocol: act_batch is optional
# ---------------------------------------------------------------------------


class TestPolicyProtocolOptionalBatch:
    def test_stub_policy_does_not_expose_act_batch(self):
        """StubPolicy (and hf_policy_factory) do NOT add act_batch; the
        rollout's hasattr-based fast-path detection must degrade gracefully."""
        policy = StubPolicy(seed=0)
        assert not hasattr(policy, "act_batch")


# ---------------------------------------------------------------------------
# 4 & 5. Rollout: fallback vs fast path
# ---------------------------------------------------------------------------


class _MockPolicyActOnly:
    """A policy that only implements .act — no act_batch. Must still work."""

    def __init__(self):
        self.act_calls: list[tuple[str, str]] = []

    def act(self, prompt, agent_id, role):
        self.act_calls.append((agent_id, role))
        # Emit a wait action that parses as valid JSON.
        import json
        return json.dumps(
            {
                "episode_id": "ep_mock",
                "round_id": 0,
                "agent_id": agent_id,
                "action_id": f"a_{agent_id}",
                "action_type": "wait",
                "arguments": {},
            }
        )


class _MockPolicyWithBatch(_MockPolicyActOnly):
    """A policy that implements both act and act_batch. Rollout should prefer
    act_batch: .act must never be called when act_batch is present."""

    def __init__(self):
        super().__init__()
        self.batch_calls: list[int] = []

    def act_batch(self, prompts, agent_ids, roles):
        assert len(prompts) == len(agent_ids) == len(roles)
        self.batch_calls.append(len(prompts))
        # Return one wait-action JSON per prompt.
        import json
        return [
            json.dumps(
                {
                    "episode_id": "ep_mock",
                    "round_id": 0,
                    "agent_id": aid,
                    "action_id": f"a_{aid}",
                    "action_type": "wait",
                    "arguments": {},
                }
            )
            for aid in agent_ids
        ]


def _run_one_rollout_episode(policy, max_rounds=3):
    """Drive collect_episode for a short rollout against a real EvacEnvironment.

    Returns (episode_result, policy_handle).
    """
    from evacos_ma.env import EvacEnvironment
    from evacos_ma.models import DisasterType

    from training.rollout import collect_episode

    env = EvacEnvironment()
    result = collect_episode(
        env,
        policy,
        seed=7,
        tier="easy",
        disaster_family=DisasterType.fire,
        max_rounds=max_rounds,
    )
    return result


class TestRolloutFastPath:
    def test_fallback_policy_without_act_batch_runs_cleanly(self, tmp_path, monkeypatch):
        """Rollout must continue to work against the per-agent .act loop when
        no act_batch method is present. Regression check for Phase 7.
        """
        # Re-home JSONL outputs under tmp_path to avoid polluting outputs/logs.
        from training import rollout as rollout_mod
        monkeypatch.setattr(
            rollout_mod,
            "write_trace_row",
            lambda *a, **kw: None,
        )

        policy = _MockPolicyActOnly()
        assert not hasattr(policy, "act_batch")

        result = _run_one_rollout_episode(policy, max_rounds=2)

        # Sanity: at least one round ran (>= 1 samples for orch and 5 floors)
        assert result.num_rounds >= 1
        # At least 6 * num_rounds .act calls should have been issued.
        assert len(policy.act_calls) >= 6 * result.num_rounds

    def test_fast_path_uses_act_batch_once_per_round(self, monkeypatch):
        """When act_batch is present, the rollout must call it exactly once
        per round (1 call, 6 prompts) instead of 6x .act per round."""
        from training import rollout as rollout_mod
        monkeypatch.setattr(
            rollout_mod,
            "write_trace_row",
            lambda *a, **kw: None,
        )

        policy = _MockPolicyWithBatch()
        assert hasattr(policy, "act_batch")

        result = _run_one_rollout_episode(policy, max_rounds=2)

        # Exactly one act_batch call per round, each with 6 prompts
        # (1 orchestrator + 5 floor agents).
        assert len(policy.batch_calls) == result.num_rounds
        assert all(count == 6 for count in policy.batch_calls)
        # Critically: .act was never called — fast path dominated.
        assert policy.act_calls == []
