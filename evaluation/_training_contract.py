from __future__ import annotations

"""Single contract surface between evaluation/ and the training package."""

from training import policy_adapter as _policy_adapter, reward as _reward, rollout as _rollout

Policy = _policy_adapter.Policy
StubPolicy = _policy_adapter.StubPolicy
hf_policy_factory = _policy_adapter.hf_policy_factory
RewardNormalizer = _reward.RewardNormalizer
collect_batch = _rollout.collect_batch

__all__ = [
    "Policy",
    "RewardNormalizer",
    "collect_batch",
    "StubPolicy",
    "hf_policy_factory",
]
