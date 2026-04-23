from __future__ import annotations

"""Single contract surface between evaluation/ and the training package."""

from training import (
    policy_adapter as _policy_adapter,
    reward as _reward,
    rollout as _rollout,
    scope_router as _scope_router,
)

Policy = _policy_adapter.Policy
RoleRoutedPolicy = _policy_adapter.RoleRoutedPolicy
StubPolicy = _policy_adapter.StubPolicy
hf_policy_factory = _policy_adapter.hf_policy_factory
RewardNormalizer = _reward.RewardNormalizer
collect_batch = _rollout.collect_batch
ScopeDecision = _scope_router.ScopeDecision
route_scope = _scope_router.route_scope

__all__ = [
    "Policy",
    "RoleRoutedPolicy",
    "RewardNormalizer",
    "ScopeDecision",
    "collect_batch",
    "StubPolicy",
    "hf_policy_factory",
    "route_scope",
]
