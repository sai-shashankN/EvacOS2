"""training — EvacOS-MA multi-agent training pipeline.

This package intentionally avoids importing rollout / reward modules at package
import time so ``python -m training.train`` can control heavy dependency import
order (notably Unsloth before transformers / TRL / PEFT).
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "EpisodeRolloutResult",
    "Policy",
    "TrajectorySample",
    "collect_batch",
    "collect_episode",
    "RewardNormalizer",
    "TierNormalizerState",
    "normalize_per_role",
]


def __getattr__(name: str):
    if name in {
        "PROMPT_TEMPLATE_VERSION",
        "EpisodeRolloutResult",
        "Policy",
        "StubPolicy",
        "TrajectorySample",
        "collect_batch",
        "collect_episode",
    }:
        module = import_module("training.rollout")
        return getattr(module, name)

    if name in {
        "RewardNormalizer",
        "TierNormalizerState",
        "normalize_per_role",
    }:
        module = import_module("training.reward")
        return getattr(module, name)

    raise AttributeError(f"module 'training' has no attribute {name!r}")
