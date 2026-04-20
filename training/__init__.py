"""training — EvacOS-MA multi-agent training pipeline.

All heavy dependencies (torch, transformers, trl, peft, accelerate, bitsandbytes)
are import-guarded so this package compiles without them.  Only the Colab notebook
and training.train.run_training() actually require them at runtime.
"""

from training.rollout import (
    PROMPT_TEMPLATE_VERSION,
    EpisodeRolloutResult,
    Policy,
    StubPolicy,
    TrajectorySample,
    collect_batch,
    collect_episode,
)
from training.reward import (
    RewardNormalizer,
    TierNormalizerState,
    normalize_per_role,
)

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
