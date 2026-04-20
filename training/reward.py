"""Per-role reward normalization with Welford running statistics.

Heavy-dependency-free.  Uses only stdlib + numpy (already in requirements).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from evacos_ma.schemas.rewards import RewardsByRole


# ---------------------------------------------------------------------------
# Tier normalizer state (Welford)
# ---------------------------------------------------------------------------


@dataclass
class TierNormalizerState:
    """Per (role, tier) running stats for z-score normalization."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Welford's M2 accumulator

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / self.count

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


# ---------------------------------------------------------------------------
# Reward normalizer
# ---------------------------------------------------------------------------

_MIN_SAMPLES_FOR_ZSCORE: int = 30
_STDDEV_FLOOR: float = 1e-6


class RewardNormalizer:
    """Maintains Welford running statistics keyed by (role, tier)."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], TierNormalizerState] = {}

    # -- internal helpers --------------------------------------------------

    def _get_state(self, role: str, tier: str) -> TierNormalizerState:
        key = (role, tier)
        if key not in self._states:
            self._states[key] = TierNormalizerState()
        return self._states[key]

    # -- public API --------------------------------------------------------

    def update(self, role: str, tier: str, raw: float) -> None:
        """Incorporate *raw* into the running Welford stats for (role, tier)."""
        state = self._get_state(role, tier)
        state.count += 1
        delta = raw - state.mean
        state.mean += delta / state.count
        delta2 = raw - state.mean
        state.m2 += delta * delta2

    def normalize(
        self,
        role: str,
        tier: str,
        raw: float,
        *,
        clip: float = 1.0,
    ) -> float:
        """Return z-scored + clipped reward.  Falls back to tanh if insufficient data."""
        state = self._get_state(role, tier)
        if state.count < _MIN_SAMPLES_FOR_ZSCORE:
            return max(-clip, min(clip, math.tanh(raw)))
        if state.stddev < _STDDEV_FLOOR:
            return max(-clip, min(clip, math.tanh(raw)))
        z = (raw - state.mean) / state.stddev
        return max(-clip, min(clip, z))

    # -- snapshot / restore ------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all normalizer state."""
        out: dict = {}
        for (role, tier), state in sorted(self._states.items()):
            out[f"{role}:{tier}"] = {
                "count": state.count,
                "mean": state.mean,
                "m2": state.m2,
            }
        return out

    def load_snapshot(self, data: dict) -> None:
        """Restore normalizer state from a snapshot dict."""
        self._states.clear()
        for key_str, vals in data.items():
            parts = key_str.split(":", 1)
            if len(parts) != 2:
                continue
            role, tier = parts
            self._states[(role, tier)] = TierNormalizerState(
                count=vals["count"],
                mean=vals["mean"],
                m2=vals["m2"],
            )


# ---------------------------------------------------------------------------
# Convenience: normalize a full RewardsByRole envelope
# ---------------------------------------------------------------------------


def normalize_per_role(
    rewards_by_role: RewardsByRole,
    tier: str,
    normalizer: RewardNormalizer,
    *,
    update: bool = True,
) -> dict[str, float]:
    """Return ``{agent_id: normalized_reward}`` for orchestrator + floor agents.

    Parameters
    ----------
    update:
        When *True* (training mode), feed raw rewards into the normalizer before
        computing the normalized value.  When *False* (eval mode), only query
        without contaminating training stats.
    """
    result: dict[str, float] = {}

    orch_raw = rewards_by_role.orchestrator.raw
    if update:
        normalizer.update("orchestrator", tier, orch_raw)
    orch_norm = normalizer.normalize("orchestrator", tier, orch_raw)
    result["orchestrator"] = orch_norm

    for agent_id, role_reward in rewards_by_role.floors.items():
        floor_raw = role_reward.raw
        if update:
            normalizer.update("floor_agent", tier, floor_raw)
        floor_norm = normalizer.normalize("floor_agent", tier, floor_raw)
        result[agent_id] = floor_norm

    return result
