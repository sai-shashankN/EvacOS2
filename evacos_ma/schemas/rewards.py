"""EvacOS-MA Reward Schema Types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Version constant
REWARD_SCHEMA_VERSION: str = "v1"


class RewardBreakdown(BaseModel):
    """Arbitrary reward component dict + schema version."""
    model_config = ConfigDict(extra="allow")

    reward_schema_version: str = REWARD_SCHEMA_VERSION

    def get_components(self) -> dict[str, float]:
        """Return only the numeric reward components (excluding metadata)."""
        return {k: v for k, v in self.model_dump().items() if k != "reward_schema_version" and isinstance(v, (int, float))}


class RoleReward(BaseModel):
    """Per-role reward envelope: raw, normalized, breakdown, version."""
    model_config = ConfigDict(extra="forbid")

    raw: float = 0.0
    normalized: float = 0.0
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    reward_schema_version: str = REWARD_SCHEMA_VERSION


class RewardsByRole(BaseModel):
    """Reward for orchestrator + per-floor-agent rewards."""
    model_config = ConfigDict(extra="forbid")

    orchestrator: RoleReward = Field(default_factory=RoleReward)
    floors: dict[str, RoleReward] = Field(default_factory=dict)
