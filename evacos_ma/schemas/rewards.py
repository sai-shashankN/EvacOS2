"""EvacOS-MA Reward Schema Types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_serializer

# Version constant
REWARD_SCHEMA_VERSION: str = "v1"


class RewardBreakdown(BaseModel):
    """Arbitrary reward component dict + schema version."""
    model_config = ConfigDict(extra="allow")

    reward_schema_version: str = REWARD_SCHEMA_VERSION
    base_sim_reward: float = 0.0
    base_sim_reward_share: float = 0.0
    team_progress_dense: float = 0.0
    floor_saved: float = 0.0
    floor_lost: float = 0.0
    floor_invalid_action: float = 0.0
    total_saved_terminal: float = 0.0
    total_lost_terminal: float = 0.0
    coordination_bonus: float = 0.0
    directive_quality: float = 0.0
    priority_top_match: float = 0.0
    priority_rank_score: float = 0.0
    priority_coverage: float = 0.0
    priority_duplicate_or_unknown_penalty: float = 0.0
    priority_effect_bonus: float = 0.0
    rationale_bonus: float = 0.0

    @model_serializer(mode="plain")
    def _serialize(self) -> dict:
        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__class__.model_fields
        }
        if self.__pydantic_extra__:
            payload.update(self.__pydantic_extra__)
        sparse = {
            "reward_schema_version": payload.get(
                "reward_schema_version", REWARD_SCHEMA_VERSION
            )
        }
        for key, value in payload.items():
            if key == "reward_schema_version":
                continue
            if key in self.model_fields_set or value != 0.0:
                sparse[key] = value
        return sparse

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
