"""Training configuration schema.

Uses Pydantic v2 with ``extra="forbid"`` and validates config constraints.
Heavy-dependency-free (no pyyaml at import time — caller provides a dict).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# EVAL_SEEDS from curriculum — used for validation
_EVAL_SEEDS_SET = frozenset({42, 123, 456, 789, 1024})
_ROLE_NAMES = ("orchestrator", "floor_agent")
RoleName = Literal["orchestrator", "floor_agent"]
FloorSpecialistFamily = Literal["fire", "flood", "gas"]
TierName = Literal["easy"]
_FLOOR_SPECIALIST_FAMILIES = frozenset({"fire", "flood", "gas"})
_TIER_NAMES = frozenset({"easy"})
_FAMILY_ALIASES = {
    "gas_leak": "gas",
    "gasleak": "gas",
    "smoke": "fire",
    "water": "flood",
}


def _normalize_family_name(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        value = enum_value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _FAMILY_ALIASES.get(text, text)


class RolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainable: list[RoleName] = Field(
        default_factory=lambda: ["orchestrator", "floor_agent"]
    )
    orchestrator_policy: Literal["model", "stub"] = "model"
    frozen_adapter_paths: dict[RoleName, str] = Field(default_factory=dict)
    frozen_floor_specialist_adapter_paths: dict[FloorSpecialistFamily, str] = Field(
        default_factory=dict,
        description=(
            "Frozen floor-agent LoRA adapters keyed by disaster family. "
            "Used for phase-2 7B orchestrator training over fire/flood/gas specialists."
        ),
    )

    @field_validator("trainable")
    @classmethod
    def trainable_roles_must_be_nonempty_and_unique(
        cls,
        value: list[RoleName],
    ) -> list[RoleName]:
        if not value:
            raise ValueError("roles.trainable must contain at least one role")
        duplicates = [
            role
            for index, role in enumerate(value)
            if role in value[:index]
        ]
        if duplicates:
            raise ValueError(
                f"roles.trainable contains duplicate roles: {sorted(set(duplicates))!r}"
            )
        return value

    @property
    def trainable_set(self) -> set[RoleName]:
        return set(self.trainable)

    def is_trainable(self, role: RoleName) -> bool:
        return role in self.trainable_set

    def frozen_adapter_path_for_role(self, role: RoleName) -> str | None:
        return self.frozen_adapter_paths.get(role)

    def frozen_floor_specialist_adapter_path_for_family(
        self,
        family: FloorSpecialistFamily,
    ) -> str | None:
        return self.frozen_floor_specialist_adapter_paths.get(family)

    def policy_for_role(self, role: RoleName) -> Literal["model", "stub"]:
        if role == "orchestrator":
            return self.orchestrator_policy
        if role == "floor_agent":
            return "model"
        raise ValueError(f"Unknown role {role!r}")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str = "Qwen/Qwen2.5-3B-Instruct"
    orchestrator_base: str | None = None
    floor_base: str | None = None
    dtype: str = "bfloat16"
    max_prompt_tokens: int = 3500
    max_completion_tokens: int = 256

    @field_validator("dtype")
    @classmethod
    def dtype_must_be_known(cls, value: str) -> str:
        allowed = {"bfloat16", "float16", "float32"}
        if value not in allowed:
            raise ValueError(
                f"model.dtype must be one of {sorted(allowed)}; got {value!r}"
            )
        return value

    def resolved_base_for_role(self, role: Literal["orchestrator", "floor_agent"]) -> str:
        if role == "orchestrator":
            return self.orchestrator_base or self.base
        if role == "floor_agent":
            return self.floor_base or self.base
        raise ValueError(f"Unknown role {role!r}")

    def resolved_bases(self) -> dict[str, str]:
        return {
            "orchestrator": self.resolved_base_for_role("orchestrator"),
            "floor_agent": self.resolved_base_for_role("floor_agent"),
        }

    @property
    def uses_split_bases(self) -> bool:
        resolved = self.resolved_bases()
        return resolved["orchestrator"] != resolved["floor_agent"]


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class TierScheduleBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int
    mix: dict[TierName, int]

    @field_validator("steps")
    @classmethod
    def steps_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rollout.tier_schedule[].steps must be > 0")
        return value

    @field_validator("mix")
    @classmethod
    def mix_must_be_nonempty_positive(cls, value: dict[TierName, int]) -> dict[TierName, int]:
        if not value:
            raise ValueError("rollout.tier_schedule[].mix must not be empty")
        for tier, count in value.items():
            if tier not in _TIER_NAMES:
                raise ValueError(f"unknown tier {tier!r}")
            if count <= 0:
                raise ValueError(
                    f"rollout.tier_schedule[].mix[{tier!r}] must be > 0"
                )
        return value

    @model_validator(mode="after")
    def mix_total_must_match_steps(self) -> "TierScheduleBlock":
        total = sum(self.mix.values())
        if total != self.steps:
            raise ValueError(
                "rollout.tier_schedule block mix counts must sum to steps; "
                f"got mix total {total} for steps {self.steps}"
            )
        return self

    def expanded_tiers(self) -> list[TierName]:
        total = sum(self.mix.values())
        ordered_tiers = [tier for tier, count in self.mix.items() if count > 0]
        weights = dict(self.mix)
        remaining = dict(self.mix)
        current = {tier: 0 for tier in ordered_tiers}
        tiers: list[TierName] = []

        # Smooth weighted round-robin keeps any future replay buckets interleaved
        # instead of clumping repeated samples at the start of a stage.
        for _ in range(total):
            active = [tier for tier in ordered_tiers if remaining[tier] > 0]
            for tier in active:
                current[tier] += weights[tier]
            chosen = max(
                active,
                key=lambda tier: (current[tier], weights[tier], -ordered_tiers.index(tier)),
            )
            tiers.append(chosen)
            current[chosen] -= total
            remaining[chosen] -= 1

        return tiers


class RolloutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes_per_step: int = 4
    max_rounds_per_episode: int = 80
    seed_retry_limit: int = 1000
    use_vllm: bool = True
    candidates_per_floor_prompt: int = Field(
        default=1,
        description=(
            "Training-only number of sampled floor-agent completions per identical "
            "floor prompt. Values >1 give GRPO same-prompt candidate contrast."
        ),
    )
    include_oracle_floor_candidate: bool = Field(
        default=False,
        description=(
            "Training-only bootstrap: append one exact-ID floor expert action to each "
            "candidate group so GRPO has a valid positive completion to compare against."
        ),
    )
    sampling_temperature: float = Field(
        default=0.7,
        description=(
            "Policy generation temperature. Keep >0 when "
            "candidates_per_floor_prompt >1 so candidate groups are not deterministic."
        ),
    )
    disaster_families: list[str] = Field(
        default_factory=lambda: ["fire", "flood", "gas"]
    )
    tier_schedule: list[TierScheduleBlock] | None = None

    @field_validator("candidates_per_floor_prompt")
    @classmethod
    def candidates_per_floor_prompt_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rollout.candidates_per_floor_prompt must be > 0")
        return value

    @field_validator("sampling_temperature")
    @classmethod
    def sampling_temperature_must_be_nonnegative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("rollout.sampling_temperature must be >= 0")
        return value

    @field_validator("seed_retry_limit")
    @classmethod
    def seed_retry_limit_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rollout.seed_retry_limit must be > 0")
        return value

    def expanded_tier_schedule(self) -> list[TierName]:
        if not self.tier_schedule:
            return []
        tiers: list[TierName] = []
        for block in self.tier_schedule:
            tiers.extend(block.expanded_tiers())
        return tiers


class GRPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 5e-6
    kl_coef: float = 0.04
    clip_range: float = 0.2
    num_train_epochs_per_step: int = 1
    prefer_trl: bool = False


class WatchdogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    warmup_steps: int = 10
    zero_signal_window: int = 5
    min_advantage_std: float = 1e-6
    min_group_raw_reward_std_mean: float = 1e-6
    max_valid_but_hollow_action_rate: float = 0.7
    max_floor_scout_action_rate: float = 0.8
    min_floor_route_action_rate: float = 0.05
    route_required_after_steps: int = 25
    abort_on_trigger: bool = True

    @field_validator(
        "warmup_steps",
        "zero_signal_window",
        "route_required_after_steps",
    )
    @classmethod
    def nonnegative_integer_thresholds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("watchdog integer thresholds must be >= 0")
        return value

    @field_validator(
        "min_advantage_std",
        "min_group_raw_reward_std_mean",
        "max_valid_but_hollow_action_rate",
        "max_floor_scout_action_rate",
        "min_floor_route_action_rate",
    )
    @classmethod
    def nonnegative_float_thresholds(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("watchdog float thresholds must be >= 0")
        return value


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale_scaling: str = "linear_capped"
    alpha: float = 0.01
    beta: float = 0.25
    cap: float = 1.0
    eligible_token_ceiling: int = 160
    clip_normalized_to: float = 1.0

    @field_validator("rationale_scaling")
    @classmethod
    def rationale_scaling_must_be_known(cls, value: str) -> str:
        allowed = {"off", "linear_capped", "log_uncapped"}
        if value not in allowed:
            raise ValueError(
                f"reward.rationale_scaling must be one of {sorted(allowed)}; got {value!r}"
            )
        return value


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every_steps: int = 20
    tiers: list[TierName] = Field(default_factory=lambda: ["easy"])
    seeds: list[int] = Field(default_factory=lambda: [42, 123, 456, 789, 1024])

    @field_validator("seeds")
    @classmethod
    def seeds_must_be_eval_seeds(cls, v: list[int]) -> list[int]:
        invalid = set(v) - _EVAL_SEEDS_SET
        if invalid:
            raise ValueError(
                f"Eval seeds {invalid} are not in curriculum.EVAL_SEEDS {_EVAL_SEEDS_SET}"
            )
        return v


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every_steps: int = 10
    keep_last_n: int = 5
    root_dir: str = "outputs/training/checkpoints"


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csv_path: str = "outputs/training/metrics.csv"
    jsonl_dir: str = "outputs/logs"


class SeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_rng: int = 12345


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Backend selector for the training path.
    #   "hf"      — HuggingFace TRL + transformers + PEFT (default; Windows + Colab).
    #   "unsloth" — Unsloth quantized kernels; Colab/Linux+CUDA only. Opt-in.
    backend: Literal["hf", "unsloth"] = "unsloth"
    # Unsloth-specific knobs (ignored when backend == "hf").
    unsloth_max_seq_length: int = 4096
    load_in_4bit: bool = True

    max_steps: int | None = Field(
        default=None,
        description="Hard cap on GRPO steps; None means run until SIGTERM / KeyboardInterrupt.",
    )

    @field_validator("max_steps")
    @classmethod
    def max_steps_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_steps must be positive when set")
        return v

    model: ModelConfig = Field(default_factory=ModelConfig)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    seed: SeedConfig = Field(default_factory=SeedConfig)

    @property
    def trainable_roles(self) -> tuple[RoleName, ...]:
        return tuple(self.roles.trainable)

    def is_role_trainable(self, role: RoleName) -> bool:
        return self.roles.is_trainable(role)

    def policy_for_role(self, role: RoleName) -> Literal["model", "stub"]:
        return self.roles.policy_for_role(role)

    def frozen_adapter_path_for_role(self, role: RoleName) -> str | None:
        return self.roles.frozen_adapter_path_for_role(role)

    def frozen_floor_specialist_adapter_path_for_family(
        self,
        family: FloorSpecialistFamily,
    ) -> str | None:
        return self.roles.frozen_floor_specialist_adapter_path_for_family(family)

    @property
    def uses_role_routed_policy(self) -> bool:
        return (
            self.model.uses_split_bases
            or self.roles.orchestrator_policy != "model"
            or bool(self.roles.frozen_adapter_paths)
            or bool(self.roles.frozen_floor_specialist_adapter_paths)
            or self.roles.trainable_set != set(_ROLE_NAMES)
        )

    @model_validator(mode="after")
    def vllm_requires_unsloth_backend(self) -> "TrainingConfig":
        if self.rollout.use_vllm and self.backend != "unsloth":
            raise ValueError(
                f"rollout.use_vllm=true requires backend='unsloth' "
                f"(current backend={self.backend!r}); the HF backend has no vLLM path. "
                f"Either set backend='unsloth' or leave rollout.use_vllm=false."
            )
        trainable_roles = self.roles.trainable_set
        if self.roles.orchestrator_policy == "stub" and "orchestrator" in trainable_roles:
            raise ValueError(
                "roles.orchestrator_policy='stub' is incompatible with "
                "roles.trainable including 'orchestrator'"
            )
        for role, adapter_path in self.roles.frozen_adapter_paths.items():
            if role in trainable_roles:
                raise ValueError(
                    "roles.frozen_adapter_paths may only target non-trainable "
                    f"roles; got frozen path for trainable role {role!r}"
                )
            if self.policy_for_role(role) != "model":
                raise ValueError(
                    "roles.frozen_adapter_paths may only target model-backed "
                    f"roles; got {role!r} with policy {self.policy_for_role(role)!r}"
                )
            if not adapter_path.strip():
                raise ValueError(
                    f"roles.frozen_adapter_paths[{role!r}] must be a non-empty path"
                )
        specialist_paths = self.roles.frozen_floor_specialist_adapter_paths
        if specialist_paths:
            if "floor_agent" in trainable_roles:
                raise ValueError(
                    "roles.frozen_floor_specialist_adapter_paths requires "
                    "roles.trainable to exclude 'floor_agent'; floor specialists "
                    "must be frozen while the orchestrator trains."
                )
            for family, adapter_path in specialist_paths.items():
                if not adapter_path.strip():
                    raise ValueError(
                        "roles.frozen_floor_specialist_adapter_paths"
                        f"[{family!r}] must be a non-empty path"
                    )

            configured_families = set(specialist_paths)
            requested_families = {
                _normalize_family_name(family)
                for family in self.rollout.disaster_families
            }
            specialist_requested = requested_families & _FLOOR_SPECIALIST_FAMILIES
            missing_specialists = specialist_requested - configured_families
            has_generalist_fallback = "floor_agent" in self.roles.frozen_adapter_paths
            if missing_specialists and not has_generalist_fallback:
                raise ValueError(
                    "rollout.disaster_families includes specialist disaster(s) "
                    f"{sorted(missing_specialists)!r}, but no matching "
                    "roles.frozen_floor_specialist_adapter_paths entry or "
                    "roles.frozen_adapter_paths.floor_agent fallback was provided."
                )

            unsupported_requested = requested_families - _FLOOR_SPECIALIST_FAMILIES
            if unsupported_requested and not has_generalist_fallback:
                raise ValueError(
                    "scope-routed floor specialists only cover fire/flood/gas. "
                    f"Unsupported rollout.disaster_families={sorted(unsupported_requested)!r} "
                    "require roles.frozen_adapter_paths.floor_agent as a "
                    "generalist fallback."
                )
        tier_schedule = self.rollout.expanded_tier_schedule()
        if tier_schedule:
            if self.max_steps is not None and len(tier_schedule) != self.max_steps:
                raise ValueError(
                    "rollout.tier_schedule must expand to max_steps when max_steps is set; "
                    f"got {len(tier_schedule)} scheduled steps for max_steps={self.max_steps}"
                )
        return self
