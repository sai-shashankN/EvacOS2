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


class RolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainable: list[RoleName] = Field(
        default_factory=lambda: ["orchestrator", "floor_agent"]
    )
    orchestrator_policy: Literal["model", "stub"] = "model"

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


class RolloutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes_per_step: int = 4
    max_rounds_per_episode: int = 80
    seed_retry_limit: int = 1000
    use_vllm: bool = True
    disaster_families: list[str] = Field(
        default_factory=lambda: [
            "fire", "flood", "gas", "structural",
            "active_threat", "multi_cascade",
        ]
    )

    @field_validator("seed_retry_limit")
    @classmethod
    def seed_retry_limit_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rollout.seed_retry_limit must be > 0")
        return value


class GRPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 5e-6
    kl_coef: float = 0.04
    clip_range: float = 0.2
    num_train_epochs_per_step: int = 1
    prefer_trl: bool = False


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
    tiers: list[str] = Field(default_factory=lambda: ["easy", "medium"])
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

    @property
    def uses_role_routed_policy(self) -> bool:
        return self.model.uses_split_bases or self.roles.orchestrator_policy != "model"

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
        if (
            trainable_roles != set(_ROLE_NAMES)
            and self.roles.orchestrator_policy != "stub"
        ):
            raise ValueError(
                "Selective roles.trainable currently requires "
                "roles.orchestrator_policy='stub'; model-backed selective-role "
                "training is not supported yet."
            )
        return self
