"""Training configuration schema.

Uses Pydantic v2 with ``extra="forbid"`` and validates config constraints.
Heavy-dependency-free (no pyyaml at import time — caller provides a dict).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# EVAL_SEEDS from curriculum — used for validation
_EVAL_SEEDS_SET = frozenset({42, 123, 456, 789, 1024})


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base: str = "Qwen/Qwen2.5-1.5B-Instruct"
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
    use_vllm: bool = False
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
    backend: Literal["hf", "unsloth"] = "hf"
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
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    seed: SeedConfig = Field(default_factory=SeedConfig)

    @model_validator(mode="after")
    def vllm_requires_unsloth_backend(self) -> "TrainingConfig":
        if self.rollout.use_vllm and self.backend != "unsloth":
            raise ValueError(
                f"rollout.use_vllm=true requires backend='unsloth' "
                f"(current backend={self.backend!r}); the HF backend has no vLLM path. "
                f"Either set backend='unsloth' or leave rollout.use_vllm=false."
            )
        return self
