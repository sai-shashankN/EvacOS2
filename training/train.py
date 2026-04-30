"""GRPO training loop.

All heavy dependencies (torch, transformers, trl, peft) are imported
inside ``run_training()`` so the module itself stays import-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

from training.compat import patch_transformers_cache_exports
from training.config_schema import TrainingConfig
from training.metrics import (
    append_training_metrics_row,
    read_training_metrics_rows,
    write_trace_row,
    write_training_metrics_rows,
)

_TRAINER_DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "loss",
    "policy_loss",
    "kl_loss",
    "ratio_mean",
    "ratio_std",
    "clip_fraction",
    "kl_max",
    "mask_coverage",
    "mean_advantage",
    "advantage_std",
    "group_raw_reward_std_mean",
    "group_raw_reward_std_min",
    "group_raw_reward_std_max",
    "singleton_group_rate",
    "loss_mean_across_epochs",
    "policy_loss_mean_across_epochs",
    "kl_loss_mean_across_epochs",
    "ratio_mean_across_epochs",
    "ratio_std_mean_across_epochs",
    "clip_fraction_mean_across_epochs",
    "kl_max_across_epochs",
    "num_inner_epochs",
)
_ROLE_DIAGNOSTIC_ROLES: tuple[str, ...] = ("orchestrator", "floor_agent")
_TRAINER_DIAGNOSTIC_MAX_KEYS: frozenset[str] = frozenset({"kl_max", "kl_max_across_epochs"})
_TRAINER_DIAGNOSTIC_INT_KEYS: frozenset[str] = frozenset({"num_inner_epochs"})


def _load_yaml_config(config_path: Path) -> dict:
    """Load a YAML config file without requiring PyYAML at import time."""
    try:
        import yaml  # type: ignore

        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass

    with open(config_path, encoding="utf-8") as f:
        text = f.read()

    result: dict[str, Any] = {}
    current_section: str | None = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            result.setdefault(current_section, {})
            i += 1
            continue

        if indent == 0 and ":" in stripped:
            key, _, val = stripped.partition(":")
            result[key.strip()] = _parse_yaml_value(val.strip())
            i += 1
            continue

        if current_section is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("- "):
                    lst: list[Any] = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith("- "):
                        lst.append(_parse_yaml_value(lines[j].strip()[2:].strip()))
                        j += 1
                    result[current_section][key] = lst
                    i = j
                    continue
                result[current_section][key] = {}
                i += 1
                continue

            result[current_section][key] = _parse_yaml_value(val)
        i += 1

    return result


def _parse_yaml_value(val: str) -> Any:
    """Parse a simple YAML scalar value."""
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in {"null", "none"}:
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _config_hash(config: TrainingConfig) -> str:
    blob = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def _validate_config_path_identity(config_path: Path, config: TrainingConfig) -> None:
    """Fail fast when a run-name suffix disagrees with the loaded config.

    This guards the exact Vast failure mode where a file named ``*-750.yaml``
    contained a 2000-step curriculum and silently wrote 750-looking artifacts.
    """
    stem = config_path.stem
    match = re.search(r"-(\d+)$", stem)
    if match and config.max_steps is not None:
        expected_steps = int(match.group(1))
        if config.max_steps != expected_steps:
            raise ValueError(
                f"Config path {config_path} implies {expected_steps} steps, "
                f"but max_steps={config.max_steps}. Rename the config/artifacts "
                "or fix max_steps before launching a paid run."
            )

    path_fields = {
        "checkpoint.root_dir": config.checkpoint.root_dir,
        "metrics.csv_path": config.metrics.csv_path,
        "metrics.jsonl_dir": config.metrics.jsonl_dir,
    }
    for field_name, raw_path in path_fields.items():
        path_match = re.search(
            r"-(\d+)(?:-(?:metrics|logs))?(?:\.[^.]+)?$",
            str(raw_path),
        )
        if path_match and config.max_steps is not None:
            expected_steps = int(path_match.group(1))
            if config.max_steps != expected_steps:
                raise ValueError(
                    f"{field_name}={raw_path!r} implies {expected_steps} steps, "
                    f"but max_steps={config.max_steps}. Use step-consistent "
                    "output roots so artifacts cannot be mislabeled."
                )


class _FixedTierCurriculum:
    """Tiny curriculum shim for eval sweeps over an explicit tier."""

    def __init__(self, tier: str) -> None:
        self._tier = tier

    def suggest_next_tier(self, disaster_family: str) -> str:
        return self._tier

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:
        return None


class _ScheduledTierCurriculum:
    """Training-time curriculum with an explicit step-indexed tier schedule."""

    def __init__(self, tiers: list[str]) -> None:
        if not tiers:
            raise ValueError("Scheduled curriculum requires at least one tier")
        self._tiers = list(tiers)
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = max(0, step)

    def suggest_next_tier(self, disaster_family: str) -> str:
        del disaster_family
        return self._tiers[min(self._step, len(self._tiers) - 1)]

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "scheduled",
            "step": self._step,
            "tiers": self._tiers,
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        self._step = int(data.get("step", 0) or 0)


def _extract_trainer_model(policy: Any) -> Any:
    return getattr(policy, "_model", policy)


def _extract_trainer_tokenizer(policy: Any) -> Any:
    return getattr(policy, "_tokenizer", None)


def _disable_dropout_modules(model: Any) -> None:
    """Disable dropout without forcing the full model into eval mode.

    Unsloth's training kernels can fail backward when the GRPO loss path runs
    under ``model.eval()``. For stable ratios we still want dropout off, so we
    zero module dropout probabilities once and keep the model in training mode.
    """
    modules = getattr(model, "modules", None)
    if modules is None:
        return

    for module in modules():
        dropout_p = getattr(module, "p", None)
        if dropout_p is None:
            continue
        module_name = type(module).__name__.lower()
        if "dropout" in module_name:
            try:
                module.p = 0.0
            except Exception:
                continue


def _enable_gradient_checkpointing_if_available(model: Any) -> None:
    """Re-arm gradient checkpointing hooks after Unsloth / adapter reloads."""
    seen: set[int] = set()
    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        enable = getattr(candidate, "gradient_checkpointing_enable", None)
        if not callable(enable):
            continue
        try:
            enable()
        except TypeError:
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                continue


def _resolved_role_model_names(config: TrainingConfig) -> dict[str, str]:
    return config.model.resolved_bases()


def _checkpoint_role_model_names(config: TrainingConfig) -> dict[str, str] | None:
    if not config.uses_role_routed_policy:
        return None
    return _resolved_role_model_names(config)


def _checkpoint_orchestrator_policy(config: TrainingConfig) -> str | None:
    if not config.uses_role_routed_policy:
        return None
    return config.policy_for_role("orchestrator")


def _role_uses_model_policy(config: TrainingConfig, role: str) -> bool:
    return config.policy_for_role(role) == "model"


def _model_backed_trainable_roles(config: TrainingConfig) -> tuple[str, ...]:
    return tuple(
        role
        for role in config.trainable_roles
        if _role_uses_model_policy(config, role)
    )


def _configured_frozen_adapter_paths(config: TrainingConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role in _ROLE_DIAGNOSTIC_ROLES:
        adapter_path = config.frozen_adapter_path_for_role(role)  # type: ignore[arg-type]
        if adapter_path is None:
            continue
        path = Path(adapter_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                f"roles.frozen_adapter_paths[{role!r}] points to {path!s}, "
                "but that adapter directory does not exist."
            )
        paths[role] = path
    return paths


def _frozen_adapter_paths_for_run(
    config: TrainingConfig,
    bundle: Any | None,
) -> dict[str, Path]:
    """Resolve frozen role adapters from checkpoint first, config second.

    Checkpoints copy frozen adapters into ``latest/lora_adapter``. On resume or
    evaluation from another machine, the original configured source path may no
    longer exist, so the checkpoint-local copy must take precedence.
    """

    if not config.uses_role_routed_policy:
        return {}

    bundle_paths = getattr(bundle, "role_lora_weights_paths", None) or {}
    paths: dict[str, Path] = {}
    missing_roles: list[str] = []
    for role in _ROLE_DIAGNOSTIC_ROLES:
        if config.is_role_trainable(role):  # type: ignore[arg-type]
            continue

        bundle_path = bundle_paths.get(role)
        if bundle_path is not None and Path(bundle_path).exists():
            paths[role] = Path(bundle_path)
            continue

        configured_path = config.frozen_adapter_path_for_role(role)  # type: ignore[arg-type]
        if configured_path is not None:
            path = Path(configured_path).expanduser()
            if not path.exists():
                raise RuntimeError(
                    f"roles.frozen_adapter_paths[{role!r}] points to {path!s}, "
                    "but that adapter directory does not exist and no "
                    "checkpoint-local copy is available."
                )
            paths[role] = path
            continue

        if bundle_path is not None:
            missing_roles.append(role)

    if missing_roles:
        raise RuntimeError(
            "Checkpoint bundle references missing frozen adapter role(s): "
            f"{missing_roles!r}"
        )
    return paths


def _configured_floor_specialist_adapter_paths(config: TrainingConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for family, adapter_path in config.roles.frozen_floor_specialist_adapter_paths.items():
        path = Path(adapter_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                "roles.frozen_floor_specialist_adapter_paths"
                f"[{family!r}] points to {path!s}, but that adapter directory "
                "does not exist."
            )
        paths[family] = path
    return paths


def _frozen_floor_specialist_adapter_paths_for_run(
    config: TrainingConfig,
    bundle: Any | None,
) -> dict[str, Path]:
    """Resolve routed frozen floor specialists from checkpoint first."""

    configured_paths = config.roles.frozen_floor_specialist_adapter_paths
    if not configured_paths:
        return {}

    bundle_paths = getattr(bundle, "floor_specialist_lora_weights_paths", None) or {}
    paths: dict[str, Path] = {}
    missing_families: list[str] = []
    for family, configured_path_raw in configured_paths.items():
        bundle_path = bundle_paths.get(family)
        if bundle_path is not None and Path(bundle_path).exists():
            paths[family] = Path(bundle_path)
            continue

        configured_path = Path(configured_path_raw).expanduser()
        if configured_path.exists():
            paths[family] = configured_path
            continue

        if bundle_path is not None:
            missing_families.append(family)
            continue

        raise RuntimeError(
            "roles.frozen_floor_specialist_adapter_paths"
            f"[{family!r}] points to {configured_path!s}, but that adapter "
            "directory does not exist and no checkpoint-local copy is available."
        )

    if missing_families:
        raise RuntimeError(
            "Checkpoint bundle references missing frozen floor specialist "
            f"adapter(s): {missing_families!r}"
        )
    return paths


def _resolved_model_name_tag(config: TrainingConfig) -> str:
    resolved = _resolved_role_model_names(config)
    if resolved["orchestrator"] == resolved["floor_agent"]:
        return resolved["orchestrator"]
    return (
        f"orchestrator={resolved['orchestrator']};"
        f"floor_agent={resolved['floor_agent']}"
    )


def _set_policy_sampling_temperature(policy: Any, temperature: float) -> list[tuple[Any, str, Any]]:
    """Set nested policy generation to a temperature and return restore tokens."""

    restore_tokens: list[tuple[Any, str, Any]] = []
    seen: set[int] = set()

    def _visit(candidate: Any) -> None:
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)

        if hasattr(candidate, "_temperature"):
            restore_tokens.append((candidate, "_temperature", getattr(candidate, "_temperature")))
            setattr(candidate, "_temperature", float(temperature))

        gen_kwargs = getattr(candidate, "_gen_kwargs", None)
        if isinstance(gen_kwargs, dict):
            restore_tokens.append((gen_kwargs, "temperature", gen_kwargs.get("temperature")))
            restore_tokens.append((gen_kwargs, "do_sample", gen_kwargs.get("do_sample")))
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["do_sample"] = float(temperature) > 0.0

        role_policies = getattr(candidate, "_role_policies", None)
        if isinstance(role_policies, dict):
            for child in role_policies.values():
                _visit(child)

        specialist_policies = getattr(candidate, "_specialist_policies", None)
        if isinstance(specialist_policies, dict):
            for child in specialist_policies.values():
                _visit(child)

        _visit(getattr(candidate, "_generalist_policy", None))

    _visit(policy)
    return restore_tokens


def _restore_policy_sampling_temperature(tokens: list[tuple[Any, str, Any]]) -> None:
    for target, key, value in reversed(tokens):
        if isinstance(target, dict):
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
        else:
            setattr(target, key, value)


def _merge_trainer_diagnostics_into_metrics(
    metrics_row: dict[str, Any],
    trainer_diagnostics: dict[str, Any],
) -> None:
    """Copy trainer diagnostics into the metrics row.

    Split-role runs emit per-role diagnostics like ``orchestrator_loss`` and
    ``floor_agent_loss``. For plotting compatibility we also synthesize the
    canonical aggregate keys (``loss``, ``ratio_mean``, etc.) from those
    per-role values while preserving the role-specific columns.
    """

    if not trainer_diagnostics:
        return

    for role in _ROLE_DIAGNOSTIC_ROLES:
        sample_key = f"{role}_sample_groups"
        if sample_key in trainer_diagnostics:
            metrics_row[sample_key] = trainer_diagnostics[sample_key]

    role_weights: dict[str, float] = {}
    for role in _ROLE_DIAGNOSTIC_ROLES:
        weight = trainer_diagnostics.get(f"{role}_sample_groups")
        if isinstance(weight, (int, float)) and weight > 0:
            role_weights[role] = float(weight)

    for key in _TRAINER_DIAGNOSTIC_KEYS:
        if key in trainer_diagnostics:
            metrics_row[key] = trainer_diagnostics[key]
            continue

        role_values: list[tuple[str, float]] = []
        for role in _ROLE_DIAGNOSTIC_ROLES:
            role_key = f"{role}_{key}"
            if role_key in trainer_diagnostics:
                value = trainer_diagnostics[role_key]
                metrics_row[role_key] = value
                if isinstance(value, (int, float)):
                    role_values.append((role, float(value)))

        if not role_values:
            continue

        if key in _TRAINER_DIAGNOSTIC_MAX_KEYS:
            metrics_row[key] = max(value for _, value in role_values)
            continue

        if key in _TRAINER_DIAGNOSTIC_INT_KEYS:
            metrics_row[key] = int(max(value for _, value in role_values))
            continue

        total_weight = sum(role_weights.get(role, 1.0) for role, _ in role_values)
        weighted_total = sum(
            value * role_weights.get(role, 1.0) for role, value in role_values
        )
        metrics_row[key] = weighted_total / max(total_weight, 1.0)


class MultiAgentGRPOTrainer:
    """GRPO-family trainer with role-specific group-relative advantages.

    Implements a clipped surrogate objective (PPO-style) with Schulman k3 KL
    penalty to a reference model obtained by disabling LoRA adapters.  Per-
    token, masked, numerically stable.  This is *not* plain GRPO — it is a
    multi-agent adaptation where advantages are computed within role-specific
    groups (orchestrator pooled across the rollout batch; floor agents per
    (episode, round)).
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        learning_rate: float,
        kl_coef: float,
        clip_range: float,
        num_train_epochs_per_step: int,
        optimizer_state: dict | None = None,
    ) -> None:
        import torch

        if tokenizer is None:
            raise RuntimeError("MultiAgentGRPOTrainer requires a tokenizer")

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.kl_coef = float(kl_coef)
        self.clip_range = float(clip_range)
        self.num_train_epochs_per_step = max(1, int(num_train_epochs_per_step))
        self.logprob_microbatch_size = max(
            1,
            int(os.getenv("EVACOS_LOGPROB_MICROBATCH_SIZE", "4")),
        )

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_training(
                self.model,
                use_gradient_checkpointing=True,
            )
        except Exception:
            self.model.train()
        _enable_gradient_checkpointing_if_available(self.model)
        _disable_dropout_modules(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = self._recover_trainable_params()
        if not trainable_params:
            raise RuntimeError("MultiAgentGRPOTrainer found no trainable parameters")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        self._warned_missing_completion_token_ids = False
        self._step_counter = 0

        # Phase 12: restore optimizer state when resuming from checkpoint
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

    def _recover_trainable_params(self) -> list[Any]:
        """Re-enable LoRA params when an inference-mode wrapper froze them.

        Unsloth's inference path can leave all parameters non-trainable even
        after `for_training(...)`. In that case, recover the expected LoRA
        adapter parameters by name.
        """
        recovered: list[Any] = []
        for name, param in self.model.named_parameters():
            lower = name.lower()
            if "lora_" in lower or "lora" in lower or "modules_to_save" in lower:
                param.requires_grad_(True)
                recovered.append(param)
        return recovered

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _effective_max_length(self) -> int:
        """Return the usable sequence length for trainer forward passes.

        Some Unsloth wrappers keep ``tokenizer.model_max_length`` very large
        even when the loaded model was initialized with a shorter
        ``max_seq_length``.  The trainer must respect the model-side limit or
        labels can become wider than logits during log-prob gathering.
        """

        candidates: list[int] = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000:
            candidates.append(tokenizer_limit)
        else:
            candidates.append(4096)

        for owner in (self.model, getattr(self.model, "config", None)):
            if owner is None:
                continue
            for attr in ("max_seq_length", "max_position_embeddings", "seq_length"):
                value = getattr(owner, attr, None)
                if isinstance(value, int) and value > 0:
                    candidates.append(value)

        return max(2, min(candidates))

    def _tokenize_batch(
        self,
        prompts: list[Any],
        completions: list[str],
        completion_token_ids: list[list[int] | None] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Tokenize prompts+completions, build labels with prompt masked out.

        Returns (encoded_full, shifted_labels) where shifted_labels has -100
        at every position that is padding or prompt (completion-only mask).
        """
        import logging

        torch = self._torch
        logger = logging.getLogger(__name__)

        if completion_token_ids is None:
            completion_token_ids = [None] * len(completions)

        # Fix C8: tokenizer may be left-padded from generation paths.
        # Force right-padding for the training tokenization so that the
        # prompt-masking logic labels[row_idx, :plen] = -100 is correct.
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            rendered_prompts = [
                self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

            has_direct_completion_ids = any(bool(ids) for ids in completion_token_ids)
            max_len = self._effective_max_length()

            if not has_direct_completion_ids:
                if not self._warned_missing_completion_token_ids:
                    logger.warning(
                        "Trainer received rollout samples without completion_token_ids; "
                        "falling back to decode/re-encode for completion text."
                    )
                    self._warned_missing_completion_token_ids = True

                full_texts = [
                    ptext + comp
                    for ptext, comp in zip(rendered_prompts, completions, strict=False)
                ]

                encoded_full = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                encoded_prompt = self.tokenizer(
                    rendered_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )

                device = next(self.model.parameters()).device
                encoded_full = {k: v.to(device) for k, v in encoded_full.items()}
                prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1).tolist()

                labels = encoded_full["input_ids"].clone()
                labels[encoded_full["attention_mask"] == 0] = -100
                L_full = encoded_full["input_ids"].shape[1]
                prompt_only_rows: list[tuple[int, int, int]] = []
                for row_idx, plen in enumerate(prompt_lengths):
                    labels[row_idx, : int(plen)] = -100
                    if plen >= L_full:
                        logger.warning(
                            "Row %d truncated to prompt-only (plen=%d, L=%d); "
                            "completion mask will be empty for this row.",
                            row_idx,
                            plen,
                            L_full,
                        )
                        prompt_only_rows.append((row_idx, plen, L_full))
                if prompt_only_rows:
                    row_idx, plen, L = prompt_only_rows[0]
                    raise RuntimeError(
                        f"Prompt-only truncation detected: row {row_idx} has "
                        f"plen={plen} >= L_full={L}. "
                        f"{len(prompt_only_rows)} row(s) affected. "
                        f"Increase max_length or shorten prompts/completions."
                    )

                shifted_labels = labels[:, 1:]
                return encoded_full, shifted_labels

            prompt_token_ids_per_row: list[list[int]] = []
            full_token_ids_per_row: list[list[int]] = []
            prompt_lengths: list[int] = []
            prompt_only_rows: list[tuple[int, int, int]] = []

            for row_idx, (prompt, rendered_prompt, completion, row_completion_ids) in enumerate(
                zip(prompts, rendered_prompts, completions, completion_token_ids, strict=False)
            ):
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if not isinstance(prompt_ids, list):
                    prompt_ids = self.tokenizer(
                        rendered_prompt,
                        add_special_tokens=False,
                    )["input_ids"]
                prompt_ids = list(prompt_ids)
                prompt_token_ids_per_row.append(prompt_ids)

                if row_completion_ids:
                    completion_ids = list(row_completion_ids)
                else:
                    if not self._warned_missing_completion_token_ids:
                        logger.warning(
                            "Trainer received rollout samples without completion_token_ids; "
                            "falling back to decode/re-encode for completion text."
                        )
                        self._warned_missing_completion_token_ids = True
                    completion_ids = list(
                        self.tokenizer(completion, add_special_tokens=False)["input_ids"]
                    )

                if len(prompt_ids) + len(completion_ids) > max_len:
                    # Preserve the completion/reward-bearing action and keep
                    # as much recent prompt context as fits.  Right truncation
                    # would drop the action entirely on long observations.
                    if len(completion_ids) >= max_len:
                        prompt_tail = prompt_ids[-1:] if prompt_ids else []
                        completion_budget = max_len - len(prompt_tail)
                        completion_ids = completion_ids[-completion_budget:]
                    else:
                        prompt_budget = max_len - len(completion_ids)
                        prompt_tail = prompt_ids[-prompt_budget:]
                    full_ids = prompt_tail + completion_ids
                    truncated_prompt_len = len(prompt_tail)
                else:
                    full_ids = prompt_ids + completion_ids
                    truncated_prompt_len = len(prompt_ids)
                prompt_lengths.append(truncated_prompt_len)
                if truncated_prompt_len >= len(full_ids):
                    logger.warning(
                        "Row %d truncated to prompt-only (plen=%d, L=%d); "
                        "completion mask will be empty for this row.",
                        row_idx,
                        truncated_prompt_len,
                        len(full_ids),
                    )
                    prompt_only_rows.append((row_idx, truncated_prompt_len, len(full_ids)))
                full_token_ids_per_row.append(full_ids)

            if prompt_only_rows:
                row_idx, plen, L = prompt_only_rows[0]
                raise RuntimeError(
                    f"Prompt-only truncation detected: row {row_idx} has "
                    f"plen={plen} >= L_full={L}. "
                    f"{len(prompt_only_rows)} row(s) affected. "
                    f"Increase max_length or shorten prompts/completions."
                )

            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id")

            batch_width = max(len(row) for row in full_token_ids_per_row)
            padded_input_ids: list[list[int]] = []
            attention_masks: list[list[int]] = []
            for full_ids in full_token_ids_per_row:
                pad_width = batch_width - len(full_ids)
                padded_input_ids.append(full_ids + [pad_token_id] * pad_width)
                attention_masks.append([1] * len(full_ids) + [0] * pad_width)

            device = next(self.model.parameters()).device
            encoded_full = {
                "input_ids": torch.tensor(padded_input_ids).to(device),
                "attention_mask": torch.tensor(attention_masks).to(device),
            }

            labels = encoded_full["input_ids"].clone()
            labels[encoded_full["attention_mask"] == 0] = -100
            for row_idx, plen in enumerate(prompt_lengths):
                labels[row_idx, : int(plen)] = -100

            shifted_labels = labels[:, 1:]
            return encoded_full, shifted_labels
        finally:
            self.tokenizer.padding_side = previous_padding_side

    def _masked_token_logprobs(
        self,
        encoded_full: dict[str, Any],
        shifted_labels: Any,
    ) -> Any:
        """Compute per-token log-probs for the completion tokens.

        Returns (S, L-1) tensor of log-probs at every position.  Positions
        where shifted_labels == -100 are set to 0 (caller should mask them).
        """
        torch = self._torch
        batch_size = int(shifted_labels.shape[0])
        chunk_size = min(self.logprob_microbatch_size, batch_size)
        chunks: list[Any] = []

        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            encoded_chunk = {
                key: value[start:stop]
                for key, value in encoded_full.items()
            }
            labels_chunk = shifted_labels[start:stop]
            outputs = self.model(**encoded_chunk)
            logits = outputs.logits[:, :-1, :]  # (S, L-1, V)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            gather_labels = labels_chunk.masked_fill(labels_chunk == -100, 0)
            token_log_probs = log_probs.gather(
                -1,
                gather_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_log_probs = token_log_probs.masked_fill(labels_chunk == -100, 0.0)
            chunks.append(token_log_probs)

        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def _compute_group_advantages(
        self,
        grouped_raw_rewards: dict[str, list[float]],
        total_samples: int,
    ) -> Any:
        """Compute group-normalised advantages from raw rewards.

        Args:
            grouped_raw_rewards: maps group_id -> list of raw (un-normalized)
                reward floats, in the order samples were flattened.
            total_samples: total number of samples across all groups.

        Returns:
            (total_samples,) float tensor of advantages, detached.
        """
        import warnings

        torch = self._torch
        device = next(self.model.parameters()).device
        advantages = torch.zeros(total_samples, device=device, dtype=torch.float32)
        offset = 0

        for group_id, rewards in grouped_raw_rewards.items():
            n = len(rewards)
            if n == 0:
                continue
            r = torch.tensor(rewards, device=device, dtype=torch.float32)
            if n >= 2:
                mean_r = r.mean()
                std_r = r.std(unbiased=False).clamp_min(1e-8)
                group_adv = (r - mean_r) / std_r
            else:
                warnings.warn(
                    f"Group {group_id!r} has only 1 sample; "
                    "setting advantage to 0.0 as defensive fallback.",
                    stacklevel=2,
                )
                group_adv = torch.zeros_like(r)
            advantages[offset : offset + n] = group_adv
            offset += n

        return advantages.detach()

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def step(
        self,
        grouped_inputs: dict[str, list[list[Any]]],
    ) -> dict[str, float]:
        """Execute one GRPO step with PPO-style clipped surrogate + KL penalty.

        grouped_inputs is produced by ``_group_for_grpo`` and has keys:
            "prompts"   -> list[list[chat_messages]]  (per group)
            "completions" -> list[list[str]]           (per group)
            "completion_token_ids" -> list[list[list[int] | None]] (per group)
            "raw_rewards" -> list[list[float]]         (per group, RAW not normalised)
            "samples"     -> list[list[TrajectorySample]] (per group, for diagnostics)
        """
        torch = self._torch
        import warnings

        # --- Flatten across groups -----------------------------------------
        all_prompts: list[Any] = []
        all_completions: list[str] = []
        grouped_raw_rewards: dict[str, list[float]] = {}
        total_samples = 0
        all_completion_token_ids: list[list[int] | None] = []
        prompt_groups = grouped_inputs["prompts"]
        completion_groups = grouped_inputs["completions"]
        completion_token_id_groups = grouped_inputs.get("completion_token_ids", [])
        if not completion_token_id_groups:
            completion_token_id_groups = [
                [None for _ in completions]
                for completions in completion_groups
            ]
        raw_reward_groups = grouped_inputs["raw_rewards"]
        # Use the ordering from the grouped_inputs (already sorted by key)
        for idx, (prompts, completions, completion_ids, rewards) in enumerate(
            zip(
                prompt_groups,
                completion_groups,
                completion_token_id_groups,
                raw_reward_groups,
                strict=False,
            )
        ):
            group_key = f"group_{idx}"
            all_prompts.extend(prompts)
            all_completions.extend(completions)
            all_completion_token_ids.extend(completion_ids)
            grouped_raw_rewards[group_key] = rewards
            total_samples += len(prompts)

        if total_samples == 0:
            raise RuntimeError("MultiAgentGRPOTrainer received no rollout samples")

        # --- Tokenize ONCE -------------------------------------------------
        encoded_full, shifted_labels = self._tokenize_batch(
            all_prompts,
            all_completions,
            all_completion_token_ids,
        )
        completion_mask = (shifted_labels != -100).float()  # (S, L-1)
        self._step_counter += 1

        # Keep the model on the training path for Unsloth compatibility.
        # Dropout was already zeroed in __init__, so old/ref/new log-probs are
        # still computed deterministically without switching to eval mode.
        previous_training_mode = self.model.training
        self.model.train()
        try:

        # --- 1. Old log-probs: frozen, captured ONCE -----------------------
            with torch.no_grad():
                old_lp = self._masked_token_logprobs(
                    encoded_full,
                    shifted_labels,
                ).detach()  # (S, L-1)

        # --- 2. Ref log-probs: LoRA adapter disabled -----------------------
            with torch.no_grad():
                try:
                    cm = self.model.disable_adapter()
                except AttributeError:
                    # Model doesn't support disable_adapter (e.g. non-PEFT);
                    # fall back to old_lp (KL will be zero, which is harmless).
                    cm = None

                if cm is not None:
                    with cm:
                        ref_lp = self._masked_token_logprobs(
                            encoded_full,
                            shifted_labels,
                        ).detach()
                else:
                    warnings.warn(
                        "Model does not expose disable_adapter(); "
                        "reference log-probs will equal old log-probs (KL = 0).",
                        stacklevel=2,
                    )
                    ref_lp = old_lp.clone().detach()

        # --- 3. Advantages: computed once, detached -------------------------
            advantages = self._compute_group_advantages(
                grouped_raw_rewards, total_samples
            )  # (S,)

        # --- 4. Inner PPO epoch loop ----------------------------------------
            epoch_ratio_means: list[float] = []
            epoch_ratio_stds: list[float] = []
            epoch_clip_fractions: list[float] = []
            epoch_kl_maxes: list[float] = []
            epoch_policy_losses: list[float] = []
            epoch_kl_losses: list[float] = []
            epoch_losses: list[float] = []

            mask_sum_tensor = completion_mask.sum().clamp_min(1.0)
            mask_sum_value = float(mask_sum_tensor.detach().item())
            batch_size = int(shifted_labels.shape[0])
            chunk_size = min(self.logprob_microbatch_size, batch_size)

            for _epoch in range(self.num_train_epochs_per_step):
                ratio_sum_epoch = 0.0
                ratio_sq_sum_epoch = 0.0
                clip_count_epoch = 0.0
                kl_max_epoch = 0.0
                policy_loss_total = 0.0
                kl_loss_total = 0.0
                loss_total = 0.0

                self.optimizer.zero_grad()
                for start in range(0, batch_size, chunk_size):
                    stop = min(start + chunk_size, batch_size)
                    encoded_chunk = {
                        key: value[start:stop]
                        for key, value in encoded_full.items()
                    }
                    labels_chunk = shifted_labels[start:stop]
                    mask_chunk = completion_mask[start:stop]
                    old_chunk = old_lp[start:stop]
                    ref_chunk = ref_lp[start:stop]
                    adv_chunk = advantages[start:stop]

                    new_lp = self._masked_token_logprobs(
                        encoded_chunk,
                        labels_chunk,
                    )

                    # FP16-safe log-prob ratio via log-space delta.
                    delta = (new_lp - old_chunk).clamp(-5.0, 5.0)
                    ratio = delta.exp()

                    # Broadcast advantage to token dimension: (S,) -> (S, L-1).
                    A_tok = adv_chunk.unsqueeze(-1) * torch.ones_like(ratio)

                    surr1 = ratio * A_tok
                    surr2 = (
                        ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
                        * A_tok
                    )
                    policy_loss_chunk = -(
                        torch.min(surr1, surr2) * mask_chunk
                    ).sum() / mask_sum_tensor

                    # Schulman k3 KL estimator: k3 = exp(ref - new) - (ref - new) - 1.
                    ref_delta = (ref_chunk - new_lp).clamp(-5.0, 5.0)
                    kl_per_tok = ref_delta.exp() - ref_delta - 1.0
                    kl_loss_chunk = (
                        (kl_per_tok * mask_chunk).sum() / mask_sum_tensor
                    )
                    if hasattr(torch, "isfinite"):
                        finite_kl = torch.isfinite(kl_loss_chunk).all()
                        if hasattr(finite_kl, "item"):
                            finite_kl = finite_kl.item()
                        if not finite_kl:
                            raise RuntimeError(
                                f"Non-finite kl_loss detected at step {self._step_counter}; "
                                f"dtype={getattr(kl_loss_chunk, 'dtype', '?')}, "
                                f"value={kl_loss_chunk.detach()}"
                            )
                        finite_policy = torch.isfinite(policy_loss_chunk).all()
                        if hasattr(finite_policy, "item"):
                            finite_policy = finite_policy.item()
                        if not finite_policy:
                            raise RuntimeError(
                                f"Non-finite policy_loss detected at step {self._step_counter}; "
                                f"dtype={getattr(policy_loss_chunk, 'dtype', '?')}, "
                                f"value={policy_loss_chunk.detach()}"
                            )

                    loss_chunk = policy_loss_chunk + self.kl_coef * kl_loss_chunk
                    loss_chunk.backward()

                    ratio_detached = ratio.detach()
                    masked_ratio_chunk = ratio_detached * mask_chunk
                    ratio_sum_epoch += masked_ratio_chunk.sum().item()
                    ratio_sq_sum_epoch += ((ratio_detached ** 2) * mask_chunk).sum().item()
                    outside_clip_chunk = (
                        ((ratio_detached < 1.0 - self.clip_range)
                         | (ratio_detached > 1.0 + self.clip_range))
                        * mask_chunk
                    )
                    clip_count_epoch += outside_clip_chunk.sum().item()
                    if mask_chunk.sum().item() > 0:
                        kl_max_epoch = max(
                            kl_max_epoch,
                            (kl_per_tok.detach() * mask_chunk).max().item(),
                        )
                    policy_loss_total += policy_loss_chunk.detach().item()
                    kl_loss_total += kl_loss_chunk.detach().item()
                    loss_total += loss_chunk.detach().item()

                    del new_lp, ratio, kl_per_tok, loss_chunk

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                ratio_mean_epoch = ratio_sum_epoch / mask_sum_value
                ratio_sq_mean_epoch = ratio_sq_sum_epoch / mask_sum_value
                ratio_std_epoch = (
                    max(ratio_sq_mean_epoch - ratio_mean_epoch ** 2, 0.0)
                ) ** 0.5
                clip_fraction_epoch = clip_count_epoch / mask_sum_value

                epoch_ratio_means.append(ratio_mean_epoch)
                epoch_ratio_stds.append(ratio_std_epoch)
                epoch_clip_fractions.append(clip_fraction_epoch)
                epoch_kl_maxes.append(kl_max_epoch)
                epoch_policy_losses.append(policy_loss_total)
                epoch_kl_losses.append(kl_loss_total)
                epoch_losses.append(loss_total)

        # --- 5. Diagnostics -------------------------------------------------
            mask_coverage = completion_mask.mean().item()
            mean_advantage = advantages.mean().item()
            advantage_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            group_raw_reward_stds = [
                _population_std([float(value) for value in rewards])
                for rewards in grouped_raw_rewards.values()
            ]
            singleton_group_rate = sum(
                1 for rewards in grouped_raw_rewards.values() if len(rewards) < 2
            ) / max(len(grouped_raw_rewards), 1)

            return {
                "loss": epoch_losses[-1],
                "policy_loss": epoch_policy_losses[-1],
                "kl_loss": epoch_kl_losses[-1],
                "ratio_mean": epoch_ratio_means[-1],
                "ratio_std": epoch_ratio_stds[-1],
                "clip_fraction": epoch_clip_fractions[-1],
                "kl_max": epoch_kl_maxes[-1],
                "mask_coverage": mask_coverage,
                "mean_advantage": mean_advantage,
                "advantage_std": advantage_std,
                "group_raw_reward_std_mean": (
                    sum(group_raw_reward_stds) / max(len(group_raw_reward_stds), 1)
                ),
                "group_raw_reward_std_min": min(group_raw_reward_stds, default=0.0),
                "group_raw_reward_std_max": max(group_raw_reward_stds, default=0.0),
                "singleton_group_rate": singleton_group_rate,
                "loss_mean_across_epochs": sum(epoch_losses) / max(len(epoch_losses), 1),
                "policy_loss_mean_across_epochs": sum(epoch_policy_losses) / max(len(epoch_policy_losses), 1),
                "kl_loss_mean_across_epochs": sum(epoch_kl_losses) / max(len(epoch_kl_losses), 1),
                "ratio_mean_across_epochs": sum(epoch_ratio_means) / max(len(epoch_ratio_means), 1),
                "ratio_std_mean_across_epochs": sum(epoch_ratio_stds) / max(len(epoch_ratio_stds), 1),
                "clip_fraction_mean_across_epochs": sum(epoch_clip_fractions) / max(len(epoch_clip_fractions), 1),
                "kl_max_across_epochs": max(epoch_kl_maxes) if epoch_kl_maxes else 0.0,
                "num_inner_epochs": len(epoch_losses),
            }
        finally:
            if previous_training_mode:
                self.model.train()
            else:
                self.model.eval()


class DualRoleGRPOTrainer:
    """Wrapper that trains the available role-specific policies independently."""

    def __init__(
        self,
        *,
        orchestrator_trainer: Any | None = None,
        floor_trainer: Any | None = None,
        role_trainers: dict[str, Any] | None = None,
    ) -> None:
        self._role_trainers: dict[str, Any] = dict(role_trainers or {})
        if orchestrator_trainer is not None:
            self._role_trainers["orchestrator"] = orchestrator_trainer
        if floor_trainer is not None:
            self._role_trainers["floor_agent"] = floor_trainer
        if not self._role_trainers:
            raise ValueError("DualRoleGRPOTrainer requires at least one role trainer")

    def trainer_for_role(self, role: str) -> Any:
        if role not in self._role_trainers:
            raise ValueError(f"Unknown role {role!r}")
        return self._role_trainers[role]

    def step(self, *, grouped_inputs: dict[str, list[list[Any]]]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        split_inputs = _split_grouped_inputs_by_role(grouped_inputs)
        for role, role_inputs in split_inputs.items():
            if role not in self._role_trainers:
                continue
            if not role_inputs["samples"]:
                continue
            trainer = self.trainer_for_role(role)
            role_diag = trainer.step(grouped_inputs=role_inputs) or {}
            diagnostics[f"{role}_sample_groups"] = len(role_inputs["samples"])
            diagnostics.update(
                {f"{role}_{key}": value for key, value in role_diag.items()}
            )
        return diagnostics


def _build_role_trainer(
    *,
    role: str,
    role_policy: Any,
    config: TrainingConfig,
    optimizer_state: dict | None = None,
) -> Any:
    model = _extract_trainer_model(role_policy)
    tokenizer = _extract_trainer_tokenizer(role_policy)
    if model is None or tokenizer is None:
        raise RuntimeError(
            f"Trainable role {role!r} does not expose model/tokenizer handles "
            "required by MultiAgentGRPOTrainer."
        )
    return MultiAgentGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config.grpo.learning_rate,
        kl_coef=config.grpo.kl_coef,
        clip_range=config.grpo.clip_range,
        num_train_epochs_per_step=config.grpo.num_train_epochs_per_step,
        optimizer_state=optimizer_state,
    )


def _build_grpo_trainer(
    GRPOTrainer: Any,
    policy: Any,
    config: TrainingConfig,
    optimizer_state: dict | None = None,
    role_optimizer_states: dict[str, dict] | None = None,
) -> Any:
    """Instantiate the project GRPO trainer by default, with opt-in TRL probing."""
    import inspect

    role_policies = getattr(policy, "_role_policies", None)
    if role_policies is not None:
        role_trainers: dict[str, Any] = {}
        for role in config.trainable_roles:
            if role not in role_policies:
                raise RuntimeError(
                    f"Policy is missing the trainable role {role!r}; cannot build GRPO trainer."
                )
            role_trainers[role] = _build_role_trainer(
                role=role,
                role_policy=role_policies[role],
                config=config,
                optimizer_state=(role_optimizer_states or {}).get(role),
            )
        return DualRoleGRPOTrainer(role_trainers=role_trainers)

    trainer_model = _extract_trainer_model(policy)
    tokenizer = _extract_trainer_tokenizer(policy)
    use_project_trainer = (
        not getattr(config.grpo, "prefer_trl", False)
        or GRPOTrainer is MultiAgentGRPOTrainer
        or GRPOTrainer is None
    )
    if use_project_trainer:
        return MultiAgentGRPOTrainer(
            model=trainer_model,
            tokenizer=tokenizer,
            learning_rate=config.grpo.learning_rate,
            kl_coef=config.grpo.kl_coef,
            clip_range=config.grpo.clip_range,
            num_train_epochs_per_step=config.grpo.num_train_epochs_per_step,
            optimizer_state=optimizer_state,
        )

    trainer_kwargs = {
        "model": trainer_model,
        "tokenizer": tokenizer,
        "learning_rate": config.grpo.learning_rate,
        "kl_coef": config.grpo.kl_coef,
        "clip_range": config.grpo.clip_range,
        "num_train_epochs_per_step": config.grpo.num_train_epochs_per_step,
    }
    params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    filtered = {
        key: value
        for key, value in trainer_kwargs.items()
        if key in params and value is not None
    }
    try:
        return GRPOTrainer(**filtered)
    except TypeError as exc:
        raise RuntimeError(
            "TRL GRPOTrainer signature does not accept our kwargs "
            f"({sorted(filtered)}); set grpo.prefer_trl=false to use MultiAgentGRPOTrainer."
        ) from exc


def _call_trainer_step(trainer: Any, grouped_inputs: dict[str, list[list[Any]]]) -> Any:
    """Call the trainer step with the grouped-inputs dict.

    MultiAgentGRPOTrainer.step expects a single dict argument produced by
    ``_group_for_grpo``.  TRL-based trainers (when compatible) are tried
    first with keyword args, then positional.
    """
    step_fn = getattr(trainer, "step", None)
    if step_fn is None:
        raise RuntimeError("GRPOTrainer does not expose a step(...) method)")

    # Fast path: our MultiAgentGRPOTrainer accepts a single dict
    try:
        return step_fn(grouped_inputs=grouped_inputs)
    except TypeError:
        pass

    prompt_groups = grouped_inputs["prompts"]
    completion_groups = grouped_inputs["completions"]
    reward_groups = grouped_inputs["raw_rewards"]

    try:
        return step_fn(
            grouped_prompts=prompt_groups,
            grouped_completions=completion_groups,
            grouped_rewards=reward_groups,
        )
    except TypeError:
        return step_fn(prompt_groups, completion_groups, reward_groups)


def _group_for_grpo(results: list[Any]) -> dict[str, list[list[Any]]]:
    grouped: dict[str, dict[str, list[Any]]] = {}
    for result in results:
        for sample in result.samples:
            bucket = grouped.setdefault(
                sample.group_id,
                {
                    "prompts": [],
                    "completions": [],
                    "completion_token_ids": [],
                    "raw_rewards": [],
                    "normalized_rewards": [],
                    "samples": [],
                },
            )
            bucket["prompts"].append(sample.prompt)
            bucket["completions"].append(sample.completion_text)
            bucket["completion_token_ids"].append(sample.completion_token_ids)
            bucket["raw_rewards"].append(sample.raw_reward)
            bucket["normalized_rewards"].append(sample.normalized_reward)
            bucket["samples"].append(sample)

    # Downstream trainers consume these grouped arrays positionally and do not
    # inspect the original group_id strings, so deterministic lexicographic
    # ordering is sufficient here.
    ordered_keys = sorted(grouped)
    return {
        "prompts": [grouped[key]["prompts"] for key in ordered_keys],
        "completions": [grouped[key]["completions"] for key in ordered_keys],
        "completion_token_ids": [grouped[key]["completion_token_ids"] for key in ordered_keys],
        "raw_rewards": [grouped[key]["raw_rewards"] for key in ordered_keys],
        "normalized_rewards": [grouped[key]["normalized_rewards"] for key in ordered_keys],
        "samples": [grouped[key]["samples"] for key in ordered_keys],
    }


def _split_grouped_inputs_by_role(
    grouped_inputs: dict[str, list[list[Any]]]
) -> dict[str, dict[str, list[list[Any]]]]:
    role_buckets: dict[str, dict[str, list[list[Any]]]] = {
        "orchestrator": {key: [] for key in grouped_inputs},
        "floor_agent": {key: [] for key in grouped_inputs},
    }
    sample_groups = grouped_inputs.get("samples", [])
    for idx, samples in enumerate(sample_groups):
        roles = {getattr(sample, "role", None) for sample in samples}
        if len(roles) != 1:
            raise RuntimeError(
                f"Grouped GRPO inputs must be role-pure; got roles={sorted(roles)!r}"
            )
        role = next(iter(roles))
        if role not in role_buckets:
            raise RuntimeError(f"Unexpected role {role!r} in grouped inputs")
        for key, value in grouped_inputs.items():
            role_buckets[role][key].append(value[idx])
    return role_buckets


def _compute_rollout_metrics(results: list[Any]) -> dict[str, float]:
    override_count = sum(int(getattr(result, "override_count", 0)) for result in results)
    orchestrator_action_count = sum(
        int(getattr(result, "orchestrator_action_count", 0)) for result in results
    )
    override_win_count = sum(int(getattr(result, "override_win_count", 0)) for result in results)
    rationale_bonus_total = sum(
        float(getattr(result, "rationale_bonus_total", 0.0)) for result in results
    )
    rationale_bonus_count = sum(
        int(getattr(result, "rationale_bonus_count", 0)) for result in results
    )
    samples = [
        sample
        for result in results
        for sample in getattr(result, "samples", [])
    ]

    def _parsed_action(sample: Any) -> dict[str, Any]:
        parsed = getattr(sample, "parsed_action", {})
        return parsed if isinstance(parsed, dict) else {}

    def _action_type(sample: Any) -> str:
        return str(_parsed_action(sample).get("action_type", ""))

    def _arguments(sample: Any) -> dict[str, Any]:
        arguments = _parsed_action(sample).get("arguments", {})
        return arguments if isinstance(arguments, dict) else {}

    def _is_valid_action(sample: Any) -> bool:
        return "fallback_reason" not in _parsed_action(sample)

    def _is_metric_action_sample(sample: Any) -> bool:
        parsed = _parsed_action(sample)
        return bool(parsed.get("selected_for_execution", True))

    def _is_wait_action(sample: Any) -> bool:
        return _action_type(sample) == "wait"

    def _has_empty_arguments(sample: Any) -> bool:
        return len(_arguments(sample)) == 0

    def _is_floor_route(sample: Any) -> bool:
        return getattr(sample, "role", "") == "floor_agent" and _action_type(sample) == "route_within_floor"

    def _route_target_kind(sample: Any) -> str:
        args = _arguments(sample)
        if args.get("exit_id"):
            return "exit"
        if args.get("stairwell_id"):
            return "stairwell"
        to_room_id = str(args.get("to_room_id", ""))
        if to_room_id.startswith(("exit_", "exit-", "EX", "stairwell_", "stair_")):
            return "legacy_egress_alias"
        if to_room_id:
            return "room"
        return "missing"

    metric_samples = [sample for sample in samples if _is_metric_action_sample(sample)]
    valid_metric_samples = [sample for sample in metric_samples if _is_valid_action(sample)]
    floor_samples = [sample for sample in valid_metric_samples if getattr(sample, "role", "") == "floor_agent"]
    orchestrator_samples = [
        sample for sample in valid_metric_samples if getattr(sample, "role", "") == "orchestrator"
    ]
    wait_count = sum(1 for sample in valid_metric_samples if _is_wait_action(sample))
    floor_wait_count = sum(1 for sample in floor_samples if _is_wait_action(sample))
    orchestrator_wait_count = sum(
        1 for sample in orchestrator_samples if _is_wait_action(sample)
    )
    empty_args_count = sum(1 for sample in valid_metric_samples if _has_empty_arguments(sample))
    floor_active_count = sum(1 for sample in floor_samples if not _is_wait_action(sample))
    active_empty_args_count = sum(
        1
        for sample in valid_metric_samples
        if not _is_wait_action(sample) and _has_empty_arguments(sample)
    )
    valid_but_hollow_count = sum(
        1
        for sample in valid_metric_samples
        if _is_valid_action(sample)
        and _is_wait_action(sample)
        and _has_empty_arguments(sample)
    )
    floor_route_samples = [sample for sample in floor_samples if _is_floor_route(sample)]
    floor_route_count = len(floor_route_samples)
    floor_scout_count = sum(
        1 for sample in floor_samples if _action_type(sample) == "scout"
    )
    floor_evacuate_count = sum(
        1 for sample in floor_samples if _action_type(sample) == "evacuate_floor_priority"
    )
    floor_route_exit_count = sum(1 for sample in floor_route_samples if _route_target_kind(sample) == "exit")
    floor_route_stairwell_count = sum(1 for sample in floor_route_samples if _route_target_kind(sample) == "stairwell")
    floor_route_room_count = sum(1 for sample in floor_route_samples if _route_target_kind(sample) == "room")
    floor_route_legacy_alias_count = sum(
        1 for sample in floor_route_samples if _route_target_kind(sample) == "legacy_egress_alias"
    )
    floor_route_missing_target_count = sum(
        1 for sample in floor_route_samples if _route_target_kind(sample) == "missing"
    )
    priority_action_count = sum(
        1
        for sample in orchestrator_samples
        if _action_type(sample) == "evacuate_floor_priority"
    )
    priority_directive_issue_count = sum(
        int(getattr(result, "priority_directive_issue_count", 0))
        for result in results
    )
    priority_component_keys = (
        "priority_top_match",
        "priority_rank_score",
        "priority_coverage",
        "priority_effect_bonus",
    )
    priority_component_totals: dict[str, float] = {key: 0.0 for key in priority_component_keys}
    priority_component_counts: dict[str, int] = {key: 0 for key in priority_component_keys}
    for result in results:
        totals = getattr(result, "priority_component_totals", {}) or {}
        counts = getattr(result, "priority_component_counts", {}) or {}
        if not isinstance(totals, dict) or not isinstance(counts, dict):
            continue
        for key in priority_component_keys:
            priority_component_totals[key] += float(totals.get(key, 0.0) or 0.0)
            priority_component_counts[key] += int(counts.get(key, 0) or 0)
    family_counts = {
        "fire": 0,
        "flood": 0,
        "gas": 0,
    }
    for result in results:
        family = str(getattr(result, "disaster_family", ""))
        if family in family_counts:
            family_counts[family] += 1

    def _priority_component_mean(key: str) -> float:
        return round(
            priority_component_totals[key] / max(priority_component_counts[key], 1),
            4,
        )

    return {
        "wait_rate": round(wait_count / max(len(valid_metric_samples), 1), 4),
        "floor_agent_wait_rate": round(floor_wait_count / max(len(floor_samples), 1), 4),
        "orchestrator_wait_rate": round(
            orchestrator_wait_count / max(len(orchestrator_samples), 1), 4
        ),
        "empty_args_rate": round(empty_args_count / max(len(valid_metric_samples), 1), 4),
        "floor_agent_active_action_rate": round(
            floor_active_count / max(len(floor_samples), 1), 4
        ),
        "active_empty_args_rate": round(active_empty_args_count / max(len(valid_metric_samples), 1), 4),
        "valid_but_hollow_action_rate": round(
            valid_but_hollow_count / max(len(valid_metric_samples), 1),
            4,
        ),
        "floor_scout_action_rate": round(
            floor_scout_count / max(len(floor_samples), 1),
            4,
        ),
        "floor_route_action_rate": round(floor_route_count / max(len(floor_samples), 1), 4),
        "floor_evacuate_action_rate": round(
            floor_evacuate_count / max(len(floor_samples), 1),
            4,
        ),
        "floor_route_exit_rate": round(floor_route_exit_count / max(floor_route_count, 1), 4),
        "floor_route_stairwell_rate": round(
            floor_route_stairwell_count / max(floor_route_count, 1),
            4,
        ),
        "floor_route_room_rate": round(floor_route_room_count / max(floor_route_count, 1), 4),
        "floor_route_legacy_egress_alias_rate": round(
            floor_route_legacy_alias_count / max(floor_route_count, 1),
            4,
        ),
        "floor_route_missing_target_rate": round(
            floor_route_missing_target_count / max(floor_route_count, 1),
            4,
        ),
        "override_rate": round(override_count / max(orchestrator_action_count, 1), 4),
        "override_win_rate": round(override_win_count / max(override_count, 1), 4),
        "rationale_bonus_mean": round(rationale_bonus_total / max(rationale_bonus_count, 1), 4),
        "priority_top_match_mean": _priority_component_mean("priority_top_match"),
        "priority_rank_score_mean": _priority_component_mean("priority_rank_score"),
        "priority_coverage_mean": _priority_component_mean("priority_coverage"),
        "priority_effect_bonus_mean": _priority_component_mean("priority_effect_bonus"),
        "priority_action_rate": round(
            priority_action_count / max(len(orchestrator_samples), 1),
            4,
        ),
        "priority_directive_issue_rate": round(
            priority_directive_issue_count / max(priority_action_count, 1),
            4,
        ),
        "family_fire_fraction": round(family_counts["fire"] / max(len(results), 1), 4),
        "family_flood_fraction": round(family_counts["flood"] / max(len(results), 1), 4),
        "family_gas_fraction": round(family_counts["gas"] / max(len(results), 1), 4),
    }


def _population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _training_watchdog_reason(
    *,
    step: int,
    metrics_row: dict[str, Any],
    config: TrainingConfig,
    zero_signal_streak: int,
) -> str | None:
    """Return a stop reason when a paid run is no longer producing signal."""
    watchdog = config.watchdog
    if not watchdog.enabled or step + 1 <= watchdog.warmup_steps:
        return None

    if zero_signal_streak >= watchdog.zero_signal_window:
        return (
            "zero_grpo_signal:"
            f"advantage_std={_as_float(metrics_row, 'advantage_std'):.6g},"
            f"group_raw_reward_std_mean={_as_float(metrics_row, 'group_raw_reward_std_mean'):.6g},"
            f"policy_loss={_as_float(metrics_row, 'policy_loss'):.6g},"
            f"streak={zero_signal_streak}"
        )

    hollow_rate = _as_float(metrics_row, "valid_but_hollow_action_rate")
    if hollow_rate >= watchdog.max_valid_but_hollow_action_rate:
        return (
            "valid_but_hollow_dominance:"
            f"valid_but_hollow_action_rate={hollow_rate:.4f}"
        )

    scout_rate = _as_float(metrics_row, "floor_scout_action_rate")
    route_rate = _as_float(metrics_row, "floor_route_action_rate")
    if (
        scout_rate >= watchdog.max_floor_scout_action_rate
        and route_rate < watchdog.min_floor_route_action_rate
    ):
        return (
            "scout_dominance:"
            f"floor_scout_action_rate={scout_rate:.4f},"
            f"floor_route_action_rate={route_rate:.4f}"
        )

    if step + 1 >= watchdog.route_required_after_steps:
        evacuate_rate = _as_float(metrics_row, "floor_evacuate_action_rate")
        if route_rate < watchdog.min_floor_route_action_rate and evacuate_rate <= 0.0:
            return (
                "route_starvation:"
                f"floor_route_action_rate={route_rate:.4f},"
                f"floor_evacuate_action_rate={evacuate_rate:.4f}"
            )

    return None


def _copy_adapter_tree(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)


def _copy_floor_specialist_adapter_trees(
    specialist_adapter_paths_to_copy: dict[str, Path] | None,
    target_dir: Path,
) -> dict[str, Path] | None:
    if not specialist_adapter_paths_to_copy:
        return None

    saved_paths: dict[str, Path] = {}
    for family, source_dir in specialist_adapter_paths_to_copy.items():
        family_dir = target_dir / family
        _copy_adapter_tree(source_dir, family_dir)
        saved_paths[family] = family_dir
    return saved_paths


def _save_adapter_weights(
    policy: Any,
    target_dir: Path,
    *,
    roles_to_save: set[str] | None = None,
    role_adapter_paths_to_copy: dict[str, Path] | None = None,
) -> dict[str, Path] | None:
    role_policies = getattr(policy, "_role_policies", None)
    if role_policies is not None:
        saved_paths: dict[str, Path] = {}
        role_adapter_paths_to_copy = role_adapter_paths_to_copy or {}
        for role, role_policy in role_policies.items():
            role_dir = target_dir / role
            frozen_source = role_adapter_paths_to_copy.get(role)
            if frozen_source is not None:
                _copy_adapter_tree(frozen_source, role_dir)
                saved_paths[role] = role_dir
                continue
            if roles_to_save is not None and role not in roles_to_save:
                continue
            model = getattr(role_policy, "_model", None)
            save_pretrained = getattr(model, "save_pretrained", None)
            if callable(save_pretrained):
                role_dir.mkdir(parents=True, exist_ok=True)
                save_pretrained(str(role_dir))
                saved_paths[role] = role_dir
        return saved_paths or None
    model = getattr(policy, "_model", None)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        target_dir.mkdir(parents=True, exist_ok=True)
        save_pretrained(str(target_dir))
    return None


def _extract_optimizer_state(trainer: Any) -> tuple[dict | None, dict[str, dict] | None]:
    role_trainers = getattr(trainer, "_role_trainers", None)
    if role_trainers is not None:
        role_states: dict[str, dict] = {}
        for role, role_trainer in role_trainers.items():
            optimizer = getattr(role_trainer, "optimizer", None)
            state_dict = getattr(optimizer, "state_dict", None)
            if callable(state_dict):
                role_states[role] = state_dict()
        return None, (role_states or None)

    optimizer = getattr(trainer, "optimizer", None)
    state_dict = getattr(optimizer, "state_dict", None)
    if callable(state_dict):
        return state_dict(), None
    return None, None


def _maybe_init_wandb(
    config: TrainingConfig,
    *,
    run_id: str | None = None,
) -> Any | None:
    """Initialize wandb if WANDB_API_KEY is set and wandb is importable.

    Returns the wandb run handle, or None if either precondition is unmet.
    Silent no-op on missing env var or missing package — this is how the HF
    backend path stays green when wandb isn't installed locally.

    When *run_id* is provided, the run is resumed via ``id=run_id,
    resume="allow"``.
    """
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        return None

    project = os.environ.get("WANDB_PROJECT", "evacos-ma").strip() or "evacos-ma"
    run_name = os.environ.get("WANDB_RUN_NAME", "").strip() or None

    wandb.login(key=api_key)
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": config.model_dump(mode="json"),
        "settings": wandb.Settings(start_method="thread"),
    }
    if run_id is not None:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
    return wandb.init(**init_kwargs)


def _build_policy(
    config: TrainingConfig,
    bundle: Any | None,
    *,
    LoraConfig: Any,
) -> Any:
    resolved_models = _resolved_role_model_names(config)
    shared_model_name = resolved_models["orchestrator"]

    lora_adapter_path: str | None = None
    frozen_adapter_paths = _frozen_adapter_paths_for_run(config, bundle)
    floor_specialist_adapter_paths = _frozen_floor_specialist_adapter_paths_for_run(
        config,
        bundle,
    )
    role_lora_adapter_paths: dict[str, str] = {
        role: str(path) for role, path in frozen_adapter_paths.items()
    }
    if bundle is not None:
        if not bundle.lora_weights_path.exists():
            raise RuntimeError(
                f"Checkpoint bundle references adapter at "
                f"{bundle.lora_weights_path!s} but that directory does not exist. "
                f"Cannot resume from a corrupt or partial checkpoint."
            )
        lora_adapter_path = str(bundle.lora_weights_path)
        bundle_role_paths = getattr(bundle, "role_lora_weights_paths", None) or {}
        if bundle_role_paths:
            for role, path in bundle_role_paths.items():
                if not path.exists():
                    raise RuntimeError(
                        f"Checkpoint bundle references {role} adapter at "
                        f"{path!s} but that directory does not exist."
                    )
                role_lora_adapter_paths[role] = str(path)
        elif config.uses_role_routed_policy:
            for role in _model_backed_trainable_roles(config):
                role_dir = bundle.lora_weights_path / role
                if role_dir.exists():
                    role_lora_adapter_paths[role] = str(role_dir)

        missing_role_paths = [
            role
            for role in _model_backed_trainable_roles(config)
            if role not in role_lora_adapter_paths
        ]
        if config.uses_role_routed_policy and missing_role_paths:
            raise RuntimeError(
                "Checkpoint bundle is missing adapter weights for trainable role(s): "
                f"{missing_role_paths!r}"
            )

    backend = config.backend
    sampling_temperature = float(config.rollout.sampling_temperature)
    do_sample = sampling_temperature > 0.0
    if backend == "unsloth":
        from training.policy_adapter import (
            RoleRoutedPolicy,
            ScopeRoutedFloorPolicy,
            StubPolicy,
            unsloth_policy_factory,
        )

        if config.uses_role_routed_policy:
            if config.policy_for_role("orchestrator") == "stub":
                orchestrator_policy = StubPolicy(seed=config.seed.training_rng)
            else:
                orchestrator_policy = unsloth_policy_factory(
                    resolved_models["orchestrator"],
                    lora_adapter_path=role_lora_adapter_paths.get("orchestrator"),
                    lora_r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    lora_target_modules=list(config.lora.target_modules),
                    max_seq_length=config.unsloth_max_seq_length,
                    load_in_4bit=config.load_in_4bit,
                    dtype=config.model.dtype,
                    max_prompt_tokens=config.model.max_prompt_tokens,
                    use_vllm=config.rollout.use_vllm,
                    max_new_tokens=config.model.max_completion_tokens,
                    temperature=sampling_temperature,
                    seed=config.seed.training_rng,
                )
            def _build_unsloth_floor_policy(adapter_path: str | None) -> Any:
                return unsloth_policy_factory(
                    resolved_models["floor_agent"],
                    lora_adapter_path=adapter_path,
                    lora_r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    lora_target_modules=list(config.lora.target_modules),
                    max_seq_length=config.unsloth_max_seq_length,
                    load_in_4bit=config.load_in_4bit,
                    dtype=config.model.dtype,
                    max_prompt_tokens=config.model.max_prompt_tokens,
                    use_vllm=config.rollout.use_vllm,
                    max_new_tokens=config.model.max_completion_tokens,
                    temperature=sampling_temperature,
                    seed=config.seed.training_rng,
                )

            if floor_specialist_adapter_paths:
                floor_policy = ScopeRoutedFloorPolicy(
                    specialist_policies={
                        family: _build_unsloth_floor_policy(str(path))
                        for family, path in floor_specialist_adapter_paths.items()
                    },
                    generalist_policy=(
                        _build_unsloth_floor_policy(role_lora_adapter_paths["floor_agent"])
                        if "floor_agent" in role_lora_adapter_paths
                        else None
                    ),
                )
            else:
                floor_policy = _build_unsloth_floor_policy(
                    role_lora_adapter_paths.get("floor_agent")
                )
            return RoleRoutedPolicy(
                orchestrator_policy=orchestrator_policy,
                floor_policy=floor_policy,
            )

        return unsloth_policy_factory(
            shared_model_name,
            lora_adapter_path=lora_adapter_path,
            lora_r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_target_modules=list(config.lora.target_modules),
            max_seq_length=config.unsloth_max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=config.model.dtype,
            max_prompt_tokens=config.model.max_prompt_tokens,
            use_vllm=config.rollout.use_vllm,
            max_new_tokens=config.model.max_completion_tokens,
            temperature=sampling_temperature,
            seed=config.seed.training_rng,
        )
    if backend == "hf":
        from training.policy_adapter import (
            RoleRoutedPolicy,
            ScopeRoutedFloorPolicy,
            StubPolicy,
            hf_policy_factory,
        )

        peft_config = None
        if lora_adapter_path is None:
            peft_config = LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                target_modules=list(config.lora.target_modules),
                lora_dropout=config.lora.dropout,
            )

        if config.uses_role_routed_policy:
            orchestrator_peft = None
            floor_peft = None
            if (
                config.is_role_trainable("orchestrator")
                and config.policy_for_role("orchestrator") == "model"
                and role_lora_adapter_paths.get("orchestrator") is None
            ):
                orchestrator_peft = LoraConfig(
                    r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    target_modules=list(config.lora.target_modules),
                    lora_dropout=config.lora.dropout,
                )
            if (
                config.is_role_trainable("floor_agent")
                and role_lora_adapter_paths.get("floor_agent") is None
            ):
                floor_peft = LoraConfig(
                    r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    target_modules=list(config.lora.target_modules),
                    lora_dropout=config.lora.dropout,
                )
            if config.policy_for_role("orchestrator") == "stub":
                orchestrator_policy = StubPolicy(seed=config.seed.training_rng)
            else:
                orchestrator_policy = hf_policy_factory(
                    resolved_models["orchestrator"],
                    lora_adapter_path=role_lora_adapter_paths.get("orchestrator"),
                    peft_config=orchestrator_peft,
                    torch_dtype=config.model.dtype,
                    max_prompt_tokens=config.model.max_prompt_tokens,
                    max_new_tokens=config.model.max_completion_tokens,
                    do_sample=do_sample,
                    temperature=sampling_temperature,
                )
            def _build_hf_floor_policy(
                adapter_path: str | None,
                peft_config_for_policy: Any | None = None,
            ) -> Any:
                return hf_policy_factory(
                    resolved_models["floor_agent"],
                    lora_adapter_path=adapter_path,
                    peft_config=peft_config_for_policy,
                    torch_dtype=config.model.dtype,
                    max_prompt_tokens=config.model.max_prompt_tokens,
                    max_new_tokens=config.model.max_completion_tokens,
                    do_sample=do_sample,
                    temperature=sampling_temperature,
                )

            if floor_specialist_adapter_paths:
                floor_policy = ScopeRoutedFloorPolicy(
                    specialist_policies={
                        family: _build_hf_floor_policy(str(path))
                        for family, path in floor_specialist_adapter_paths.items()
                    },
                    generalist_policy=(
                        _build_hf_floor_policy(role_lora_adapter_paths["floor_agent"])
                        if "floor_agent" in role_lora_adapter_paths
                        else None
                    ),
                )
            else:
                floor_policy = _build_hf_floor_policy(
                    role_lora_adapter_paths.get("floor_agent"),
                    floor_peft,
                )
            return RoleRoutedPolicy(
                orchestrator_policy=orchestrator_policy,
                floor_policy=floor_policy,
            )

        return hf_policy_factory(
            shared_model_name,
            lora_adapter_path=lora_adapter_path,
            peft_config=peft_config,
            torch_dtype=config.model.dtype,
            max_prompt_tokens=config.model.max_prompt_tokens,
            max_new_tokens=config.model.max_completion_tokens,
            do_sample=do_sample,
            temperature=sampling_temperature,
        )
    raise ValueError(f"Unknown training.backend: {backend!r}")


def _run_eval(
    *,
    env: Any,
    policy: Any,
    config: TrainingConfig,
    normalizer: Any,
    checkpoint_tag: str,
    model_name: str,
    collect_batch: Any,
    disaster_families: list[Any],
    jsonl_dir: Path,
) -> list[Any]:
    eval_results: list[Any] = []
    family_count = len(disaster_families)
    reward_config = config.reward.model_dump(mode="python")
    restore_tokens = _set_policy_sampling_temperature(policy, 0.0)
    try:
        for tier in config.eval.tiers:
            fixed_curriculum = _FixedTierCurriculum(tier)
            seeds: list[int] = []
            for seed in config.eval.seeds:
                seeds.extend([seed] * family_count)
            seed_iter = iter(seeds)
            eval_results.extend(
                collect_batch(
                    env,
                    policy,
                    fixed_curriculum,
                    num_episodes=len(seeds),
                    seed_generator=lambda: next(seed_iter),
                    disaster_families=disaster_families,
                    max_rounds=config.rollout.max_rounds_per_episode,
                    checkpoint_tag=checkpoint_tag,
                    model_name=model_name,
                    is_eval=True,
                    normalizer=normalizer,
                    jsonl_dir=jsonl_dir,
                    seed_collision_retry_limit=config.rollout.seed_retry_limit,
                    rationale_mode=config.reward.rationale_scaling,
                    reward_config=reward_config,
                )
            )
    finally:
        _restore_policy_sampling_temperature(restore_tokens)
    return eval_results


def run_training(config_path: Path = Path("training/config.yaml")) -> None:
    """Run the Phase 7 training loop."""
    import logging

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    raw = _load_yaml_config(config_path)
    config = TrainingConfig(**raw)
    _validate_config_path_identity(config_path, config)
    logger.info(
        "Reward normalization: group-mean-std on raw_rewards (trainer-owned). "
        "RoleReward.normalized and TrajectorySample.normalized_reward are identity "
        "passthroughs retained for trace compatibility."
    )
    reward_config = config.reward.model_dump(mode="python")

    if config.backend == "unsloth":
        try:
            # Import Unsloth before TRL / transformers / PEFT so its monkey
            # patches apply to the training stack as intended.
            import unsloth  # type: ignore  # noqa: F401
        except ImportError as exc:
            if "HybridCache" in str(exc):
                import sys

                patch_transformers_cache_exports()
                for module_name in list(sys.modules):
                    if module_name == "unsloth" or module_name.startswith("unsloth."):
                        sys.modules.pop(module_name, None)
                try:
                    import unsloth  # type: ignore  # noqa: F401
                except ImportError:
                    pass
            else:
                pass

    patch_transformers_cache_exports()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed. Install requirements-training.txt and retry."
        ) from exc

    try:
        from trl import GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "trl is not installed. Install requirements-training.txt and retry."
        ) from exc

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError(
            "peft is not installed. Install requirements-training.txt and retry."
        ) from exc

    from curriculum.controller import CurriculumController
    from evacos_ma.env import EvacEnvironment
    from evacos_ma.models import DisasterType

    from training.checkpoint import (
        acquire_run_output_lock,
        CheckpointBundle,
        atomic_replace_latest,
        load_checkpoint,
        rotate_checkpoints,
        save_checkpoint,
    )
    from training.reward import RewardNormalizer
    from training.rollout import collect_batch

    ckpt_root = Path(config.checkpoint.root_dir)
    metrics_path = Path(config.metrics.csv_path)
    jsonl_dir = Path(config.metrics.jsonl_dir)

    ckpt_root.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_run_output_lock(ckpt_root, metrics_path, jsonl_dir)

    rng = random.Random(config.seed.training_rng)
    torch.manual_seed(config.seed.training_rng)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed.training_rng)

    bundle = load_checkpoint(ckpt_root)
    start_step = 0
    wall_total = 0.0
    if bundle is not None:
        start_step = bundle.step + 1
        wall_total = bundle.wall_seconds_total
        rng.setstate(pickle.loads(bundle.rollout_rng_state))

    # Restore torch RNG states when resuming
    if bundle is not None and bundle.torch_rng_state is not None:
        torch.set_rng_state(pickle.loads(bundle.torch_rng_state))
    if bundle is not None and bundle.torch_cuda_rng_state is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(pickle.loads(bundle.torch_cuda_rng_state))

    wandb_run = _maybe_init_wandb(
        config,
        run_id=bundle.wandb_run_id if bundle is not None else None,
    )

    tier_schedule = config.rollout.expanded_tier_schedule()
    if tier_schedule:
        curriculum = _ScheduledTierCurriculum(tier_schedule)
    else:
        curriculum = CurriculumController()
    normalizer = RewardNormalizer()
    if bundle is not None:
        curriculum.load_snapshot(bundle.curriculum_snapshot)
        normalizer.load_snapshot(bundle.normalizer_snapshot)

    env = EvacEnvironment()
    policy = _build_policy(config, bundle, LoraConfig=LoraConfig)
    trainer = _build_grpo_trainer(
        GRPOTrainer,
        policy,
        config,
        optimizer_state=bundle.optimizer_state if bundle is not None else None,
        role_optimizer_states=bundle.role_optimizer_states if bundle is not None else None,
    )
    checkpoint_role_model_names = _checkpoint_role_model_names(config)
    checkpoint_orchestrator_policy = _checkpoint_orchestrator_policy(config)
    model_name = _resolved_model_name_tag(config)
    config_hash = _config_hash(config)
    disaster_families = [DisasterType(item) for item in config.rollout.disaster_families]

    stop_requested = False
    last_completed_step = start_step - 1
    last_metrics_checkpoint_step = bundle.step if bundle is not None else start_step - 1
    zero_signal_streak = 0

    def _signal_handler(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _signal_handler)

    def _seed_gen() -> int:
        return rng.randint(0, 2_147_483_647)

    def _metrics_step(row: dict[str, Any]) -> int | None:
        try:
            return int(row.get("step", ""))
        except (TypeError, ValueError):
            return None

    def _floatish(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _recent_means(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
        recent = rows[-10:]
        means: dict[str, float] = {}
        for key in keys:
            values = [
                parsed
                for row in recent
                if (parsed := _floatish(row.get(key))) is not None
            ]
            if values:
                means[key] = round(sum(values) / len(values), 6)
        return means

    def _write_checkpoint_metrics_snapshot(step: int, ckpt_dir: Path) -> None:
        """Save plot-friendly metrics slices beside each checkpoint."""
        nonlocal last_metrics_checkpoint_step

        window_path = ckpt_dir / "metrics_window.csv"
        if step <= last_metrics_checkpoint_step and window_path.exists():
            return

        rows = read_training_metrics_rows(metrics_path)
        to_date = [
            row
            for row in rows
            if (row_step := _metrics_step(row)) is not None and row_step <= step
        ]
        window = [
            row
            for row in to_date
            if (row_step := _metrics_step(row)) is not None
            and row_step > last_metrics_checkpoint_step
        ]

        write_training_metrics_rows(ckpt_dir / "metrics_to_date.csv", to_date)
        write_training_metrics_rows(window_path, window)

        summary = {
            "checkpoint_step": step,
            "previous_checkpoint_step": last_metrics_checkpoint_step,
            "window_row_count": len(window),
            "to_date_row_count": len(to_date),
            "window_start_step": _metrics_step(window[0]) if window else None,
            "window_end_step": _metrics_step(window[-1]) if window else None,
            "source_metrics_csv": str(metrics_path),
            "latest_metrics": to_date[-1] if to_date else {},
            "last_10_means": _recent_means(
                to_date,
                (
                    "mean_raw_reward_floor",
                    "mean_norm_reward_floor",
                    "invalid_action_rate",
                    "floor_agent_group_raw_reward_std_mean",
                    "floor_agent_advantage_std",
                    "floor_agent_policy_loss",
                    "policy_loss",
                ),
            ),
        }
        with open(ckpt_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        last_metrics_checkpoint_step = step

    def _write_checkpoint(step: int) -> None:
        ckpt_dir = ckpt_root / f"ckpt_{step}"
        adapter_path = ckpt_dir / "lora_adapter"

        # 1. Write adapter weights into ckpt_N/lora_adapter (durable first)
        role_adapter_paths = _save_adapter_weights(
            policy,
            adapter_path,
            roles_to_save=(
                set(_model_backed_trainable_roles(config))
                if config.uses_role_routed_policy
                else None
            ),
            role_adapter_paths_to_copy=(
                _frozen_adapter_paths_for_run(config, bundle)
                if config.uses_role_routed_policy
                else None
            ),
        )
        floor_specialist_adapter_paths = _copy_floor_specialist_adapter_trees(
            _frozen_floor_specialist_adapter_paths_for_run(config, bundle),
            adapter_path / "floor_agent" / "specialists",
        )

        # 2. Build the bundle pointing at the durable adapter path
        import torch as _torch_mod

        torch_rng_bytes: bytes | None = None
        torch_cuda_rng_bytes: bytes | None = None
        try:
            torch_rng_bytes = pickle.dumps(_torch_mod.get_rng_state())
            if _torch_mod.cuda.is_available():
                torch_cuda_rng_bytes = pickle.dumps(
                    _torch_mod.cuda.get_rng_state_all()
                )
        except Exception:
            pass

        optimizer_state, role_optimizer_states = _extract_optimizer_state(trainer)

        new_bundle = CheckpointBundle(
            step=step,
            wall_seconds_total=wall_total,
            curriculum_snapshot=curriculum.snapshot(),
            normalizer_snapshot=normalizer.snapshot(),
            rollout_rng_state=pickle.dumps(rng.getstate()),
            lora_weights_path=adapter_path,
            model_name=model_name,
            config_hash=config_hash,
            config_path=str(config_path),
            max_steps=config.max_steps,
            rollout_max_rounds_per_episode=config.rollout.max_rounds_per_episode,
            rollout_disaster_families=list(config.rollout.disaster_families),
            rollout_tier_schedule=[
                block.model_dump(mode="json")
                for block in (config.rollout.tier_schedule or [])
            ]
            or None,
            role_lora_weights_paths=role_adapter_paths,
            floor_specialist_lora_weights_paths=floor_specialist_adapter_paths,
            role_model_names=checkpoint_role_model_names,
            orchestrator_policy=checkpoint_orchestrator_policy,
            optimizer_state=optimizer_state,
            role_optimizer_states=role_optimizer_states,
            torch_rng_state=torch_rng_bytes,
            torch_cuda_rng_state=torch_cuda_rng_bytes,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
        )

        # 3. Write meta.json, RNG files, optimizer state into ckpt_N
        save_checkpoint(ckpt_root, new_bundle)
        _write_checkpoint_metrics_snapshot(step, ckpt_dir)

        # 4. Publish latest/ last (atomic)
        atomic_replace_latest(ckpt_root, ckpt_dir)

        # 5. Rotate old checkpoints
        rotate_checkpoints(ckpt_root, config.checkpoint.keep_last_n)

    try:
        step_cap = config.max_steps if config.max_steps is not None else 100_000
        for step in range(start_step, step_cap):
            if stop_requested:
                break

            step_started = time.monotonic()
            set_step = getattr(curriculum, "set_step", None)
            if set_step is not None:
                set_step(step)
            checkpoint_tag = f"ckpt_{step}" if step > 0 else "baseline"
            results = collect_batch(
                env,
                policy,
                curriculum=curriculum,
                num_episodes=config.rollout.episodes_per_step,
                seed_generator=_seed_gen,
                disaster_families=disaster_families,
                max_rounds=config.rollout.max_rounds_per_episode,
                checkpoint_tag=checkpoint_tag,
                model_name=model_name,
                is_eval=False,
                normalizer=normalizer,
                jsonl_dir=jsonl_dir,
                seed_collision_retry_limit=config.rollout.seed_retry_limit,
                rationale_mode=config.reward.rationale_scaling,
                reward_config=reward_config,
                candidates_per_floor_prompt=config.rollout.candidates_per_floor_prompt,
                include_oracle_floor_candidate=config.rollout.include_oracle_floor_candidate,
            )

            grouped_inputs = _group_for_grpo(results)
            trainer_diagnostics = _call_trainer_step(trainer, grouped_inputs) or {}

            step_wall = time.monotonic() - step_started
            wall_total += step_wall

            all_raw_orch: list[float] = []
            all_raw_floor: list[float] = []
            all_norm_orch: list[float] = []
            all_norm_floor: list[float] = []
            invalid_count = 0
            total_samples = 0
            invalid_by_role: dict[str, int] = {"orchestrator": 0, "floor_agent": 0}
            total_by_role: dict[str, int] = {"orchestrator": 0, "floor_agent": 0}
            for result in results:
                all_raw_orch.append(result.total_raw_reward_by_role.get("orchestrator", 0.0))
                all_norm_orch.append(result.total_normalized_reward_by_role.get("orchestrator", 0.0))
                for agent_id, value in result.total_raw_reward_by_role.items():
                    if agent_id != "orchestrator":
                        all_raw_floor.append(value)
                for agent_id, value in result.total_normalized_reward_by_role.items():
                    if agent_id != "orchestrator":
                        all_norm_floor.append(value)
                for sample in result.samples:
                    total_samples += 1
                    role = "orchestrator" if getattr(sample, "role", "") == "orchestrator" else "floor_agent"
                    total_by_role[role] = total_by_role.get(role, 0) + 1
                    if sample.parsed_action.get("fallback_reason"):
                        invalid_count += 1
                        invalid_by_role[role] = invalid_by_role.get(role, 0) + 1

            rollout_metrics = _compute_rollout_metrics(results)
            metrics_row = {
                "step": step,
                "max_steps": config.max_steps,
                "wall_seconds": round(wall_total, 2),
                "run_name": ckpt_root.name,
                "config_hash": config_hash,
                "tier_mix": ";".join(sorted({result.tier for result in results})),
                "disaster_families": ";".join(item.value for item in disaster_families),
                "episodes_per_step": config.rollout.episodes_per_step,
                "max_rounds_per_episode": config.rollout.max_rounds_per_episode,
                "candidates_per_floor_prompt": config.rollout.candidates_per_floor_prompt,
                "include_oracle_floor_candidate": config.rollout.include_oracle_floor_candidate,
                "sampling_temperature": config.rollout.sampling_temperature,
                "mean_raw_reward_orch": round(sum(all_raw_orch) / max(len(all_raw_orch), 1), 4),
                "mean_raw_reward_floor": round(sum(all_raw_floor) / max(len(all_raw_floor), 1), 4),
                "raw_reward_std_orch": round(_population_std(all_raw_orch), 4),
                "raw_reward_std_floor": round(_population_std(all_raw_floor), 4),
                "mean_norm_reward_orch": round(sum(all_norm_orch) / max(len(all_norm_orch), 1), 4),
                "mean_norm_reward_floor": round(sum(all_norm_floor) / max(len(all_norm_floor), 1), 4),
                "norm_reward_std_orch": round(_population_std(all_norm_orch), 4),
                "norm_reward_std_floor": round(_population_std(all_norm_floor), 4),
                "invalid_action_rate": round(invalid_count / max(total_samples, 1), 4),
                "orchestrator_invalid_action_rate": round(
                    invalid_by_role.get("orchestrator", 0)
                    / max(total_by_role.get("orchestrator", 0), 1),
                    4,
                ),
                "floor_agent_invalid_action_rate": round(
                    invalid_by_role.get("floor_agent", 0)
                    / max(total_by_role.get("floor_agent", 0), 1),
                    4,
                ),
                "orchestrator_invalid_action_count": invalid_by_role.get("orchestrator", 0),
                "floor_agent_invalid_action_count": invalid_by_role.get("floor_agent", 0),
                **rollout_metrics,
                "episodes_seen": (step + 1) * config.rollout.episodes_per_step,
            }
            _merge_trainer_diagnostics_into_metrics(metrics_row, trainer_diagnostics)

            watchdog = config.watchdog
            if watchdog.enabled and step + 1 > watchdog.warmup_steps:
                no_advantage = (
                    _as_float(metrics_row, "advantage_std")
                    <= watchdog.min_advantage_std
                )
                no_reward_contrast = (
                    _as_float(metrics_row, "group_raw_reward_std_mean")
                    <= watchdog.min_group_raw_reward_std_mean
                )
                no_policy_update = abs(_as_float(metrics_row, "policy_loss")) <= 1e-12
                zero_signal_streak = (
                    zero_signal_streak + 1
                    if no_advantage and no_reward_contrast and no_policy_update
                    else 0
                )
            else:
                zero_signal_streak = 0

            watchdog_reason = _training_watchdog_reason(
                step=step,
                metrics_row=metrics_row,
                config=config,
                zero_signal_streak=zero_signal_streak,
            )
            metrics_row["watchdog_status"] = (
                "triggered" if watchdog_reason else "healthy"
            )
            metrics_row["watchdog_reason"] = watchdog_reason or ""
            append_training_metrics_row(metrics_path, metrics_row)
            if wandb_run is not None:
                wandb_run.log(metrics_row, step=step)

            last_completed_step = step

            if watchdog_reason:
                write_trace_row(
                    jsonl_dir / "training_watchdog.jsonl",
                    {
                        "step": step,
                        "run_name": ckpt_root.name,
                        "status": "triggered",
                        "reason": watchdog_reason,
                        "zero_signal_streak": zero_signal_streak,
                        "metrics": {
                            key: metrics_row.get(key)
                            for key in (
                                "policy_loss",
                                "advantage_std",
                                "group_raw_reward_std_mean",
                                "valid_but_hollow_action_rate",
                                "floor_scout_action_rate",
                                "floor_route_action_rate",
                                "floor_evacuate_action_rate",
                            )
                        },
                    },
                )
                if config.watchdog.abort_on_trigger:
                    raise RuntimeError(
                        "Training watchdog stopped this run before more GPU "
                        f"time was wasted: {watchdog_reason}"
                    )

            if (step + 1) % config.eval.every_steps == 0:
                _run_eval(
                    env=env,
                    policy=policy,
                    config=config,
                    normalizer=normalizer,
                    checkpoint_tag=checkpoint_tag,
                    model_name=model_name,
                    collect_batch=collect_batch,
                    disaster_families=disaster_families,
                    jsonl_dir=jsonl_dir,
                )

            if (step + 1) % config.checkpoint.every_steps == 0:
                _write_checkpoint(step)

            if stop_requested:
                break

    except KeyboardInterrupt:
        stop_requested = True
    finally:
        try:
            # Write a final checkpoint only if we made net-new progress.
            # Avoid duplicating an already-on-disk checkpoint when zero steps
            # completed after resume.
            if last_completed_step >= start_step:
                _write_checkpoint(last_completed_step)
            if wandb_run is not None:
                wandb_run.finish()
        finally:
            run_lock.release()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m training.train [config.yaml]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run EvacOS2 GRPO training.")
    parser.add_argument(
        "config",
        nargs="?",
        default="training/config.yaml",
        help="Path to the training YAML config.",
    )
    args = parser.parse_args(argv)
    run_training(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
