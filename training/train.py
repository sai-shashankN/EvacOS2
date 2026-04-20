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
import signal
import time
from pathlib import Path
from typing import Any

from training.config_schema import TrainingConfig
from training.metrics import append_training_metrics_row


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


class _FixedTierCurriculum:
    """Tiny curriculum shim for eval sweeps over an explicit tier."""

    def __init__(self, tier: str) -> None:
        self._tier = tier

    def suggest_next_tier(self, disaster_family: str) -> str:
        return self._tier

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:
        return None


def _extract_trainer_model(policy: Any) -> Any:
    return getattr(policy, "_model", policy)


def _extract_trainer_tokenizer(policy: Any) -> Any:
    return getattr(policy, "_tokenizer", None)


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
        group_size: int,
        num_train_epochs_per_step: int,
    ) -> None:
        import torch

        if tokenizer is None:
            raise RuntimeError("MultiAgentGRPOTrainer requires a tokenizer")

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.kl_coef = float(kl_coef)
        self.clip_range = float(clip_range)
        self.group_size = group_size
        self.num_train_epochs_per_step = max(1, int(num_train_epochs_per_step))

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_training(self.model)
        except Exception:
            self.model.train()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = self._recover_trainable_params()
        if not trainable_params:
            raise RuntimeError("MultiAgentGRPOTrainer found no trainable parameters")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

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

    def _tokenize_batch(
        self,
        prompts: list[Any],
        completions: list[str],
    ) -> tuple[dict[str, Any], Any]:
        """Tokenize prompts+completions, build labels with prompt masked out.

        Returns (encoded_full, shifted_labels) where shifted_labels has -100
        at every position that is padding or prompt (completion-only mask).
        """
        torch = self._torch

        rendered_prompts = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        full_texts = [
            ptext + comp
            for ptext, comp in zip(rendered_prompts, completions, strict=False)
        ]

        encoded_full = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded_prompt = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        device = next(self.model.parameters()).device
        encoded_full = {k: v.to(device) for k, v in encoded_full.items()}
        prompt_lengths = encoded_prompt["attention_mask"].sum(dim=1).tolist()

        labels = encoded_full["input_ids"].clone()
        labels[encoded_full["attention_mask"] == 0] = -100
        for row_idx, plen in enumerate(prompt_lengths):
            labels[row_idx, : int(plen)] = -100

        # Shift to align logits(t) -> label(t+1)
        shifted_labels = labels[:, 1:]
        return encoded_full, shifted_labels

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
        outputs = self.model(**encoded_full)
        logits = outputs.logits[:, :-1, :]  # (S, L-1, V)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        gather_labels = shifted_labels.masked_fill(shifted_labels == -100, 0)
        token_log_probs = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
        # Zero out invalid positions but don't multiply — keep raw logprobs
        token_log_probs = token_log_probs.masked_fill(shifted_labels == -100, 0.0)
        return token_log_probs

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
        prompt_groups = grouped_inputs["prompts"]
        completion_groups = grouped_inputs["completions"]
        raw_reward_groups = grouped_inputs["raw_rewards"]
        group_keys = sorted(grouped_raw_rewards) if grouped_raw_rewards else []
        # Use the ordering from the grouped_inputs (already sorted by key)
        for idx, (prompts, completions, rewards) in enumerate(
            zip(prompt_groups, completion_groups, raw_reward_groups, strict=False)
        ):
            group_key = f"group_{idx}"
            all_prompts.extend(prompts)
            all_completions.extend(completions)
            grouped_raw_rewards[group_key] = rewards
            total_samples += len(prompts)

        if total_samples == 0:
            raise RuntimeError("MultiAgentGRPOTrainer received no rollout samples")

        # --- Tokenize ONCE -------------------------------------------------
        encoded_full, shifted_labels = self._tokenize_batch(all_prompts, all_completions)
        completion_mask = (shifted_labels != -100).float()  # (S, L-1)

        # --- 1. Old log-probs: frozen, captured ONCE -----------------------
        with torch.no_grad():
            self.model.eval()
            old_lp = self._masked_token_logprobs(encoded_full, shifted_labels)  # (S, L-1)
            self.model.train()

        # --- 2. Ref log-probs: LoRA adapter disabled -----------------------
        with torch.no_grad():
            self.model.eval()
            try:
                cm = self.model.disable_adapter()
            except AttributeError:
                # Model doesn't support disable_adapter (e.g. non-PEFT);
                # fall back to old_lp (KL will be zero, which is harmless).
                cm = None

            if cm is not None:
                with cm:
                    ref_lp = self._masked_token_logprobs(encoded_full, shifted_labels)
            else:
                warnings.warn(
                    "Model does not expose disable_adapter(); "
                    "reference log-probs will equal old log-probs (KL = 0).",
                    stacklevel=2,
                )
                ref_lp = old_lp.clone()
            self.model.train()

        # --- 3. Advantages: computed once, detached -------------------------
        advantages = self._compute_group_advantages(
            grouped_raw_rewards, total_samples
        )  # (S,)

        # --- 4. Inner PPO epoch loop ----------------------------------------
        for _epoch in range(self.num_train_epochs_per_step):
            new_lp = self._masked_token_logprobs(encoded_full, shifted_labels)  # (S, L-1), WITH grad

            # FP16-safe log-prob ratio via log-space delta
            delta = (new_lp - old_lp).clamp(-5.0, 5.0)  # (S, L-1)
            ratio = delta.exp()  # (S, L-1)

            # Broadcast advantage to token dimension: (S,) -> (S, L-1)
            A_tok = advantages.unsqueeze(-1) * torch.ones_like(ratio)

            # Clipped surrogate
            surr1 = ratio * A_tok
            surr2 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * A_tok
            policy_loss = (
                -(torch.min(surr1, surr2) * completion_mask).sum()
                / completion_mask.sum().clamp_min(1.0)
            )

            # Schulman k3 KL estimator: k3 = exp(ref - new) - (ref - new) - 1
            ref_delta = ref_lp - new_lp
            kl_per_tok = ref_delta.exp() - ref_delta - 1.0  # >= 0 by construction
            kl_loss = (
                (kl_per_tok * completion_mask).sum()
                / completion_mask.sum().clamp_min(1.0)
            )

            loss = policy_loss + self.kl_coef * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # --- 5. Diagnostics -------------------------------------------------
        mask_sum = completion_mask.sum().clamp_min(1.0).item()
        masked_ratio = ratio.detach() * completion_mask
        ratio_mean = masked_ratio.sum().item() / mask_sum
        # Masked std: E[r^2] - E[r]^2  under the mask
        ratio_sq_mean = (masked_ratio ** 2).sum().item() / mask_sum
        ratio_std = (max(ratio_sq_mean - ratio_mean ** 2, 0.0)) ** 0.5
        # Clip fraction: fraction of valid tokens where ratio is outside clip range
        outside_clip = (
            ((ratio.detach() < 1.0 - self.clip_range)
             | (ratio.detach() > 1.0 + self.clip_range))
            * completion_mask
        )
        clip_fraction = outside_clip.sum().item() / mask_sum
        kl_max = (kl_per_tok.detach() * completion_mask).max().item()
        mask_coverage = completion_mask.mean().item()
        mean_advantage = advantages.mean().item()
        advantage_std = advantages.std().item() if advantages.numel() > 1 else 0.0

        return {
            "loss": loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "clip_fraction": clip_fraction,
            "kl_max": kl_max,
            "mask_coverage": mask_coverage,
            "mean_advantage": mean_advantage,
            "advantage_std": advantage_std,
        }


def _build_grpo_trainer(GRPOTrainer: Any, policy: Any, config: TrainingConfig) -> Any:
    """Instantiate GRPOTrainer across permissive stub and real implementations."""
    trainer_model = _extract_trainer_model(policy)
    tokenizer = _extract_trainer_tokenizer(policy)
    trainer_kwargs = {
        "model": trainer_model,
        "tokenizer": tokenizer,
        "learning_rate": config.grpo.learning_rate,
        "kl_coef": config.grpo.kl_coef,
        "clip_range": config.grpo.clip_range,
        "num_train_epochs_per_step": config.grpo.num_train_epochs_per_step,
        "group_size": config.grpo.group_size,
    }
    candidate_kwargs = [
        trainer_kwargs,
        {k: v for k, v in trainer_kwargs.items() if v is not None},
        {"model": trainer_model},
        {},
    ]
    for kwargs in candidate_kwargs:
        try:
            return GRPOTrainer(**kwargs)
        except TypeError:
            continue
    try:
        return GRPOTrainer(trainer_model)
    except TypeError as exc:
        message = str(exc)
        if "reward_funcs" in message:
            print(
                "Falling back to local multi-agent GRPO trainer because the "
                "installed TRL GRPOTrainer expects the newer reward_funcs-based API."
            )
            return MultiAgentGRPOTrainer(
                model=trainer_model,
                tokenizer=tokenizer,
                learning_rate=config.grpo.learning_rate,
                kl_coef=config.grpo.kl_coef,
                clip_range=config.grpo.clip_range,
                group_size=config.grpo.group_size,
                num_train_epochs_per_step=config.grpo.num_train_epochs_per_step,
            )
        raise RuntimeError(f"Unable to initialize GRPOTrainer: {exc}") from exc


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
                {"prompts": [], "completions": [], "raw_rewards": [], "normalized_rewards": [], "samples": []},
            )
            bucket["prompts"].append(sample.prompt)
            bucket["completions"].append(sample.completion_text)
            bucket["raw_rewards"].append(sample.raw_reward)
            bucket["normalized_rewards"].append(sample.normalized_reward)
            bucket["samples"].append(sample)

    ordered_keys = sorted(grouped)
    return {
        "prompts": [grouped[key]["prompts"] for key in ordered_keys],
        "completions": [grouped[key]["completions"] for key in ordered_keys],
        "raw_rewards": [grouped[key]["raw_rewards"] for key in ordered_keys],
        "normalized_rewards": [grouped[key]["normalized_rewards"] for key in ordered_keys],
        "samples": [grouped[key]["samples"] for key in ordered_keys],
    }


def _save_adapter_weights(policy: Any, target_dir: Path) -> None:
    model = getattr(policy, "_model", None)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        target_dir.mkdir(parents=True, exist_ok=True)
        save_pretrained(str(target_dir))


def _maybe_init_wandb(config: TrainingConfig) -> Any | None:
    """Initialize wandb if WANDB_API_KEY is set and wandb is importable.

    Returns the wandb run handle, or None if either precondition is unmet.
    Silent no-op on missing env var or missing package — this is how the HF
    backend path stays green when wandb isn't installed locally.
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
    return wandb.init(
        project=project,
        name=run_name,
        config=config.model_dump(mode="json"),
        settings=wandb.Settings(start_method="thread"),
    )


def _build_policy(
    config: TrainingConfig,
    bundle: Any | None,
    *,
    LoraConfig: Any,
) -> Any:
    lora_adapter_path: str | None = None
    if bundle is not None and bundle.lora_weights_path.exists():
        lora_adapter_path = str(bundle.lora_weights_path)

    backend = config.backend
    if backend == "unsloth":
        from training.policy_adapter import unsloth_policy_factory

        return unsloth_policy_factory(
            config.model.base,
            lora_adapter_path=lora_adapter_path,
            lora_r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_target_modules=list(config.lora.target_modules),
            max_seq_length=config.unsloth_max_seq_length,
            load_in_4bit=config.load_in_4bit,
            use_vllm=config.rollout.use_vllm,
            max_new_tokens=config.model.max_completion_tokens,
            temperature=0.0,
            seed=config.seed.training_rng,
        )
    if backend == "hf":
        from training.policy_adapter import hf_policy_factory

        peft_config = None
        if lora_adapter_path is None:
            peft_config = LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                target_modules=list(config.lora.target_modules),
                lora_dropout=config.lora.dropout,
            )

        return hf_policy_factory(
            config.model.base,
            lora_adapter_path=lora_adapter_path,
            peft_config=peft_config,
            torch_dtype=config.model.dtype,
            max_new_tokens=config.model.max_completion_tokens,
            do_sample=False,
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
            )
        )
    return eval_results


def run_training(config_path: Path = Path("training/config.yaml")) -> None:
    """Run the Phase 7 training loop."""
    raw = _load_yaml_config(config_path)
    config = TrainingConfig(**raw)

    if config.backend == "unsloth":
        try:
            # Import Unsloth before TRL / transformers / PEFT so its monkey
            # patches apply to the training stack as intended.
            import unsloth  # type: ignore  # noqa: F401
        except ImportError:
            pass

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
        CheckpointBundle,
        load_checkpoint,
        rotate_checkpoints,
        save_checkpoint,
    )
    from training.reward import RewardNormalizer
    from training.rollout import collect_batch

    ckpt_root = Path(config.checkpoint.root_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config.metrics.csv_path)
    jsonl_dir = Path(config.metrics.jsonl_dir)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(config)

    rng = random.Random(config.seed.training_rng)
    bundle = load_checkpoint(ckpt_root)
    start_step = 0
    wall_total = 0.0
    if bundle is not None:
        start_step = bundle.step + 1
        wall_total = bundle.wall_seconds_total
        rng.setstate(pickle.loads(bundle.rollout_rng_state))

    curriculum = CurriculumController()
    normalizer = RewardNormalizer()
    if bundle is not None:
        curriculum.load_snapshot(bundle.curriculum_snapshot)
        normalizer.load_snapshot(bundle.normalizer_snapshot)

    env = EvacEnvironment()
    policy = _build_policy(config, bundle, LoraConfig=LoraConfig)
    trainer = _build_grpo_trainer(GRPOTrainer, policy, config)
    model_name = config.model.base
    config_hash = _config_hash(config)
    disaster_families = [DisasterType(item) for item in config.rollout.disaster_families]

    stop_requested = False

    def _signal_handler(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _signal_handler)

    def _seed_gen() -> int:
        return rng.randint(0, 2_147_483_647)

    def _write_checkpoint(step: int) -> None:
        adapter_path = ckpt_root / f"ckpt_{step}" / "lora_adapter"
        new_bundle = CheckpointBundle(
            step=step,
            wall_seconds_total=wall_total,
            curriculum_snapshot=curriculum.snapshot(),
            normalizer_snapshot=normalizer.snapshot(),
            rollout_rng_state=pickle.dumps(rng.getstate()),
            lora_weights_path=adapter_path,
            model_name=model_name,
            config_hash=config_hash,
        )
        saved_dir = save_checkpoint(ckpt_root, new_bundle)
        _save_adapter_weights(policy, saved_dir / "lora_adapter")
        _save_adapter_weights(policy, ckpt_root / "latest" / "lora_adapter")
        rotate_checkpoints(ckpt_root, config.checkpoint.keep_last_n)

    try:
        step_cap = config.max_steps if config.max_steps is not None else 100_000
        for step in range(start_step, step_cap):
            if stop_requested:
                break

            step_started = time.monotonic()
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
            )

            grouped_inputs = _group_for_grpo(results)
            _call_trainer_step(trainer, grouped_inputs)

            step_wall = time.monotonic() - step_started
            wall_total += step_wall

            all_raw_orch: list[float] = []
            all_raw_floor: list[float] = []
            all_norm_orch: list[float] = []
            all_norm_floor: list[float] = []
            invalid_count = 0
            total_samples = 0
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
                    if sample.parsed_action.get("fallback_reason") == "parse_error":
                        invalid_count += 1

            metrics_row = {
                "step": step,
                "wall_seconds": round(wall_total, 2),
                "tier_mix": ";".join(sorted({result.tier for result in results})),
                "mean_raw_reward_orch": round(sum(all_raw_orch) / max(len(all_raw_orch), 1), 4),
                "mean_raw_reward_floor": round(sum(all_raw_floor) / max(len(all_raw_floor), 1), 4),
                "mean_norm_reward_orch": round(sum(all_norm_orch) / max(len(all_norm_orch), 1), 4),
                "mean_norm_reward_floor": round(sum(all_norm_floor) / max(len(all_norm_floor), 1), 4),
                "invalid_action_rate": round(invalid_count / max(total_samples, 1), 4),
                "override_rate": 0.0,
                "override_win_rate": 0.0,
                "rationale_bonus_mean": 0.0,
                "episodes_seen": (step + 1) * config.rollout.episodes_per_step,
            }
            append_training_metrics_row(metrics_path, metrics_row)
            if wandb_run is not None:
                wandb_run.log(metrics_row, step=step)

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
        if stop_requested or bundle is not None:
            final_step = max(start_step, 0)
            _write_checkpoint(final_step if final_step > 0 else 0)
        if wandb_run is not None:
            wandb_run.finish()
