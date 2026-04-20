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


class _CompatGRPOTrainer:
    """Minimal local fallback when TRL's GRPOTrainer API is incompatible.

    This trainer uses group-centered rewards as advantages and optimizes
    completion-token log-probabilities directly. It is intentionally simple:
    enough to keep Colab training unblocked when the installed TRL release no
    longer supports the older step-wise constructor expected by this repo.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        learning_rate: float,
        group_size: int,
        num_train_epochs_per_step: int,
    ) -> None:
        import torch

        if tokenizer is None:
            raise RuntimeError("Compat GRPO trainer requires a tokenizer")

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
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
            raise RuntimeError("Compat GRPO trainer found no trainable parameters")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    def step(
        self,
        grouped_prompts: list[list[Any]],
        grouped_completions: list[list[str]],
        grouped_rewards: list[list[float]],
    ) -> dict[str, float]:
        torch = self._torch
        losses: list[float] = []

        for _ in range(self.num_train_epochs_per_step):
            self.optimizer.zero_grad()
            total_loss = None

            for prompts, completions, rewards in zip(
                grouped_prompts, grouped_completions, grouped_rewards, strict=False
            ):
                group_loss = self._compute_group_loss(prompts, completions, rewards)
                total_loss = group_loss if total_loss is None else total_loss + group_loss

            if total_loss is None:
                raise RuntimeError("Compat GRPO trainer received no rollout groups")

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(total_loss.detach().item()))

        return {"loss": sum(losses) / max(len(losses), 1)}

    def _compute_group_loss(
        self,
        prompts: list[Any],
        completions: list[str],
        rewards: list[float],
    ) -> Any:
        torch = self._torch

        rendered_prompts = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        full_texts = [prompt_text + completion for prompt_text, completion in zip(rendered_prompts, completions, strict=False)]

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
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100

        outputs = self.model(**encoded_full)
        logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        gather_labels = shifted_labels.masked_fill(shifted_labels == -100, 0)
        token_log_probs = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
        valid_mask = shifted_labels != -100
        token_log_probs = token_log_probs * valid_mask

        token_counts = valid_mask.sum(dim=1).clamp_min(1)
        seq_log_probs = token_log_probs.sum(dim=1) / token_counts

        reward_tensor = torch.tensor(rewards, device=device, dtype=seq_log_probs.dtype)
        advantages = reward_tensor - reward_tensor.mean()
        if reward_tensor.numel() > 1:
            advantages = advantages / reward_tensor.std(unbiased=False).clamp_min(1e-6)

        return -(advantages.detach() * seq_log_probs).mean()


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
                "Falling back to local compat GRPO trainer because the installed "
                "TRL GRPOTrainer expects the newer reward_funcs-based API."
            )
            return _CompatGRPOTrainer(
                model=trainer_model,
                tokenizer=tokenizer,
                learning_rate=config.grpo.learning_rate,
                group_size=config.grpo.group_size,
                num_train_epochs_per_step=config.grpo.num_train_epochs_per_step,
            )
        raise RuntimeError(f"Unable to initialize GRPOTrainer: {exc}") from exc


def _call_trainer_step(trainer: Any, grouped_inputs: dict[str, list[list[Any]]]) -> Any:
    """Call the trainer step using a permissive signature fallback."""
    step_fn = getattr(trainer, "step", None)
    if step_fn is None:
        raise RuntimeError("GRPOTrainer does not expose a step(...) method")

    prompt_groups = grouped_inputs["prompts"]
    completion_groups = grouped_inputs["completions"]
    reward_groups = grouped_inputs["rewards"]

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
                {"prompts": [], "completions": [], "rewards": []},
            )
            bucket["prompts"].append(sample.prompt)
            bucket["completions"].append(sample.completion_text)
            bucket["rewards"].append(sample.normalized_reward)

    ordered_keys = sorted(grouped)
    return {
        "prompts": [grouped[key]["prompts"] for key in ordered_keys],
        "completions": [grouped[key]["completions"] for key in ordered_keys],
        "rewards": [grouped[key]["rewards"] for key in ordered_keys],
    }


def _save_adapter_weights(policy: Any, target_dir: Path) -> None:
    model = getattr(policy, "_model", None)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        target_dir.mkdir(parents=True, exist_ok=True)
        save_pretrained(str(target_dir))


def _init_wandb(config: TrainingConfig) -> Any | None:
    """Initialize W&B only when explicitly enabled via environment."""
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
    wandb_run = _init_wandb(config)

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
        for step in range(start_step, 100_000):
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
