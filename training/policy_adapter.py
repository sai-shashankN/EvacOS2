"""Policy adapters: stub policy, completion parser, HF + Unsloth policy factories.

The `Policy` protocol requires only `.act(prompt, agent_id, role)`.
Policies MAY additionally expose an optional batched fast path::

    def act_batch(
        self,
        prompts: list[list[dict[str, str]]],
        agent_ids: list[str],
        roles: list[str],
    ) -> list[PolicyResult]: ...

Consumers detect the optional method via ``hasattr(policy, "act_batch")`` — no
protocol break, no new abstract class. The Phase 7 Unsloth backend uses this
fast path to collapse 6 per-round generate calls into 1 vLLM / HF batched call.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from evacos_ma.permissions import validate_action_for_role
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentRole,
)
from training.compat import patch_transformers_cache_exports

PolicyResult = tuple[str, list[int]]

logger = logging.getLogger(__name__)

_PROMPT_TRUNCATION_WARNED = False


def _as_policy_result(result: PolicyResult | str | object) -> PolicyResult:
    if isinstance(result, tuple) and len(result) == 2:
        text, token_ids = result
        text_str = text if isinstance(text, str) else str(text)
        if isinstance(token_ids, list):
            return text_str, [int(token_id) for token_id in token_ids]
        return text_str, []
    if isinstance(result, str):
        return result, []
    return str(result), []


def _call_tokenizer(tokenizer: Any, rendered: Any, **kwargs: Any) -> Any:
    try:
        return tokenizer(rendered, **kwargs)
    except TypeError:
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"max_length", "truncation", "padding", "add_special_tokens"}
        }
        return tokenizer(rendered, **filtered_kwargs)


def _sequence_lengths(input_ids: Any) -> list[int]:
    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, list):
        if not input_ids:
            return []
        if isinstance(input_ids[0], list):
            return [len(row) for row in input_ids]
        return [len(input_ids)]
    return []


def _warn_once_if_prompt_truncated(
    *,
    tokenizer: Any,
    rendered: list[str],
    max_prompt_tokens: int,
) -> None:
    global _PROMPT_TRUNCATION_WARNED
    if _PROMPT_TRUNCATION_WARNED:
        return

    raw_encoded = _call_tokenizer(
        tokenizer,
        rendered,
        add_special_tokens=True,
        truncation=False,
    )
    raw_lengths = _sequence_lengths(raw_encoded.get("input_ids") if isinstance(raw_encoded, dict) else None)
    if any(length > max_prompt_tokens for length in raw_lengths):
        logger.warning(
            "Prompt tokenization exceeded model.max_prompt_tokens=%s; truncating from the left.",
            max_prompt_tokens,
        )
        _PROMPT_TRUNCATION_WARNED = True


@runtime_checkable
class Policy(Protocol):
    def act(
        self,
        prompt: list[dict[str, str]],
        agent_id: str,
        role: str,
    ) -> PolicyResult | str:
        ...


class RoleRoutedPolicy:
    """Route generation to per-role policies while preserving batch order.

    When the same policy instance backs both roles, callers should keep using
    that shared instance directly so the rollout fast path stays at one
    generate call per round. This wrapper exists for true split-role setups,
    where the orchestrator and floor agents are intentionally backed by
    different trainable models.
    """

    def __init__(self, *, orchestrator_policy: Policy, floor_policy: Policy) -> None:
        self._role_policies = {
            "orchestrator": orchestrator_policy,
            "floor_agent": floor_policy,
        }

    def policy_for_role(self, role: str) -> Policy:
        if role not in self._role_policies:
            raise ValueError(f"Unknown role {role!r}")
        return self._role_policies[role]

    def act(
        self,
        prompt: list[dict[str, str]],
        agent_id: str,
        role: str,
    ) -> PolicyResult | str:
        policy = self.policy_for_role(role)
        return policy.act(prompt, agent_id, role)

    def act_batch(
        self,
        prompts: list[list[dict[str, str]]],
        agent_ids: list[str],
        roles: list[str],
    ) -> list[PolicyResult]:
        if not (len(prompts) == len(agent_ids) == len(roles)):
            raise ValueError("prompts, agent_ids, and roles must have the same length")
        if not prompts:
            return []

        grouped_indices: dict[str, list[int]] = {"orchestrator": [], "floor_agent": []}
        for idx, role in enumerate(roles):
            grouped_indices.setdefault(role, []).append(idx)

        outputs: list[PolicyResult | None] = [None] * len(prompts)
        for role, indices in grouped_indices.items():
            if not indices:
                continue
            policy = self.policy_for_role(role)
            role_prompts = [prompts[idx] for idx in indices]
            role_agent_ids = [agent_ids[idx] for idx in indices]
            role_roles = [roles[idx] for idx in indices]
            if hasattr(policy, "act_batch"):
                role_outputs = policy.act_batch(role_prompts, role_agent_ids, role_roles)  # type: ignore[attr-defined]
            else:
                role_outputs = [
                    _as_policy_result(policy.act(prompt, aid, role_name))
                    for prompt, aid, role_name in zip(role_prompts, role_agent_ids, role_roles, strict=True)
                ]
            for out_idx, original_idx in enumerate(indices):
                outputs[original_idx] = role_outputs[out_idx]

        return [output if output is not None else ("", []) for output in outputs]


@dataclass
class StubPolicy:
    """Deterministic pure-Python stub policy used by tests."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, prompt: list[dict[str, str]], agent_id: str, role: str) -> PolicyResult:
        system_msg = next((msg for msg in prompt if msg["role"] == "system"), {})
        user_msg = next((msg for msg in prompt if msg["role"] == "user"), {})
        system_text = system_msg.get("content", "")
        user_text = user_msg.get("content", "")

        episode_id = self._extract_field(system_text, "episode_id") or "ep_unknown"
        round_id = self._extract_int(system_text, "round_id") or 0

        if role == "orchestrator":
            return self._orchestrator_action(episode_id, round_id, user_text), []
        return self._floor_action(episode_id, round_id, agent_id, user_text), []

    def _orchestrator_action(self, episode_id: str, round_id: int, user_text: str) -> str:
        has_escalation = "escalation" in user_text.lower() and "urgency" in user_text.lower()
        if has_escalation:
            action_type = ActionTypeMA.broadcast_directive.value
            arguments = {
                "directive": {
                    "directive_id": f"dir_{self._rng.randint(1000, 9999)}",
                    "target": "all",
                    "directive_type": "evacuation_priority",
                    "params": {"priority": "high"},
                    "priority": "high",
                    "issued_round": round_id,
                }
            }
        else:
            action_type = ActionTypeMA.wait.value
            arguments = {}

        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": "orchestrator",
                "action_id": f"act_{self._rng.randint(10000, 99999)}",
                "action_type": action_type,
                "arguments": arguments,
                "rationale": "stub policy",
            }
        )

    def _floor_action(
        self,
        episode_id: str,
        round_id: int,
        agent_id: str,
        user_text: str,
    ) -> str:
        has_exits = '"exit_id"' in user_text and '"blocked": false' in user_text
        if has_exits:
            action_type = ActionTypeMA.route_within_floor.value
            floor_num = agent_id.split("_")[1]
            arguments = {
                "from_room_id": f"F{floor_num}_R0",
                "to_room_id": f"F{floor_num}_R1",
            }
        else:
            action_type = ActionTypeMA.wait.value
            arguments = {}

        return json.dumps(
            {
                "episode_id": episode_id,
                "round_id": round_id,
                "agent_id": agent_id,
                "action_id": f"act_{self._rng.randint(10000, 99999)}",
                "action_type": action_type,
                "arguments": arguments,
                "rationale": "stub policy",
            }
        )

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str | None:
        import re

        match = re.search(rf'"{field_name}"\s*:\s*"([^"]*)"', text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_int(text: str, field_name: str) -> int | None:
        import re

        if field_name == "round_id":
            match = re.search(r"Round:\s*(\d+)", text)
            if match:
                return int(match.group(1))
        match = re.search(rf'"{field_name}"\s*:\s*(\d+)', text)
        return int(match.group(1)) if match else None


def _should_enforce_eager_for_vllm() -> bool:
    """Return True when the current GPU needs CUDA-graph capture disabled.

    vLLM's default CUDA graph mode (FULL_AND_PIECEWISE) is incompatible with
    the FlexAttention backend that Tesla T4 / V100 fall back to (Flash
    Attention 2 requires compute capability >= 8.0). Forcing eager mode
    avoids the ``CUDAGraphMode.FULL_AND_PIECEWISE is not supported`` crash
    on pre-Ampere GPUs.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major < 8


def _vllm_kwargs_for_current_gpu() -> dict[str, Any]:
    """Return the vLLM kwargs needed for safe init on the current GPU.

    On pre-Ampere GPUs (compute capability < 8.0, e.g. T4 / V100):
    - ``enforce_eager=True`` disables vLLM's own CUDA graph capture,
      avoiding the ``CUDAGraphMode.FULL_AND_PIECEWISE is not supported``
      crash with the FlexAttention fallback backend.
    - ``compilation_config={"level": 0}`` disables torch.compile / inductor
      entirely, avoiding a second latent crash in inductor's
      ``cudagraph_trees`` allocator-pool-checkpoint machinery that fires
      when switching between rollout (vLLM) and training (compat GRPO)
      modes on shared weights.

    Returns ``{}`` on Ampere+ GPUs (the default vLLM fast path is safe).
    """
    if not _should_enforce_eager_for_vllm():
        return {}
    return {"enforce_eager": True, "compilation_config": {"level": 0}}


def parse_completion_to_action(
    completion_text: str,
    agent_id: str,
    role: str,
    expected_episode_id: str,
    expected_round_id: int,
) -> tuple[ActionEnvelopeMA | None, str]:
    text = completion_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json"

    try:
        action = ActionEnvelopeMA.model_validate(payload)
    except Exception:
        return None, "schema_invalid"

    action = action.model_copy(
        update={
            "episode_id": expected_episode_id,
            "agent_id": agent_id,
            "round_id": expected_round_id,
        }
    )

    expected_role = AgentRole.orchestrator if role == "orchestrator" else AgentRole.floor_agent
    validation = validate_action_for_role(action, expected_role)
    if not validation.valid:
        return None, validation.reason

    return action, "ok"


def hf_policy_factory(
    model_name: str,
    *,
    lora_adapter_path: str | None = None,
    max_prompt_tokens: int = 3500,
    **gen_kwargs: Any,
) -> Policy:
    """Build an import-guarded transformers/peft-backed policy."""

    patch_transformers_cache_exports()

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch/transformers/trl/peft not installed"
        ) from exc

    try:
        from peft import PeftModel, get_peft_model  # noqa: F401

        has_peft = True
    except ImportError as exc:
        if lora_adapter_path is not None or gen_kwargs.get("peft_config") is not None:
            raise RuntimeError(
                "torch/transformers/trl/peft not installed"
            ) from exc
        has_peft = False
        PeftModel = None  # type: ignore[assignment]
        get_peft_model = None  # type: ignore[assignment]

    class _HFPolicy:
        def __init__(self) -> None:
            torch_dtype = gen_kwargs.get("torch_dtype", getattr(torch, "bfloat16", torch.float32))
            if isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype, getattr(torch, "float32", None))

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._tokenizer.padding_side = "left"
            self._tokenizer.truncation_side = "left"
            if getattr(self._tokenizer, "pad_token", None) is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._max_prompt_tokens = max_prompt_tokens
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

            if lora_adapter_path is not None:
                if not has_peft or PeftModel is None:
                    raise RuntimeError("torch/transformers/trl/peft not installed")
                self._model = PeftModel.from_pretrained(self._model, lora_adapter_path)
            elif gen_kwargs.get("peft_config") is not None:
                if not has_peft or get_peft_model is None:
                    raise RuntimeError("torch/transformers/trl/peft not installed")
                self._model = get_peft_model(self._model, gen_kwargs["peft_config"])

            self._model.eval()
            self._gen_kwargs = gen_kwargs

        def act(
            self,
            prompt: list[dict[str, str]],
            agent_id: str,
            role: str,
        ) -> PolicyResult:
            del agent_id, role
            rendered = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            _warn_once_if_prompt_truncated(
                tokenizer=self._tokenizer,
                rendered=[rendered],
                max_prompt_tokens=self._max_prompt_tokens,
            )
            inputs = _call_tokenizer(
                self._tokenizer,
                rendered,
                return_tensors="pt",
                max_length=self._max_prompt_tokens,
                truncation=True,
            ).to(self._model.device)
            # Fix H29: ensure dropout is off during generation
            was_training = self._model.training
            self._model.eval()
            try:
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self._gen_kwargs.get("max_new_tokens", 256),
                        do_sample=self._gen_kwargs.get("do_sample", False),
                        temperature=self._gen_kwargs.get("temperature", 1.0),
                    )
            finally:
                if was_training:
                    self._model.train()
            first_row = outputs[0] if hasattr(outputs, "__getitem__") else next(iter(outputs))
            generated = first_row[inputs["input_ids"].shape[-1] :]
            generated_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
            return self._tokenizer.decode(generated, skip_special_tokens=True), generated_ids

    return _HFPolicy()


_DEFAULT_UNSLOTH_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def unsloth_policy_factory(
    base_model: str,
    *,
    lora_adapter_path: str | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: Sequence[str] | None = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    dtype: str | None = None,
    max_prompt_tokens: int = 3500,
    use_vllm: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 0,
) -> "UnslothPolicy":
    """Build an `UnslothPolicy` backed by Unsloth-quantized Qwen (default 1.5B).

    - Cold start: loads the base model and wraps it with a new PEFT LoRA
      adapter (`r=lora_r`, `alpha=lora_alpha`).
    - Resume: loads the base model and attaches an existing LoRA adapter from
      ``lora_adapter_path`` (directory layout is PEFT-compatible with
      ``hf_policy_factory``, so Phase 7 checkpoints resume transparently).
    - ``use_vllm=True`` enables Unsloth's shared-weight vLLM path; `.act_batch`
      will then route through vLLM's ``fast_generate``.

    Raises:
        RuntimeError: if ``unsloth`` (or, when ``use_vllm``, ``vllm``) is not
        installed. The message contains the string ``unsloth`` so callers can
        detect it. Unsloth does not support Windows — local Windows dev should
        use ``hf_policy_factory`` instead.
    """

    patch_transformers_cache_exports()

    try:
        from unsloth import FastLanguageModel  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "unsloth is not installed; unsloth_policy_factory requires the "
            "optional Colab-only dependency. Install via "
            "`pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"` "
            "inside a Colab/Linux+CUDA environment. Windows local dev should "
            "use hf_policy_factory instead."
        ) from exc

    if use_vllm:
        try:
            import vllm  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed; unsloth_policy_factory(use_vllm=True) "
                "requires vllm. Install via `pip install vllm` in a CUDA "
                "environment, or set rollout.use_vllm=false."
            ) from exc

    target_modules = (
        tuple(lora_target_modules)
        if lora_target_modules is not None
        else _DEFAULT_UNSLOTH_TARGET_MODULES
    )

    return UnslothPolicy(
        base_model=base_model,
        lora_adapter_path=lora_adapter_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_modules=target_modules,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        max_prompt_tokens=max_prompt_tokens,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


class UnslothPolicy:
    """Policy wrapping an Unsloth-loaded Qwen model, with optional vLLM rollout.

    Exposes:
        - ``.act(prompt, agent_id, role) -> PolicyResult`` (Policy protocol)
        - ``.act_batch(prompts, agent_ids, roles) -> list[PolicyResult]`` (fast path)

    Both paths apply the tokenizer's chat template. The ``act_batch`` fast path
    is the high-impact change for Colab wall-clock: one generate call per
    rollout round instead of six (1 orchestrator + 5 floor agents).
    """

    def __init__(
        self,
        *,
        base_model: str,
        lora_adapter_path: str | None,
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: Sequence[str],
        max_seq_length: int,
        load_in_4bit: bool,
        dtype: str | None,
        max_prompt_tokens: int,
        use_vllm: bool,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        from unsloth import FastLanguageModel  # type: ignore  # local re-import
        import torch

        self._base_model = base_model
        self._use_vllm = use_vllm
        self._max_new_tokens = max_new_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._temperature = temperature
        self._seed = seed

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            None: None,
        }
        resolved_dtype = dtype_map.get(dtype) if dtype in dtype_map else None
        if dtype is not None and resolved_dtype is None:
            raise ValueError(
                f"UnslothPolicy: unrecognized dtype={dtype!r}; expected one of "
                f"{sorted(k for k in dtype_map if k is not None)} or None"
            )

        from_pretrained_kwargs: dict[str, Any] = (
            _vllm_kwargs_for_current_gpu() if use_vllm else {}
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=use_vllm,
            dtype=resolved_dtype,
            **from_pretrained_kwargs,
        )

        if lora_adapter_path is not None:
            # Resume path: attach existing PEFT-format LoRA adapter.
            load_adapter = getattr(model, "load_adapter", None)
            if load_adapter is None:
                raise RuntimeError(
                    "Unsloth model does not expose load_adapter; cannot resume "
                    f"LoRA adapter from {lora_adapter_path!r}. Check unsloth version."
                )
            load_adapter(lora_adapter_path)
        else:
            # Cold start: wrap with a new PEFT LoRA.
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=seed,
            )

        # Prepare for inference (does not block subsequent training — Unsloth
        # kernels switch modes transparently via `for_training`/`for_inference`).
        FastLanguageModel.for_inference(model)

        self._model = model
        self._tokenizer = tokenizer
        self._tokenizer.padding_side = "left"
        self._tokenizer.truncation_side = "left"
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    # ------------------------------------------------------------------ Policy

    def act(
        self,
        prompt: list[dict[str, str]],
        agent_id: str,
        role: str,
    ) -> PolicyResult:
        return self.act_batch([prompt], [agent_id], [role])[0]

    def act_batch(
        self,
        prompts: list[list[dict[str, str]]],
        agent_ids: list[str],
        roles: list[str],
    ) -> list[PolicyResult]:
        del agent_ids, roles  # informational only — no role-gated sampling

        if not prompts:
            return []

        rendered = [
            self._tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

        if self._use_vllm:
            return self._vllm_generate(rendered)
        return self._hf_generate(rendered)

    # ------------------------------------------------------------ Generation

    def _vllm_generate(self, rendered: list[str]) -> list[PolicyResult]:
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "vllm is not installed; UnslothPolicy(use_vllm=True) cannot "
                "generate. Install vllm or set rollout.use_vllm=false."
            ) from exc

        sampling = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_new_tokens,
            seed=self._seed,
        )
        fast_generate = getattr(self._model, "fast_generate", None)
        if fast_generate is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Unsloth model does not expose fast_generate; "
                "fast_inference=True was requested at load but the shared-weight "
                "vLLM path is unavailable. Upgrade unsloth or disable use_vllm."
            )
        outputs = fast_generate(rendered, sampling_params=sampling)

        texts: list[PolicyResult] = []
        for output in outputs:
            completions = getattr(output, "outputs", None) or []
            if not completions:
                texts.append(("", []))
                continue
            completion = completions[0]
            texts.append(
                (
                    getattr(completion, "text", "") or "",
                    list(getattr(completion, "token_ids", []) or []),
                )
            )
        return texts

    def _hf_generate(self, rendered: list[str]) -> list[PolicyResult]:
        import torch  # type: ignore

        tokenizer = self._tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        assert original_padding_side == "left", (
            f"UnslothPolicy._hf_generate requires left-padded tokenizer; "
            f"got {original_padding_side!r}"
        )
        max_prompt_tokens = getattr(self, "_max_prompt_tokens", 3500)

        _warn_once_if_prompt_truncated(
            tokenizer=tokenizer,
            rendered=rendered,
            max_prompt_tokens=max_prompt_tokens,
        )
        encoded = _call_tokenizer(
            tokenizer,
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(self._model.device)

        # Fix H29: ensure dropout is off during generation.
        # Use for_inference when Unsloth is available for optimal kernels,
        # otherwise fall back to .eval().
        was_training = self._model.training
        _used_unsloth_inference = False
        try:
            from unsloth import FastLanguageModel  # type: ignore

            FastLanguageModel.for_inference(self._model)
            _used_unsloth_inference = True
        except Exception:
            self._model.eval()

        try:
            do_sample = self._temperature > 0.0
            generate_kwargs = {
                **encoded,
                "max_new_tokens": self._max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = self._temperature
            with torch.no_grad():
                outputs = self._model.generate(**generate_kwargs)
        finally:
            if _used_unsloth_inference:
                if was_training:
                    try:
                        from unsloth import FastLanguageModel  # type: ignore

                        FastLanguageModel.for_training(self._model)
                    except Exception:
                        self._model.train()
                else:
                    self._model.eval()
            elif was_training:
                self._model.train()
            else:
                self._model.eval()

        results: list[PolicyResult] = []
        padded_prompt_width = encoded["input_ids"].shape[-1]
        pad_id = tokenizer.pad_token_id
        for row in outputs:
            generated = row[padded_prompt_width:]
            gen_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
            while gen_ids and pad_id is not None and gen_ids[-1] == pad_id:
                gen_ids.pop()
            results.append((tokenizer.decode(generated, skip_special_tokens=True), gen_ids))
        return results
