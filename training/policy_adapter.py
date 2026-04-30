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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from evacos_ma.permissions import validate_action_for_role
from evacos_ma.schemas.multi_agent import (
    ActionEnvelopeMA,
    ActionTypeMA,
    AgentRole,
    ACTION_TYPE_TO_ARGS,
)
from training.compat import patch_transformers_cache_exports
from training.scope_router import GENERALIST_POLICY_KEY, SPECIALIST_POLICY_KEYS, route_scope

PolicyResult = tuple[str, list[int]]

logger = logging.getLogger(__name__)

_PROMPT_TRUNCATION_WARNED = False
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ACTION_TYPE_VALUES = {item.value for item in ActionTypeMA}
_CHAT_ROLE_TOKEN_RE = re.compile(r"\n\s*(?:assistant|user|system)\s*\n", re.IGNORECASE)
_ACTION_TYPE_RE = re.compile(r'"action_type"\s*:\s*"(?P<action_type>[a-z_]+)"')
_ORDERED_FLOOR_IDS_RE = re.compile(r'"ordered_floor_ids"\s*:\s*\[(?P<body>[^\]]*)\]', re.DOTALL)
_FLOOR_ID_RE = re.compile(r'"?(floor_\d+)"?')


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


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object found in text, if any."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _json_payload_candidates(completion_text: str) -> list[str]:
    """Yield likely JSON payloads from a model completion, in priority order."""
    stripped = completion_text.strip()
    if not stripped:
        return []

    candidates: list[str] = []

    def _append(candidate: str | None) -> None:
        if candidate is None:
            return
        normalized = candidate.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _append(stripped)

    fenced_blocks = _JSON_FENCE_RE.findall(stripped)
    for block in fenced_blocks:
        _append(block)

    _append(_extract_first_json_object(stripped))
    for block in fenced_blocks:
        _append(_extract_first_json_object(block))

    return candidates


def _salvage_orchestrator_priority_payload(
    completion_text: str,
    *,
    expected_episode_id: str,
    expected_round_id: int,
    agent_id: str,
) -> dict[str, Any] | None:
    """Recover the narrow malformed priority-directive shape seen in traces.

    H200 7B runs occasionally leak chat-role tokens inside JSON strings. That
    breaks ``json.loads`` even when the intended action type and floor order are
    still visible. Salvage only this deterministic orchestrator action; the
    identity fields are overwritten and the floor ids are validated downstream.
    """

    if _CHAT_ROLE_TOKEN_RE.search(completion_text):
        logger.debug("Attempting priority JSON salvage after chat-role token leakage.")

    action_types = [match.group("action_type") for match in _ACTION_TYPE_RE.finditer(completion_text)]
    if ActionTypeMA.evacuate_floor_priority.value not in action_types:
        return None

    ordered_floor_match = _ORDERED_FLOOR_IDS_RE.search(completion_text)
    if ordered_floor_match is None:
        return None

    floor_ids: list[str] = []
    for match in _FLOOR_ID_RE.finditer(ordered_floor_match.group("body")):
        floor_id = match.group(1)
        if floor_id not in floor_ids:
            floor_ids.append(floor_id)
    if not floor_ids:
        return None

    return {
        "episode_id": expected_episode_id,
        "round_id": expected_round_id,
        "agent_id": agent_id,
        "action_id": f"{agent_id}_{expected_round_id}_priority_salvaged",
        "action_type": ActionTypeMA.evacuate_floor_priority.value,
        "arguments": {"ordered_floor_ids": floor_ids},
        "client_metadata": {
            "parser_salvaged": True,
            "salvage_reason": "malformed_priority_json",
        },
    }


def _normalize_action_payload(
    payload: Any,
    *,
    expected_episode_id: str,
    expected_round_id: int,
    agent_id: str,
    role: str,
) -> dict[str, Any] | None:
    """Coerce common model formatting mistakes into ActionEnvelopeMA shape."""
    if not isinstance(payload, dict):
        return None

    normalized = dict(payload)

    episode_id = normalized.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        normalized["episode_id"] = expected_episode_id

    round_id = normalized.get("round_id")
    if isinstance(round_id, str):
        digits = re.findall(r"\d+", round_id)
        if len(digits) == 1:
            normalized["round_id"] = int(digits[0])
        else:
            normalized["round_id"] = expected_round_id
    elif not isinstance(round_id, int):
        normalized["round_id"] = expected_round_id

    model_agent_id = normalized.get("agent_id")
    if not isinstance(model_agent_id, str) or not model_agent_id.strip():
        normalized["agent_id"] = agent_id

    action_id = normalized.get("action_id")
    if isinstance(action_id, str):
        if not action_id.strip():
            normalized["action_id"] = f"{agent_id}_{expected_round_id}"
    elif action_id is None:
        normalized["action_id"] = f"{agent_id}_{expected_round_id}"
    else:
        normalized["action_id"] = str(action_id)

    if normalized.get("arguments") is None or not isinstance(normalized.get("arguments"), dict):
        normalized["arguments"] = {}

    action_type = normalized.get("action_type")
    arguments = normalized.get("arguments")
    if (
        role == "floor_agent"
        and isinstance(arguments, dict)
        and (action_type in (None, "", "action_type") or action_type not in _ACTION_TYPE_VALUES)
    ):
        action_id_hint = str(normalized.get("action_id", "")).lower()
        has_route_target = any(arguments.get(key) for key in ("to_room_id", "exit_id", "stairwell_id"))
        if "route" in action_id_hint or has_route_target:
            normalized["action_type"] = ActionTypeMA.route_within_floor.value
            action_type = normalized["action_type"]

    if action_type == ActionTypeMA.evacuate_floor_priority.value and isinstance(arguments, dict):
        nested_arguments = arguments.get("evacuate_floor_priority_arguments")
        if "ordered_floor_ids" not in arguments and isinstance(nested_arguments, dict):
            logger.warning(
                "Unwrapped nested evacuate_floor_priority_arguments; prompt should emit flat ordered_floor_ids."
            )
            arguments = dict(nested_arguments)
            normalized["arguments"] = arguments
        priority_floor = arguments.get("priority_floor")
        if "ordered_floor_ids" not in arguments and priority_floor:
            if isinstance(priority_floor, str):
                arguments["ordered_floor_ids"] = [priority_floor]
            elif isinstance(priority_floor, list):
                arguments["ordered_floor_ids"] = [
                    str(floor_id)
                    for floor_id in priority_floor
                    if isinstance(floor_id, str) and floor_id.strip()
                ]
            arguments.pop("priority_floor", None)
            normalized["arguments"] = arguments

    if normalized.get("client_metadata") is None:
        normalized["client_metadata"] = {}

    return normalized


def _validate_action_arguments(action: ActionEnvelopeMA) -> str:
    """Validate action-specific argument shape after envelope parsing."""
    args_model = ACTION_TYPE_TO_ARGS.get(action.action_type)
    if args_model is None:
        return "unknown_action_type"
    try:
        args_model.model_validate(action.arguments)
    except Exception:
        return "arguments_invalid"
    return "ok"


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


_SPECIALIST_POLICY_KEY_SET = frozenset(SPECIALIST_POLICY_KEYS.values())
_PROMPT_METADATA_LABELS: tuple[tuple[str, str], ...] = (
    ("disaster_family", "disaster_family"),
    ("disaster_type", "disaster_family"),
    ("family", "disaster_family"),
    ("Disaster", "disaster_family"),
    ("tier", "tier"),
    ("Tier", "tier"),
    ("severity", "severity"),
    ("Severity", "severity"),
)


def _prompt_text(prompt: list[dict[str, str]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in prompt)


def _extract_prompt_scope_metadata(prompt: list[dict[str, str]]) -> dict[str, Any]:
    """Extract routing metadata from chat prompts without changing Policy.act."""

    text = _prompt_text(prompt)
    metadata: dict[str, Any] = {}

    families_match = re.search(
        r"(?im)(?:disaster_families|families)\s*[:=]\s*\[([^\]]+)\]",
        text,
    )
    if families_match:
        families = re.findall(r"[A-Za-z][A-Za-z0-9_\- ]*", families_match.group(1))
        if families:
            metadata["disaster_families"] = [family.strip() for family in families]

    for label, key in _PROMPT_METADATA_LABELS:
        if key in metadata:
            continue
        match = re.search(
            rf"(?im)^\s*{re.escape(label)}\s*[:=]\s*[\"']?([^\"'\n,;]+)",
            text,
        )
        if match:
            metadata[key] = match.group(1).strip()

    return metadata


def _normalize_specialist_policy_key(key: str) -> str:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in SPECIALIST_POLICY_KEYS:
        return SPECIALIST_POLICY_KEYS[normalized]
    if normalized in _SPECIALIST_POLICY_KEY_SET:
        return normalized
    raise ValueError(
        "ScopeRoutedFloorPolicy specialist keys must be disaster families "
        f"{sorted(SPECIALIST_POLICY_KEYS)} or policy keys "
        f"{sorted(_SPECIALIST_POLICY_KEY_SET)}; got {key!r}"
    )


class ScopeRoutedFloorPolicy:
    """Route floor-agent generation to frozen disaster-specialist policies.

    The rollout interface intentionally stays unchanged.  Routing metadata is
    extracted from the prompt (`Disaster: fire`, JSON-style fields, or similar),
    then passed through `training.scope_router.route_scope` so this wrapper and
    the offline planning/router code share one deterministic decision function.
    """

    def __init__(
        self,
        *,
        specialist_policies: Mapping[str, Policy],
        generalist_policy: Policy | None = None,
    ) -> None:
        if not specialist_policies:
            raise ValueError("ScopeRoutedFloorPolicy requires at least one specialist policy")

        normalized: dict[str, Policy] = {}
        for key, policy in specialist_policies.items():
            normalized[_normalize_specialist_policy_key(key)] = policy

        self._specialist_policies = normalized
        self._generalist_policy = generalist_policy

    @property
    def specialist_policy_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._specialist_policies))

    def _policy_for_prompt(self, prompt: list[dict[str, str]]) -> tuple[Policy, str]:
        decision = route_scope(_extract_prompt_scope_metadata(prompt))
        policy = self._specialist_policies.get(decision.policy_key)
        if policy is not None:
            return policy, decision.policy_key

        if self._generalist_policy is not None:
            return self._generalist_policy, GENERALIST_POLICY_KEY

        raise RuntimeError(
            "No frozen floor policy is available for disaster route "
            f"{decision.policy_key!r} (family={decision.disaster_family!r}, "
            f"reason={decision.reason!r}). Add the matching "
            "roles.frozen_floor_specialist_adapter_paths entry or provide "
            "roles.frozen_adapter_paths.floor_agent as a generalist fallback."
        )

    def act(
        self,
        prompt: list[dict[str, str]],
        agent_id: str,
        role: str,
    ) -> PolicyResult:
        if role != "floor_agent":
            raise ValueError("ScopeRoutedFloorPolicy only supports role='floor_agent'")
        policy, _policy_key = self._policy_for_prompt(prompt)
        return _as_policy_result(policy.act(prompt, agent_id, role))

    def act_batch(
        self,
        prompts: list[list[dict[str, str]]],
        agent_ids: list[str],
        roles: list[str],
    ) -> list[PolicyResult]:
        if not (len(prompts) == len(agent_ids) == len(roles)):
            raise ValueError("prompts, agent_ids, and roles must have the same length")
        if any(role != "floor_agent" for role in roles):
            raise ValueError("ScopeRoutedFloorPolicy only supports floor_agent batches")
        if not prompts:
            return []

        grouped: dict[str, tuple[Policy, list[int]]] = {}
        for idx, prompt in enumerate(prompts):
            policy, policy_key = self._policy_for_prompt(prompt)
            if policy_key not in grouped:
                grouped[policy_key] = (policy, [])
            grouped[policy_key][1].append(idx)

        outputs: list[PolicyResult | None] = [None] * len(prompts)
        for policy, indices in grouped.values():
            policy_prompts = [prompts[idx] for idx in indices]
            policy_agent_ids = [agent_ids[idx] for idx in indices]
            policy_roles = [roles[idx] for idx in indices]
            if hasattr(policy, "act_batch"):
                policy_outputs = policy.act_batch(policy_prompts, policy_agent_ids, policy_roles)  # type: ignore[attr-defined]
            else:
                policy_outputs = [
                    _as_policy_result(policy.act(prompt, aid, role))
                    for prompt, aid, role in zip(policy_prompts, policy_agent_ids, policy_roles, strict=True)
                ]
            for out_idx, original_idx in enumerate(indices):
                outputs[original_idx] = _as_policy_result(policy_outputs[out_idx])

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
    payload: Any | None = None
    parse_status = "ok"
    for candidate in _json_payload_candidates(completion_text):
        try:
            payload = json.loads(candidate)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if payload is None:
        if role == "orchestrator":
            payload = _salvage_orchestrator_priority_payload(
                completion_text,
                expected_episode_id=expected_episode_id,
                expected_round_id=expected_round_id,
                agent_id=agent_id,
            )
            if payload is not None:
                parse_status = "salvaged_invalid_json"
        if payload is None:
            return None, "invalid_json"

    payload = _normalize_action_payload(
        payload,
        expected_episode_id=expected_episode_id,
        expected_round_id=expected_round_id,
        agent_id=agent_id,
        role=role,
    )
    if payload is None:
        return None, "schema_invalid"

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

    argument_status = _validate_action_arguments(action)
    if argument_status != "ok":
        return None, argument_status

    return action, parse_status


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
            return self.act_batch([prompt], [agent_id], [role])[0]

        def act_batch(
            self,
            prompts: list[list[dict[str, str]]],
            agent_ids: list[str],
            roles: list[str],
        ) -> list[PolicyResult]:
            del agent_ids, roles
            if not prompts:
                return []

            rendered = [
                self._tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            _warn_once_if_prompt_truncated(
                tokenizer=self._tokenizer,
                rendered=rendered,
                max_prompt_tokens=self._max_prompt_tokens,
            )
            inputs = _call_tokenizer(
                self._tokenizer,
                rendered,
                return_tensors="pt",
                max_length=self._max_prompt_tokens,
                truncation=True,
                padding=True,
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

            input_width = inputs["input_ids"].shape[-1]
            results: list[PolicyResult] = []
            for row in outputs:
                generated = row[input_width:]
                generated_ids = (
                    generated.tolist()
                    if hasattr(generated, "tolist")
                    else list(generated)
                )
                results.append(
                    (
                        self._tokenizer.decode(
                            generated,
                            skip_special_tokens=True,
                        ),
                        generated_ids,
                    )
                )
            return results

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
                # The custom "unsloth" checkpointing path has triggered
                # in-place autograd failures in our GRPO smoke runs on
                # torch 2.8 / RTX 4090. Standard gradient checkpointing is
                # slower, but stable and still memory-friendly.
                use_gradient_checkpointing=True,
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
        if original_padding_side != "left":
            # Training tokenization temporarily right-pads for label masking.
            # Recover here instead of failing the rollout path.
            tokenizer.padding_side = "left"
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

                        FastLanguageModel.for_training(
                            self._model,
                            use_gradient_checkpointing=True,
                        )
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
