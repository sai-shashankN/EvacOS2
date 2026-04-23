"""Deterministic routing for disaster-specialist policy stacks.

The router is deliberately small and dependency-free: it can sit in front of
fire/flood/gas specialist checkpoints while falling back to a generalist for
unknown, mixed, or cascading incidents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

GENERALIST_POLICY_KEY = "generalist"
SPECIALIST_POLICY_KEYS: dict[str, str] = {
    "fire": "fire_specialist",
    "flood": "flood_specialist",
    "gas": "gas_specialist",
}
CASCADE_FAMILIES = frozenset(
    {
        "active_threat",
        "cascade",
        "multi",
        "multi_cascade",
        "structural",
    }
)
_FAMILY_ALIASES = {
    "gas_leak": "gas",
    "gasleak": "gas",
    "smoke": "fire",
    "water": "flood",
}


@dataclass(frozen=True)
class ScopeDecision:
    """Selected policy lane for a scenario or episode."""

    policy_key: str
    disaster_family: str
    reason: str
    tier: str | None = None
    severity: float | None = None

    @property
    def uses_specialist(self) -> bool:
        return self.policy_key != GENERALIST_POLICY_KEY


def route_scope(metadata: Mapping[str, Any] | Any) -> ScopeDecision:
    """Route scenario metadata to a specialist or generalist policy key.

    Expected metadata keys are intentionally permissive because callers may
    pass OpenEnv reset metadata, rollout metadata, or a light object wrapper.
    The canonical key is ``disaster_family``; ``disaster_type`` and
    ``disaster_families`` are accepted for compatibility.
    """

    tier = _coerce_optional_str(_read(metadata, "tier"))
    severity = _coerce_optional_float(_read(metadata, "severity"))
    family = _extract_family(metadata)

    if _has_multi_family_marker(metadata):
        return ScopeDecision(
            policy_key=GENERALIST_POLICY_KEY,
            disaster_family=family or "mixed",
            reason="multi_disaster_or_cascade",
            tier=tier,
            severity=severity,
        )

    if not family:
        return ScopeDecision(
            policy_key=GENERALIST_POLICY_KEY,
            disaster_family="unknown",
            reason="unknown_disaster_family",
            tier=tier,
            severity=severity,
        )

    if family in CASCADE_FAMILIES:
        return ScopeDecision(
            policy_key=GENERALIST_POLICY_KEY,
            disaster_family=family,
            reason="multi_disaster_or_cascade",
            tier=tier,
            severity=severity,
        )

    policy_key = SPECIALIST_POLICY_KEYS.get(family)
    if policy_key is None:
        return ScopeDecision(
            policy_key=GENERALIST_POLICY_KEY,
            disaster_family=family,
            reason="unsupported_disaster_family",
            tier=tier,
            severity=severity,
        )

    return ScopeDecision(
        policy_key=policy_key,
        disaster_family=family,
        reason=f"single_family_{family}",
        tier=tier,
        severity=severity,
    )


def _extract_family(metadata: Mapping[str, Any] | Any) -> str:
    families = _normalize_family_list(_read(metadata, "disaster_families"))
    if len(families) == 1:
        return families[0]
    if len(families) > 1:
        return "+".join(families)

    for key in ("disaster_family", "disaster_type", "family"):
        family = _normalize_family(_read(metadata, key))
        if family:
            return family
    return ""


def _has_multi_family_marker(metadata: Mapping[str, Any] | Any) -> bool:
    families = _normalize_family_list(_read(metadata, "disaster_families"))
    if len(families) > 1:
        return True
    if _truthy_marker(_read(metadata, "cascade_hint")):
        return True
    if _truthy_marker(_read(metadata, "is_cascade")):
        return True
    if _truthy_marker(_read(metadata, "multi_disaster")):
        return True
    return False


def _normalize_family_list(value: Any) -> list[str]:
    if value is None or isinstance(value, (str, bytes)):
        family = _normalize_family(value)
        return [family] if family else []

    try:
        values = list(value)
    except TypeError:
        family = _normalize_family(value)
        return [family] if family else []

    normalized = [_normalize_family(item) for item in values]
    return sorted({item for item in normalized if item})


def _normalize_family(value: Any) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        value = enum_value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _FAMILY_ALIASES.get(text, text)


def _read(metadata: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)
