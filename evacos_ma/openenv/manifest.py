"""OpenEnv manifest for EvacOS-MA.

Serializes to the YAML shape OpenEnv expects for this environment.
"""

from __future__ import annotations

from typing import Any

from evacos_ma.openenv import VERSION
from evacos_ma.openenv.debug import is_debug_state_enabled
from evacos_ma.task_registry import TASKS

# Build task list from the existing task registry
_TASK_LIST: list[dict[str, str]] = [
    {"task_id": t.task_id, "name": t.name, "difficulty": t.difficulty}
    for t in TASKS.values()
]

MANIFEST: dict[str, Any] = {
    "env_name": "evacos-ma",
    "version": VERSION,
    "description": "Hierarchical Multi-Agent Evacuation Environment (Round 2)",
    "tasks": _TASK_LIST,
    "action_schema_ref": "evacos_ma.schemas.multi_agent.ActionEnvelopeMA",
    "observation_schema_ref": "evacos_ma.schemas.multi_agent.RoleObservationMA",
    "supported_tiers": ["easy", "medium", "hard", "brutal"],
    "agent_topology": {
        "orchestrator_count": 1,
        "floor_agent_count": 5,
    },
    "protocol": {
        "reset": "/openenv/reset",
        "step": "/openenv/step",
        "state": "/openenv/state",
        "schema": "/openenv/schema",
        "health": "/openenv/health",
        "metadata": "/openenv/metadata",
    },
}


def build_manifest() -> dict[str, Any]:
    """Build a manifest view with runtime-sensitive fields evaluated now."""
    return {
        **MANIFEST,
        "debug_state_enabled": is_debug_state_enabled(),
    }
