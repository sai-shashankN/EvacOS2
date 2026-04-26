"""OpenEnv manifest for EvacOS-MA.

Serializes to the YAML shape OpenEnv expects for this environment.
"""

from __future__ import annotations

from typing import Any

from evacos_ma.openenv import VERSION
from evacos_ma.openenv.debug import is_debug_state_enabled

_TASK_LIST: list[dict[str, str]] = [
    {
        "task_id": "openenv_fire_response",
        "name": "Fire Response Evacuation",
        "disaster_family": "fire",
    },
    {
        "task_id": "openenv_flood_response",
        "name": "Flood Response Evacuation",
        "disaster_family": "flood",
    },
    {
        "task_id": "openenv_gas_response",
        "name": "Gas Response Evacuation",
        "disaster_family": "gas",
    },
]

MANIFEST: dict[str, Any] = {
    "env_name": "evacos-ma",
    "version": VERSION,
    "description": "Hierarchical multi-agent evacuation environment for fire, flood, and gas response",
    "tasks": _TASK_LIST,
    "action_schema_ref": "evacos_ma.schemas.multi_agent.ActionBundleMA",
    "observation_schema_ref": "evacos_ma.schemas.multi_agent.RoleObservationMA",
    "supported_disaster_families": ["fire", "flood", "gas"],
    "scenario_contract": "default OpenEnv response lane",
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
