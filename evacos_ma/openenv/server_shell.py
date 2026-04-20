"""OpenEnv server shell for EvacOS-MA.

Stub FastAPI router that validates request/response types against the
frozen MA schemas. Handlers return canned payloads that conform to the
schema — they do NOT drive the simulator yet (Phase 3+).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from evacos_ma.openenv import VERSION
from evacos_ma.openenv.debug import is_debug_state_enabled
from evacos_ma.openenv.manifest import MANIFEST
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    AgentRole,
    ActionTypeMA,
    FloorAgentObservationMA,
    InterFloorView,
    ObservationsByRole,
    OrchestratorObservationMA,
    StepResultMA,
    StepResultInfo,
    Tier,
)
from evacos_ma.schemas.rewards import RewardBreakdown, RoleReward, RewardsByRole

router = APIRouter(prefix="/openenv", tags=["openenv"])


# ---------------------------------------------------------------------------
# Request / response models for the OpenEnv surface
# ---------------------------------------------------------------------------

class ResetRequestMA(BaseModel):
    task_id: str = "task_1_fire_easy"
    seed: Optional[int] = None
    tier: str = "easy"


class OpenEnvResetResponse(BaseModel):
    episode_id: str
    step_result: StepResultMA


class OpenEnvStepResponse(BaseModel):
    step_result: StepResultMA


class StateResponse(BaseModel):
    episode_id: str = ""
    step: int = 0
    done: bool = False
    metadata: dict[str, Any] = {}
    full_state: Optional[dict[str, Any]] = None


class MetadataResponseMA(BaseModel):
    name: str
    description: str
    version: str
    manifest: dict[str, Any]


class SchemaResponseMA(BaseModel):
    action_bundle: dict
    observation_floor: dict
    observation_orchestrator: dict
    step_result: dict


class HealthResponseMA(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Stub payload builders
# ---------------------------------------------------------------------------

_EPISODE_ID_STUB = "ep_stub_0001"


def _stub_floor_obs(agent_id: str, floor_id: str) -> dict[str, Any]:
    return FloorAgentObservationMA(
        episode_id=_EPISODE_ID_STUB,
        round_id=0,
        role=AgentRole.floor_agent,
        agent_id=agent_id,
        step=0,
        max_steps=350,
        seed=42,
        tier=Tier.easy,
        disaster_family="fire",
        action_mask=[a.value for a in ActionTypeMA],
        floor_id=floor_id,
    ).model_dump()


def _stub_orchestrator_obs() -> dict[str, Any]:
    return OrchestratorObservationMA(
        episode_id=_EPISODE_ID_STUB,
        round_id=0,
        role=AgentRole.orchestrator,
        agent_id="orchestrator",
        step=0,
        max_steps=350,
        seed=42,
        tier=Tier.easy,
        disaster_family="fire",
        action_mask=[a.value for a in ActionTypeMA],
    ).model_dump()


def _stub_rewards() -> dict[str, Any]:
    return RewardsByRole(
        orchestrator=RoleReward(raw=0.0, normalized=0.0, breakdown=RewardBreakdown()),
        floors={f"floor_{i}_agent": RoleReward(raw=0.0, normalized=0.0, breakdown=RewardBreakdown()) for i in range(5)},
    ).model_dump()


def _stub_step_result() -> StepResultMA:
    return StepResultMA(
        observations_by_role=ObservationsByRole(
            orchestrator=OrchestratorObservationMA(
                episode_id=_EPISODE_ID_STUB,
                round_id=0,
                role=AgentRole.orchestrator,
                agent_id="orchestrator",
                step=0,
                max_steps=350,
                seed=42,
                tier=Tier.easy,
                disaster_family="fire",
                action_mask=[a.value for a in ActionTypeMA],
            ),
            floors={f"floor_{i}_agent": FloorAgentObservationMA(
                episode_id=_EPISODE_ID_STUB,
                round_id=0,
                role=AgentRole.floor_agent,
                agent_id=f"floor_{i}_agent",
                step=0,
                max_steps=350,
                seed=42,
                tier=Tier.easy,
                disaster_family="fire",
                action_mask=[a.value for a in ActionTypeMA],
                floor_id=f"floor_{i}",
            ) for i in range(5)},
        ),
        rewards_by_role=RewardsByRole(
            orchestrator=RoleReward(raw=0.0, normalized=0.0, breakdown=RewardBreakdown()),
            floors={f"floor_{i}_agent": RoleReward(raw=0.0, normalized=0.0, breakdown=RewardBreakdown()) for i in range(5)},
        ),
        done=False,
        done_reason=None,
        invalid_actions=[],
        round_events=[],
        info=StepResultInfo(),
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponseMA)
def health() -> HealthResponseMA:
    return HealthResponseMA(status="healthy", version=VERSION)


@router.get("/schema", response_model=SchemaResponseMA)
def schema() -> SchemaResponseMA:
    return SchemaResponseMA(
        action_bundle=ActionBundleMA.model_json_schema(),
        observation_floor=FloorAgentObservationMA.model_json_schema(),
        observation_orchestrator=OrchestratorObservationMA.model_json_schema(),
        step_result=StepResultMA.model_json_schema(),
    )


@router.get("/metadata", response_model=MetadataResponseMA)
def metadata() -> MetadataResponseMA:
    return MetadataResponseMA(
        name=MANIFEST["env_name"],
        description=MANIFEST["description"],
        version=MANIFEST["version"],
        manifest=MANIFEST,
    )


@router.post("/reset", response_model=OpenEnvResetResponse)
def reset(req: ResetRequestMA = Body(default=None)) -> OpenEnvResetResponse:
    if req is None:
        req = ResetRequestMA()
    sr = _stub_step_result()
    sr.episode_id = req.episode_id if hasattr(req, 'episode_id') else _EPISODE_ID_STUB
    return OpenEnvResetResponse(episode_id=_EPISODE_ID_STUB, step_result=sr)


@router.post("/step", response_model=OpenEnvStepResponse)
def step(bundle: ActionBundleMA) -> OpenEnvStepResponse:
    # Validate the bundle parses correctly (it does via FastAPI dependency)
    sr = _stub_step_result()
    return OpenEnvStepResponse(step_result=sr)


@router.get("/state", response_model=StateResponse)
def state(episode_id: str = "") -> StateResponse:
    resp = StateResponse(
        episode_id=episode_id or _EPISODE_ID_STUB,
        step=0,
        done=False,
        metadata={"version": VERSION},
        full_state=None,
    )
    if is_debug_state_enabled():
        resp.full_state = {
            "episode_id": episode_id or _EPISODE_ID_STUB,
            "building": {"floors": 5, "rooms_per_floor": 8},
            "civilians": {"total": 60, "saved": 0, "lost": 0},
            "hazards": [],
            "rng_seed": 42,
        }
    return resp
