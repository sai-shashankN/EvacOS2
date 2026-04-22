"""OpenEnv server surface for EvacOS-MA.

This router exposes the live multi-agent simulator through a small FastAPI
surface that mirrors the OpenEnv-style reset / step / state flow.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from evacos_ma.openenv import VERSION
from evacos_ma.openenv.debug import is_debug_state_enabled
from evacos_ma.openenv.manifest import build_manifest
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
_OPENENV_ENV = EvacEnvironment()


# ---------------------------------------------------------------------------
# Request / response models for the OpenEnv surface
# ---------------------------------------------------------------------------

class ResetRequestMA(BaseModel):
    task_id: str = "task_1_fire_easy"
    seed: Optional[int] = None
    tier: str = "easy"
    disaster_family: Optional[str] = None
    max_steps: Optional[int] = None


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


def _empty_role_reward() -> RoleReward:
    return RoleReward(raw=0.0, normalized=0.0, breakdown=RewardBreakdown())


def _initial_step_result(observations: ObservationsByRole) -> StepResultMA:
    return StepResultMA(
        observations_by_role=observations,
        rewards_by_role=RewardsByRole(
            orchestrator=_empty_role_reward(),
            floors={
                agent_id: _empty_role_reward()
                for agent_id in sorted(observations.floors)
            },
        ),
        done=False,
        done_reason=None,
        invalid_actions=[],
        round_events=[],
        info=StepResultInfo(),
    )


def _http_exception_from_env_error(exc: ValueError) -> HTTPException:
    detail = str(exc)
    status_code = 404 if "Unknown episode_id" in detail else 400
    return HTTPException(status_code=status_code, detail=detail)


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
    manifest = build_manifest()
    return MetadataResponseMA(
        name=manifest["env_name"],
        description=manifest["description"],
        version=manifest["version"],
        manifest=manifest,
    )


@router.post("/reset", response_model=OpenEnvResetResponse)
def reset(req: ResetRequestMA = Body(default=None)) -> OpenEnvResetResponse:
    if req is None:
        req = ResetRequestMA()
    try:
        if req.disaster_family is not None:
            observations = _OPENENV_ENV.reset_multi_agent(
                task_id=req.task_id,
                seed=req.seed,
                procgen_tier=req.tier,
                procgen_disaster_family=DisasterType(req.disaster_family),
                procgen_max_steps=req.max_steps,
            )
        else:
            observations = _OPENENV_ENV.reset_multi_agent(
                task_id=req.task_id,
                seed=req.seed,
            )
    except ValueError as exc:
        raise _http_exception_from_env_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    episode_id, observations_by_role = observations
    return OpenEnvResetResponse(
        episode_id=episode_id,
        step_result=_initial_step_result(observations_by_role),
    )


@router.post("/step", response_model=OpenEnvStepResponse)
def step(bundle: ActionBundleMA) -> OpenEnvStepResponse:
    try:
        step_result = _OPENENV_ENV.step_multi_agent(bundle)
    except ValueError as exc:
        raise _http_exception_from_env_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OpenEnvStepResponse(step_result=step_result)


@router.get("/state", response_model=StateResponse)
def state(episode_id: str = "") -> StateResponse:
    try:
        public_state = _OPENENV_ENV.state(episode_id)
    except ValueError as exc:
        raise _http_exception_from_env_error(exc) from exc

    resp = StateResponse(
        episode_id=public_state.episode_id,
        step=public_state.step,
        done=public_state.done,
        metadata={
            "version": VERSION,
            "task_id": public_state.task_id,
            "termination_reason": public_state.termination_reason,
            "blocked_route_ids": public_state.blocked_route_ids,
        },
        full_state=None,
    )
    if is_debug_state_enabled():
        internal_state = _OPENENV_ENV.get_internal_state(public_state.episode_id)
        resp.full_state = {
            "episode_id": internal_state.episode_id,
            "step": internal_state.step,
            "done": internal_state.done,
            "termination_reason": internal_state.termination_reason,
            "task": internal_state.task.model_dump(mode="json"),
            "building": internal_state.building.model_dump(mode="json"),
            "metrics": internal_state.metrics.model_dump(mode="json"),
            "civilians_saved": internal_state.civilians_saved.model_dump(mode="json"),
            "civilians_lost": internal_state.civilians_lost.model_dump(mode="json"),
        }
    return resp
