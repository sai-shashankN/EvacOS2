"""Minimal FastAPI shell for EvacOS-MA.

Stripped down from Round 1: no dashboard, no baseline runner, no render endpoint,
no Unity bridge. Only the core reset/step/state/grader/tasks endpoints remain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter

from evacos_ma.env import EvacEnvironment
from evacos_ma.grader import grade_episode
from evacos_ma.models import (
    Action,
    Observation,
    Reward,
    ScoreValue,
    StateView,
    StepInfo,
    TaskSpecPublic,
)
from evacos_ma.task_registry import get_all_tasks, get_tasks_public
from evacos_ma.openenv.server_shell import router as openenv_router

app = FastAPI(
    title="EvacOS-MA",
    description="Multi-agent evacuation environment (Round 2)",
    version="0.1.0",
)

# Include the multi-agent OpenEnv router under /openenv prefix
app.include_router(openenv_router)

_VISUALIZER_DIR = Path(__file__).resolve().parents[1] / "visualizer"
if _VISUALIZER_DIR.exists():
    app.mount(
        "/visualizer",
        StaticFiles(directory=_VISUALIZER_DIR, html=True),
        name="visualizer",
    )

env = EvacEnvironment()


class ResetRequest(BaseModel):
    task_id: str
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: Observation


class StepResponse(BaseModel):
    episode_id: str
    observation: Observation
    reward: Reward
    done: bool
    info: StepInfo


class TasksResponse(BaseModel):
    tasks: list[TaskSpecPublic]
    action_schema: dict[str, str]
    version: str = "0.1.0"


class GraderRequest(BaseModel):
    episode_id: str


class GraderResponse(BaseModel):
    episode_id: str
    task_id: str
    score: ScoreValue
    breakdown: dict[str, float]


class MetadataResponse(BaseModel):
    name: str
    description: str
    version: str


class SchemaResponse(BaseModel):
    action: dict
    observation: dict
    state: dict


def _default_reset_request() -> ResetRequest:
    default_task = get_all_tasks()[0]
    return ResetRequest(task_id=default_task.task_id, seed=None)


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest | None = Body(default=None)) -> ResetResponse:
    if req is None:
        req = _default_reset_request()
    try:
        episode_id, observation = env.reset(req.task_id, req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResetResponse(episode_id=episode_id, observation=observation)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        observation, reward, done, info = env.step(action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Episode not found: {exc}") from exc
    except ValueError as exc:
        if "Unknown episode_id" in str(exc):
            raise HTTPException(status_code=404, detail=f"Episode not found: {action.episode_id}") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StepResponse(
        episode_id=action.episode_id,
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateView)
def get_state(episode_id: str) -> StateView:
    try:
        return env.state(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Episode not found: {exc}") from exc
    except ValueError as exc:
        if "Unknown episode_id" in str(exc):
            raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}") from exc
        raise


@app.get("/tasks", response_model=TasksResponse)
def get_tasks() -> TasksResponse:
    action_schema = {
        "action_type": "string (required) - one of: route_civilians, evacuate_floor, prioritize_room, block_route, call_elevator, open_exit, lockdown_room, request_render, wait",
        "episode_id": "string (required) - episode ID from /reset",
        "expected_step": "int (required) - current step number",
    }
    return TasksResponse(tasks=get_tasks_public(), action_schema=action_schema)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        name=app.title,
        description=app.description or "",
        version=app.version,
    )


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=TypeAdapter(Action).json_schema(),
        observation=Observation.model_json_schema(),
        state=StateView.model_json_schema(),
    )


@app.post("/grader", response_model=GraderResponse)
def grade(req: GraderRequest) -> GraderResponse:
    try:
        episode = env.get_internal_state(req.episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Episode not found: {exc}") from exc
    except ValueError as exc:
        if "Unknown episode_id" in str(exc):
            raise HTTPException(status_code=404, detail=f"Episode not found: {req.episode_id}") from exc
        raise

    if not episode.done:
        raise HTTPException(status_code=400, detail="Episode is not finished yet")

    result = grade_episode(episode)
    return GraderResponse(
        episode_id=req.episode_id,
        task_id=episode.task.task_id,
        score=result["score"],
        breakdown=result["breakdown"],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "EvacOS-MA", "version": "0.1.0", "status": "running"}
