"""Thin sync Python client for the EvacOS-MA OpenEnv surface.

Uses urllib for zero extra deps. Returns typed Pydantic models.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Optional

from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    StepResultMA,
)
from evacos_ma.schemas.rewards import RewardsByRole


class OpenEnvClient:
    """Minimal sync client for the /openenv/* endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _get(self, path: str, params: Optional[dict[str, str]] = None) -> dict[str, Any]:
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}{path}{qs}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def reset(
        self,
        task_id: str = "openenv_fire_response",
        seed: Optional[int] = None,
        disaster_family: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if disaster_family is not None:
            payload["disaster_family"] = disaster_family
        if max_steps is not None:
            payload["max_steps"] = max_steps
        return self._post("/openenv/reset", payload)

    def step(self, bundle: ActionBundleMA) -> dict[str, Any]:
        return self._post("/openenv/step", bundle.model_dump())

    def state(self, episode_id: str = "") -> dict[str, Any]:
        params = {"episode_id": episode_id} if episode_id else None
        return self._get("/openenv/state", params)

    def schema(self) -> dict[str, Any]:
        return self._get("/openenv/schema")

    def health(self) -> dict[str, str]:
        return self._get("/openenv/health")

    def metadata(self) -> dict[str, Any]:
        return self._get("/openenv/metadata")
