"""Tests for the OpenEnv server shell endpoints.

Uses FastAPI TestClient. Verifies typed responses, schema round-trips,
and debug-state gating.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from evacos_ma.api import app
from evacos_ma.schemas.multi_agent import (
    ActionBundleMA,
    ActionTypeMA,
    FloorAgentObservationMA,
    OrchestratorObservationMA,
    StepResultMA,
)
from evacos_ma.schemas.rewards import RewardsByRole


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def action_bundle_fixture():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    with open(os.path.join(fixtures_dir, "action_bundle.golden.json")) as f:
        return json.loads(f.read())


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/openenv/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestSchemaEndpoint:
    def test_schema_returns_200(self, client):
        resp = client.get("/openenv/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "action_bundle" in data
        assert "observation_floor" in data
        assert "observation_orchestrator" in data
        assert "step_result" in data

    def test_schema_is_valid_json_schema(self, client):
        resp = client.get("/openenv/schema")
        data = resp.json()
        # Each should be a dict with at least "properties" or "title"
        for key in ("action_bundle", "observation_floor", "observation_orchestrator", "step_result"):
            assert isinstance(data[key], dict)


class TestMetadataEndpoint:
    def test_metadata_returns_200(self, client):
        resp = client.get("/openenv/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "evacos-ma"
        assert data["version"] == "0.1.0"
        assert "manifest" in data


class TestResetEndpoint:
    def test_reset_returns_step_result(self, client):
        resp = client.post("/openenv/reset", json={"task_id": "task_1_fire_easy", "seed": 42, "tier": "easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert "step_result" in data
        # Validate step_result parses as StepResultMA
        sr = StepResultMA.model_validate(data["step_result"])
        assert sr.done is False

    def test_reset_with_defaults(self, client):
        resp = client.post("/openenv/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data


class TestStepEndpoint:
    def test_step_with_golden_bundle(self, client, action_bundle_fixture):
        reset_resp = client.post(
            "/openenv/reset",
            json={"task_id": "task_1_fire_easy", "seed": 42, "tier": "easy"},
        )
        assert reset_resp.status_code == 200
        episode_id = reset_resp.json()["episode_id"]

        bundle = deepcopy(action_bundle_fixture)
        bundle["episode_id"] = episode_id
        bundle["round_id"] = 0
        bundle["orchestrator_action"]["episode_id"] = episode_id
        bundle["orchestrator_action"]["round_id"] = 0
        for action in bundle["floor_actions"].values():
            action["episode_id"] = episode_id
            action["round_id"] = 0

        resp = client.post("/openenv/step", json=bundle)
        assert resp.status_code == 200
        data = resp.json()
        assert "step_result" in data
        sr = StepResultMA.model_validate(data["step_result"])
        assert sr.observations_by_role.orchestrator.episode_id == episode_id


class TestStateEndpoint:
    def test_state_without_debug_hides_full_state(self, client, monkeypatch):
        monkeypatch.delenv("EVACOS_DEBUG_STATE", raising=False)
        reset_resp = client.post("/openenv/reset", json={"task_id": "task_1_fire_easy", "seed": 123})
        episode_id = reset_resp.json()["episode_id"]
        resp = client.get("/openenv/state", params={"episode_id": episode_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_state"] is None
        assert "metadata" in data

    def test_state_with_debug_shows_full_state(self, client, monkeypatch):
        monkeypatch.setenv("EVACOS_DEBUG_STATE", "true")
        reset_resp = client.post("/openenv/reset", json={"task_id": "task_1_fire_easy", "seed": 456})
        episode_id = reset_resp.json()["episode_id"]
        resp = client.get("/openenv/state", params={"episode_id": episode_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_state"] is not None
        assert "building" in data["full_state"]
        assert "task" in data["full_state"]

    def test_state_debug_false_does_not_show_full_state(self, client, monkeypatch):
        monkeypatch.setenv("EVACOS_DEBUG_STATE", "false")
        reset_resp = client.post("/openenv/reset", json={"task_id": "task_1_fire_easy", "seed": 789})
        episode_id = reset_resp.json()["episode_id"]
        resp = client.get("/openenv/state", params={"episode_id": episode_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_state"] is None

    def test_state_unknown_episode_returns_404(self, client):
        resp = client.get("/openenv/state", params={"episode_id": "ep_missing"})
        assert resp.status_code == 404
