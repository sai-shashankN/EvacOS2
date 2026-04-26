from __future__ import annotations

from fastapi.testclient import TestClient

from evacos_ma.api import app


def test_metadata_reflects_debug_state_enabled_when_env_set(monkeypatch):
    monkeypatch.setenv("EVACOS_DEBUG_STATE", "true")
    client = TestClient(app)

    resp = client.get("/openenv/metadata")

    assert resp.status_code == 200
    assert resp.json()["manifest"]["debug_state_enabled"] is True


def test_metadata_reflects_debug_state_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("EVACOS_DEBUG_STATE", raising=False)
    client = TestClient(app)

    resp = client.get("/openenv/metadata")

    assert resp.status_code == 200
    assert resp.json()["manifest"]["debug_state_enabled"] is False


def test_metadata_and_state_agree_on_debug_state(monkeypatch):
    client = TestClient(app)

    monkeypatch.setenv("EVACOS_DEBUG_STATE", "true")
    metadata_true = client.get("/openenv/metadata").json()
    reset_true = client.post("/openenv/reset", json={"task_id": "openenv_fire_response", "seed": 1}).json()
    state_true = client.get(
        "/openenv/state",
        params={"episode_id": reset_true["episode_id"]},
    ).json()
    assert metadata_true["manifest"]["debug_state_enabled"] is True
    assert metadata_true["manifest"]["debug_state_enabled"] == (state_true["full_state"] is not None)

    monkeypatch.delenv("EVACOS_DEBUG_STATE", raising=False)
    metadata_false = client.get("/openenv/metadata").json()
    reset_false = client.post("/openenv/reset", json={"task_id": "openenv_fire_response", "seed": 2}).json()
    state_false = client.get(
        "/openenv/state",
        params={"episode_id": reset_false["episode_id"]},
    ).json()
    assert metadata_false["manifest"]["debug_state_enabled"] is False
    assert metadata_false["manifest"]["debug_state_enabled"] == (state_false["full_state"] is not None)
