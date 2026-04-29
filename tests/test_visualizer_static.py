from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from evacos_ma.api import app


ROOT = Path(__file__).resolve().parents[1]
VISUALIZER = ROOT / "visualizer"


def test_visualizer_static_mount_serves_index_and_sample_trace():
    client = TestClient(app)

    index_resp = client.get("/visualizer/")
    assert index_resp.status_code == 200
    assert "EvacOS2 Replay Visualizer" in index_resp.text
    assert 'type="importmap"' in index_resp.text
    assert "./js/app.js" in index_resp.text

    sample_resp = client.get("/visualizer/sample_trace.json")
    assert sample_resp.status_code == 200
    payload = sample_resp.json()
    assert payload["schema_version"] == "visualization_trace_v1"
    assert len(payload["building"]["floors"]) >= 3
    assert payload["frames"]


def test_visualizer_sample_trace_matches_contract():
    payload = json.loads((VISUALIZER / "sample_trace.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "visualization_trace_v1"
    assert payload["trajectory_id"]
    assert {floor["floor_id"] for floor in payload["building"]["floors"]} == {0, 1, 2, 3}
    first_frame = payload["frames"][0]
    assert "per_floor_civilians" in first_frame
    assert "per_floor_hazard_severity" in first_frame
    assert "directive_feed" in first_frame
    assert "override_feed" in first_frame
    assert "reward_ticker" in first_frame
    assert "score_snapshot" in first_frame
    assert "room_states" in first_frame
    assert first_frame["room_states"]["F2_R2"]["disaster_type"] == "fire"
    assert payload["frames"][-1]["room_states"]["F2_R2"]["casualties"] >= 1


def test_visualizer_js_keeps_legacy_trace_support_and_no_build_chain():
    trace_js = (VISUALIZER / "js" / "trace.js").read_text(encoding="utf-8")
    app_js = (VISUALIZER / "js" / "app.js").read_text(encoding="utf-8")
    index_html = (VISUALIZER / "index.html").read_text(encoding="utf-8")

    assert "Array.isArray(payload)" in trace_js
    assert "synthesizeDefaultBuilding" in trace_js
    assert "floor_action_types" in trace_js
    assert "normalizeRoomStates" in trace_js
    assert 'await import("three")' in app_js
    assert "startLivePolling" in app_js
    assert "quadraticBezier" in app_js
    assert "FRAME_SECONDS" in app_js
    assert "visualTime" in app_js
    assert "wrapFrame" in app_js
    assert "floorStrip" in app_js
    assert "pickCriticalFloorKey" in app_js
    assert "setViewMode" in app_js
    assert "baseOpacity" not in app_js
    assert "material.opacity = Math.min(material.opacity ?? 1, weight)" in app_js
    assert "deadCivilians" in app_js
    assert "disasterOverlay" in app_js
    assert "room_states" in app_js
    assert "data-view-mode=\"solo\"" in index_html
    assert "data-view-mode=\"all\"" in index_html
    assert ">Building<" in index_html
    assert "https://cdn.jsdelivr.net/npm/three" in index_html
    assert not (ROOT / "visualizer" / "package.json").exists()
