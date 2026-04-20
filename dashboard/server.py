"""FastAPI server for the Phase 8 read-only dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .log_stream import list_episodes, tail_episode

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
LOG_ROOT = Path("outputs/logs")

app = FastAPI(title="Phase 8 Dashboard", version="2026.04.20")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/static/{path:path}")
def static_files(path: str) -> FileResponse:
    file_path = STATIC_ROOT / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="static file not found")
    return FileResponse(file_path)


@app.get("/episodes")
def episodes() -> JSONResponse:
    payload = [item.model_dump(mode="json") for item in list_episodes(LOG_ROOT)]
    return JSONResponse(payload)


@app.get("/stream")
def stream(
    episode_id: str = Query(...),
    follow: bool = Query(True),
) -> StreamingResponse:
    def event_stream():
        for payload in tail_episode(LOG_ROOT, episode_id, follow=follow):
            yield f"data: {json.dumps(payload, sort_keys=True)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
