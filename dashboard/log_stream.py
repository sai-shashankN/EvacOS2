"""Read-only log streaming for the Phase 8 dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel


class EpisodeMeta(BaseModel):
    episode_id: str
    seed: int | None = None
    tier: str | None = None
    disaster_family: str | None = None
    total_steps: int | None = None
    checkpoint_tag: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def list_episodes(log_dir: Path) -> list[EpisodeMeta]:
    rows = _read_jsonl(log_dir / "episode_summary.jsonl")
    metas = [
        EpisodeMeta(
            episode_id=str(row.get("episode_id")),
            seed=row.get("seed"),
            tier=row.get("tier"),
            disaster_family=row.get("disaster_family"),
            total_steps=row.get("total_steps"),
            checkpoint_tag=row.get("checkpoint_tag"),
        )
        for row in rows
    ]
    metas.sort(key=lambda item: item.episode_id)
    return metas


def _default_floor_map(source: dict[str, Any]) -> dict[str, float]:
    if "per_floor_hazard_severity" in source:
        return dict(source["per_floor_hazard_severity"])
    if "per_floor_civilians" in source:
        return {str(key): 0.0 for key in source["per_floor_civilians"]}
    floor_actions = source.get("floor_action_types", {})
    return {agent.replace("_agent", ""): 0.0 for agent in sorted(floor_actions)}


def _default_civilians(source: dict[str, Any]) -> dict[str, int]:
    if "per_floor_civilians" in source:
        return {str(key): int(value) for key, value in source["per_floor_civilians"].items()}
    floor_actions = source.get("floor_action_types", {})
    return {agent.replace("_agent", ""): 0 for agent in sorted(floor_actions)}


def _build_payload(
    round_row: dict[str, Any],
    action_rows: list[dict[str, Any]],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    directive_feed = round_row.get("directive_feed")
    override_feed = round_row.get("override_feed")
    if directive_feed is None:
        directive_feed = [
            row for row in action_rows if row.get("action_type") == "broadcast_directive"
        ]
    if override_feed is None:
        override_feed = [
            row for row in action_rows if row.get("action_type") == "override_floor_agent"
        ]

    total_steps = int((summary or {}).get("total_steps", round_row.get("round_id", 0) + 1))
    done = bool(round_row.get("done", round_row.get("round_id", 0) >= max(total_steps - 1, 0)))
    return {
        "episode_id": round_row.get("episode_id"),
        "round_id": int(round_row.get("round_id", 0)),
        "per_floor_civilians": _default_civilians(round_row),
        "per_floor_hazard_severity": _default_floor_map(round_row),
        "directive_feed": directive_feed,
        "override_feed": override_feed,
        "reward_ticker": round_row.get("reward_ticker", {}),
        "done": done,
        "score_snapshot": round_row.get(
            "score_snapshot",
            {
                "total_steps": total_steps,
                "checkpoint_tag": (summary or {}).get("checkpoint_tag"),
            },
        ),
    }


def tail_episode(
    log_dir: Path,
    episode_id: str,
    follow: bool,
    *,
    follow_timeout_s: float = 5.0,
) -> Iterator[dict[str, Any]]:
    yielded: set[int] = set()
    started = time.monotonic()
    summary_rows = _read_jsonl(log_dir / "episode_summary.jsonl")
    summary = next((row for row in summary_rows if row.get("episode_id") == episode_id), None)

    while True:
        round_rows = [
            row for row in _read_jsonl(log_dir / "round_trace.jsonl")
            if row.get("episode_id") == episode_id
        ]
        action_rows_all = [
            row for row in _read_jsonl(log_dir / "action_trace.jsonl")
            if row.get("episode_id") == episode_id
        ]
        round_rows.sort(key=lambda row: int(row.get("round_id", 0)))
        for round_row in round_rows:
            round_id = int(round_row.get("round_id", 0))
            if round_id in yielded:
                continue
            round_actions = [row for row in action_rows_all if int(row.get("round_id", 0)) == round_id]
            yielded.add(round_id)
            yield _build_payload(round_row, round_actions, summary)

        done_round = None if summary is None else max(int(summary.get("total_steps", 0)) - 1, 0)
        if not follow or (done_round is not None and done_round in yielded):
            break
        if time.monotonic() - started >= follow_timeout_s:
            break
        time.sleep(0.25)
        summary_rows = _read_jsonl(log_dir / "episode_summary.jsonl")
        summary = next((row for row in summary_rows if row.get("episode_id") == episode_id), summary)
