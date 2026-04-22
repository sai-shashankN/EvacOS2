from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from dashboard import log_stream


def _append_jsonl(path: Path, *rows: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _tmp_dir() -> Path:
    path = Path(".phase21_test_tmp") / f"log_stream_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_tail_episode_returns_rounds_in_order_for_snapshot() -> None:
    tmp_path = _tmp_dir()
    episode_id = "ep-1"
    try:
        _append_jsonl(
            tmp_path / "episode_summary.jsonl",
            {
                "episode_id": episode_id,
                "total_steps": 3,
                "checkpoint_tag": "ckpt",
            },
        )
        _append_jsonl(
            tmp_path / "round_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 0,
                "per_floor_civilians": {"floor_0": 2},
                "per_floor_hazard_severity": {"floor_0": 0.1},
                "done": False,
            },
            {
                "episode_id": episode_id,
                "round_id": 1,
                "per_floor_civilians": {"floor_0": 1},
                "per_floor_hazard_severity": {"floor_0": 0.2},
                "done": False,
            },
        )
        _append_jsonl(
            tmp_path / "action_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 0,
                "action_id": "a0",
                "action_type": "broadcast_directive",
            },
            {
                "episode_id": episode_id,
                "round_id": 1,
                "action_id": "a1",
                "action_type": "override_floor_agent",
            },
        )

        payloads = list(log_stream.tail_episode(tmp_path, episode_id, follow=False))

        assert [payload["round_id"] for payload in payloads] == [0, 1]

        _append_jsonl(
            tmp_path / "round_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 2,
                "per_floor_civilians": {"floor_0": 0},
                "per_floor_hazard_severity": {"floor_0": 0.3},
                "done": True,
            },
        )
        _append_jsonl(
            tmp_path / "action_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 2,
                "action_id": "a2",
                "action_type": "broadcast_directive",
            },
        )

        payloads = list(log_stream.tail_episode(tmp_path, episode_id, follow=False))

        assert [payload["round_id"] for payload in payloads] == [0, 1, 2]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_tail_episode_follow_reads_trace_files_incrementally(monkeypatch) -> None:
    tmp_path = _tmp_dir()
    episode_id = "ep-follow"
    try:
        _append_jsonl(
            tmp_path / "episode_summary.jsonl",
            {
                "episode_id": episode_id,
                "total_steps": 3,
                "checkpoint_tag": "ckpt",
            },
        )
        _append_jsonl(
            tmp_path / "round_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 0,
                "per_floor_civilians": {"floor_0": 2},
                "per_floor_hazard_severity": {"floor_0": 0.1},
            },
            {
                "episode_id": episode_id,
                "round_id": 1,
                "per_floor_civilians": {"floor_0": 1},
                "per_floor_hazard_severity": {"floor_0": 0.2},
            },
        )
        _append_jsonl(
            tmp_path / "action_trace.jsonl",
            {
                "episode_id": episode_id,
                "round_id": 0,
                "action_id": "a0",
                "action_type": "broadcast_directive",
            },
            {
                "episode_id": episode_id,
                "round_id": 1,
                "action_id": "a1",
                "action_type": "override_floor_agent",
            },
        )

        read_text_calls: list[Path] = []
        original_read_text = Path.read_text

        def counting_read_text(self: Path, *args, **kwargs):
            read_text_calls.append(self)
            return original_read_text(self, *args, **kwargs)

        sleep_calls = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            if sleep_calls["count"] == 0:
                _append_jsonl(
                    tmp_path / "round_trace.jsonl",
                    {
                        "episode_id": episode_id,
                        "round_id": 2,
                        "per_floor_civilians": {"floor_0": 0},
                        "per_floor_hazard_severity": {"floor_0": 0.3},
                        "done": True,
                    },
                )
                _append_jsonl(
                    tmp_path / "action_trace.jsonl",
                    {
                        "episode_id": episode_id,
                        "round_id": 2,
                        "action_id": "a2",
                        "action_type": "broadcast_directive",
                    },
                )
            sleep_calls["count"] += 1

        monotonic_values = iter([0.0, 0.1, 0.2, 0.3])

        monkeypatch.setattr(Path, "read_text", counting_read_text)
        monkeypatch.setattr(log_stream.time, "sleep", fake_sleep)
        monkeypatch.setattr(log_stream.time, "monotonic", lambda: next(monotonic_values))

        payloads = list(log_stream.tail_episode(tmp_path, episode_id, follow=True, follow_timeout_s=1.0))

        assert [payload["round_id"] for payload in payloads] == [0, 1, 2]
        assert read_text_calls.count(tmp_path / "episode_summary.jsonl") <= 1
        assert (tmp_path / "round_trace.jsonl") not in read_text_calls
        assert (tmp_path / "action_trace.jsonl") not in read_text_calls
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
