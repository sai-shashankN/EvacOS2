import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest

from training.metrics import (
    _METRICS_COLUMNS,
    append_training_metrics_row,
    read_training_metrics_rows,
    write_training_metrics_rows,
)


def _tmp_dir() -> Path:
    path = Path(tempfile.gettempdir()) / f"evacos_metrics_test_{os.getpid()}_{random.randint(0, 99999)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_append_training_metrics_row_rejects_stale_header() -> None:
    tmp_path = _tmp_dir()
    csv_path = tmp_path / "metrics.csv"
    try:
        csv_path.write_text("step,wall_seconds,invalid_action_rate\n0,1.0,0.0\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="header does not match"):
            append_training_metrics_row(csv_path, {"step": 1})
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_append_training_metrics_row_accepts_current_header() -> None:
    tmp_path = _tmp_dir()
    csv_path = tmp_path / "metrics.csv"
    try:
        append_training_metrics_row(csv_path, {"step": 0, "wall_seconds": 1.0})
        append_training_metrics_row(csv_path, {"step": 1, "wall_seconds": 2.0})

        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",") == _METRICS_COLUMNS
        assert len(lines) == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_training_metrics_snapshot_helpers_round_trip_current_header() -> None:
    tmp_path = _tmp_dir()
    csv_path = tmp_path / "checkpoint" / "metrics_window.csv"
    try:
        rows = [
            {"step": 9, "wall_seconds": 12.5, "mean_norm_reward_floor": 0.2},
            {"step": 19, "wall_seconds": 25.0, "mean_norm_reward_floor": 0.4},
        ]

        write_training_metrics_rows(csv_path, rows)
        loaded = read_training_metrics_rows(csv_path)

        assert csv_path.read_text(encoding="utf-8").splitlines()[0].split(",") == _METRICS_COLUMNS
        assert [row["step"] for row in loaded] == ["9", "19"]
        assert loaded[-1]["mean_norm_reward_floor"] == "0.4"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
