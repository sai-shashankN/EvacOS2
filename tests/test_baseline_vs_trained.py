import csv
import math
import shutil
import uuid
from pathlib import Path

from evaluation.baseline_vs_trained import run_comparison


def _tmp_dir() -> Path:
    path = Path(".phase19_test_tmp") / f"baseline_vs_trained_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_comparison_skip_trained_writes_nan():
    tmp_dir = _tmp_dir()
    output_csv = tmp_dir / "baseline_vs_trained.csv"
    try:
        run_comparison(
            trained_checkpoint=None,
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            output_csv=output_csv,
            skip_trained=True,
        )
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert set(rows[0]) == {
            "tier",
            "seed",
            "disaster_family",
            "role",
            "metric",
            "baseline",
            "trained",
            "delta",
            "rationale_mode",
            "schema_version",
        }
        assert math.isnan(float(rows[0]["trained"]))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_comparison_row_count_matches_roles_times_metrics():
    tmp_dir = _tmp_dir()
    output_csv = tmp_dir / "baseline_vs_trained.csv"
    try:
        run_comparison(
            trained_checkpoint=None,
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            output_csv=output_csv,
            skip_trained=True,
        )
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        role_metric_rows = 1 * 1 * 1 * 2 * 5
        team_metric_rows = 1 * 1 * 1 * 1
        assert len(rows) == role_metric_rows + team_metric_rows
        assert any(row["role"] == "team" and row["metric"] == "eval_score_pct" for row in rows)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
