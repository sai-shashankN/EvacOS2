import csv
import math
from pathlib import Path

from evaluation.baseline_vs_trained import run_comparison


def test_run_comparison_skip_trained_writes_nan(tmp_path: Path):
    output_csv = tmp_path / "baseline_vs_trained.csv"
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


def test_comparison_row_count_matches_roles_times_metrics(tmp_path: Path):
    output_csv = tmp_path / "baseline_vs_trained.csv"
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
    assert len(rows) == 1 * 1 * 1 * 2 * 5
