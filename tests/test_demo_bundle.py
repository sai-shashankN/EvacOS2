from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from evaluation.demo_bundle import build_demo_bundle


def _tmp_dir() -> Path:
    path = Path(".phase26_test_tmp") / f"demo_bundle_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_demo_bundle_baseline_only_writes_summary_and_csv():
    tmp_dir = _tmp_dir()
    try:
        result = build_demo_bundle(
            trained_checkpoint=None,
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            output_dir=tmp_dir,
            skip_trained=True,
        )

        assert result.summary_md.exists()
        assert result.scorecard_md.exists()
        assert result.scorecard_json.exists()
        assert result.comparison_csv.exists()
        summary = result.summary_md.read_text(encoding="utf-8")
        scorecard = result.scorecard_md.read_text(encoding="utf-8")
        assert "Demo Bundle Summary" in summary
        assert "Baseline Metrics" in summary
        assert "trained fixed suite: skipped" in summary
        assert "EvacOS2 Submission Scorecard" in scorecard
        assert "Judge-Fast Take" in scorecard
        assert "no-trained-data metrics" in scorecard
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
