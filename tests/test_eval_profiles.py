from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from evaluation.demo_bundle import DemoBundleResult
from scripts import eval_profiles


def _tmp_dir() -> Path:
    path = Path(".phase_eval_profiles_tmp") / f"eval_profiles_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_result(output_dir: Path) -> DemoBundleResult:
    return DemoBundleResult(
        output_dir=output_dir,
        summary_md=output_dir / "demo_bundle_summary.md",
        scorecard_md=output_dir / "submission_scorecard.md",
        scorecard_json=output_dir / "submission_scorecard.json",
        comparison_csv=output_dir / "baseline_vs_trained.csv",
        baseline_json=output_dir / "fixed_suite_baseline_linear_capped.json",
        trained_json=None,
        plot_paths=[],
    )


def test_3b_fire_profile_defaults_to_fire_only(monkeypatch):
    calls: list[dict] = []
    tmp_dir = _tmp_dir()

    def fake_build_demo_bundle(**kwargs):
        calls.append(kwargs)
        return _fake_result(kwargs["output_dir"])

    monkeypatch.setattr(eval_profiles, "build_demo_bundle", fake_build_demo_bundle)

    try:
        eval_profiles.run_profile(
            "3b-fire",
            ["--skip-trained", "--output-dir", str(tmp_dir / "fire")],
        )

        assert calls[0]["disaster_families"] == ("fire",)
        assert calls[0]["tiers"] == ("easy", "medium", "hard", "brutal")
        assert calls[0]["config_path"] == Path(
            "training/config.remote-unsloth-3b-fire-floor-specialist.yaml"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_7b_orchestrator_profile_defaults_to_three_specialists(monkeypatch):
    calls: list[dict] = []
    tmp_dir = _tmp_dir()

    def fake_build_demo_bundle(**kwargs):
        calls.append(kwargs)
        return _fake_result(kwargs["output_dir"])

    monkeypatch.setattr(eval_profiles, "build_demo_bundle", fake_build_demo_bundle)

    try:
        eval_profiles.run_profile(
            "7b-orchestrator",
            ["--skip-trained", "--output-dir", str(tmp_dir / "orch")],
        )

        assert calls[0]["disaster_families"] == ("fire", "flood", "gas")
        assert calls[0]["config_path"] == Path(
            "training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
