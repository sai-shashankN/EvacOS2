import shutil
import uuid
from pathlib import Path

from training.policy_adapter import StubPolicy

from evaluation.rationale_sweep import run_rationale_sweep


def _tmp_dir() -> Path:
    path = Path(".phase19_test_tmp") / f"rationale_sweep_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_rationale_sweep_runs_three_modes():
    tmp_dir = _tmp_dir()
    try:
        result = run_rationale_sweep(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            output_path=tmp_dir / "rationale_sweep.json",
        )
        assert len(result.modes) == 3
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_suite_result_exposes_env_side_rationale_wired_true():
    tmp_dir = _tmp_dir()
    try:
        result = run_rationale_sweep(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            output_path=tmp_dir / "rationale_sweep.json",
        )
        assert all(suite.env_side_rationale_wired is True for suite in result.suites)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
