from pathlib import Path

from training.policy_adapter import StubPolicy

from evaluation.rationale_sweep import run_rationale_sweep


def test_rationale_sweep_runs_three_modes(tmp_path: Path):
    result = run_rationale_sweep(
        lambda: StubPolicy(seed=0),
        tiers=("easy",),
        seeds=(42,),
        disaster_families=("fire",),
        output_path=tmp_path / "rationale_sweep.json",
    )
    assert len(result.modes) == 3
