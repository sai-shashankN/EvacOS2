from pathlib import Path

from training.policy_adapter import StubPolicy

from evaluation.fixed_suite import FixedSuiteResult, run_fixed_suite


def test_run_fixed_suite_returns_single_episode(tmp_path: Path):
    result = run_fixed_suite(
        lambda: StubPolicy(seed=0),
        tiers=("easy",),
        seeds=(42,),
        disaster_families=("fire",),
        max_rounds=50,
        label="test",
        output_dir=tmp_path,
    )
    assert isinstance(result, FixedSuiteResult)
    assert len(result.episodes) == 1


def test_fixed_suite_json_round_trip(tmp_path: Path):
    result = run_fixed_suite(
        lambda: StubPolicy(seed=0),
        tiers=("easy",),
        seeds=(42,),
        disaster_families=("fire",),
        max_rounds=50,
        label="roundtrip",
        output_dir=tmp_path,
    )
    payload = (tmp_path / "fixed_suite_roundtrip_linear_capped.json").read_text(encoding="utf-8")
    restored = FixedSuiteResult.model_validate_json(payload)
    assert restored == result


def test_fixed_suite_eval_does_not_mutate_normalizer(tmp_path: Path, monkeypatch):
    update_calls = []

    def fail_update(self, role, tier, raw):
        update_calls.append((role, tier, raw))
        raise AssertionError("RewardNormalizer.update should not be called in eval mode")

    monkeypatch.setattr("training.reward.RewardNormalizer.update", fail_update)
    run_fixed_suite(
        lambda: StubPolicy(seed=0),
        tiers=("easy",),
        seeds=(42,),
        disaster_families=("fire",),
        max_rounds=20,
        output_dir=tmp_path,
    )
    assert update_calls == []
