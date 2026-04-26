import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from evacos_ma.models import DisasterType
from training.config_schema import TrainingConfig
from training.reward import RewardNormalizer
from training.train import _run_eval


def _tmp_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="phase22_run_eval_"))


def test_run_eval_expands_seed_family_pairs_per_tier():
    config = TrainingConfig()
    config.eval.tiers = ["easy"]
    config.eval.seeds = [1, 2, 3]
    config.rollout.max_rounds_per_episode = 7

    calls: list[dict] = []
    tmp_path = _tmp_path()

    def stub_collect_batch(env, policy, curriculum, **kwargs):
        calls.append(
            {
                "env": env,
                "policy": policy,
                "tier": curriculum.suggest_next_tier("fire"),
                **kwargs,
            }
        )
        return [SimpleNamespace(samples=[])]

    normalizer = RewardNormalizer()
    try:
        eval_results = _run_eval(
            env=object(),
            policy=object(),
            config=config,
            normalizer=normalizer,
            checkpoint_tag="test",
            model_name="stub",
            collect_batch=stub_collect_batch,
            disaster_families=[DisasterType.fire, DisasterType.flood],
            jsonl_dir=tmp_path,
        )

        assert len(eval_results) == 1
        assert len(calls) == 1
        assert [call["tier"] for call in calls] == ["easy"]
        for call in calls:
            assert call["num_episodes"] == 6
            assert call["is_eval"] is True
            assert call["jsonl_dir"] == tmp_path
            assert call["max_rounds"] == 7
            assert call["checkpoint_tag"] == "test"
            assert call["model_name"] == "stub"
            assert call["rationale_mode"] == config.reward.rationale_scaling
            assert call["reward_config"] == config.reward.model_dump(mode="python")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_eval_preserves_seed_iteration_order_with_family_expansion():
    config = TrainingConfig()
    config.eval.tiers = ["easy"]
    config.eval.seeds = [10, 20]

    seen_seeds: list[int] = []
    tmp_path = _tmp_path()

    def stub_collect_batch(env, policy, curriculum, **kwargs):
        for _ in range(kwargs["num_episodes"]):
            seen_seeds.append(kwargs["seed_generator"]())
        return []

    try:
        _run_eval(
            env=object(),
            policy=object(),
            config=config,
            normalizer=RewardNormalizer(),
            checkpoint_tag="test",
            model_name="stub",
            collect_batch=stub_collect_batch,
            disaster_families=[DisasterType.fire, DisasterType.gas],
            jsonl_dir=tmp_path,
        )
        assert seen_seeds == [10, 10, 20, 20]
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
