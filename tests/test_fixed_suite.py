import math
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from training.policy_adapter import StubPolicy

from evaluation.fixed_suite import (
    EpisodeResult,
    FixedSuiteResult,
    _count_actions,
    _seed_eval_normalizer,
    run_fixed_suite,
)


def _tmp_dir() -> Path:
    path = Path(".phase19_test_tmp") / f"fixed_suite_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_fixed_suite_returns_single_episode():
    tmp_dir = _tmp_dir()
    try:
        result = run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=50,
            label="test",
            output_dir=tmp_dir,
        )
        assert isinstance(result, FixedSuiteResult)
        assert len(result.episodes) == 1
        assert result.env_side_rationale_wired is True
        assert result.episodes[0].scope_policy_key == "fire_specialist"
        assert result.episodes[0].scope_route_reason == "single_family_fire"
        assert 0.0 <= result.episodes[0].eval_score_pct <= 100.0
        assert 0.0 <= result.aggregate.eval_score_pct.mean <= 100.0
        assert result.episodes[0].total_civilians >= result.episodes[0].civilians_saved
        assert result.episodes[0].civilians_remaining >= 0
        assert result.episodes[0].action_type_counts
        assert "floor_agent" in result.episodes[0].action_type_counts_by_role
        assert 0.0 <= result.episodes[0].wait_rate <= 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_action_diagnostics_count_current_and_legacy_action_names():
    from collections import Counter

    counts = Counter(
        {
            "scout": 2,
            "scout_status": 1,
            "evacuate_floor_priority": 3,
            "evacuate_room": 1,
        }
    )

    assert _count_actions(counts, ("scout", "scout_status")) == 3
    assert _count_actions(
        counts,
        ("evacuate_floor_priority", "evacuate_floor", "evacuate_room"),
    ) == 4


def test_fixed_suite_json_round_trip():
    tmp_dir = _tmp_dir()
    try:
        result = run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=50,
            label="roundtrip",
            output_dir=tmp_dir,
        )
        payload = (tmp_dir / "fixed_suite_roundtrip_linear_capped.json").read_text(encoding="utf-8")
        restored = FixedSuiteResult.model_validate_json(payload)
        assert restored == result
        assert "eval_score_pct" in payload
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_suite_eval_does_not_mutate_normalizer(monkeypatch):
    update_calls = []

    def fail_update(self, role, tier, raw):
        update_calls.append((role, tier, raw))
        raise AssertionError("RewardNormalizer.update should not be called in eval mode")

    monkeypatch.setattr("training.reward.RewardNormalizer.update", fail_update)
    tmp_dir = _tmp_dir()
    try:
        run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=20,
            output_dir=tmp_dir,
        )
        assert update_calls == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_suite_cold_normalizer_marks_z_scored_false():
    tmp_dir = _tmp_dir()
    try:
        result = run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=1,
            output_dir=tmp_dir,
        )
        assert result.normalizer_z_scored is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_suite_warm_normalizer_marks_z_scored_true():
    snapshot = {
        "orchestrator:easy": {"count": 100, "mean": 0.5, "m2": 25.0},
        "floor_agent:easy": {"count": 100, "mean": 0.5, "m2": 25.0},
    }
    tmp_dir = _tmp_dir()
    try:
        result = run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=1,
            output_dir=tmp_dir,
            normalizer_snapshot=snapshot,
        )
        assert result.normalizer_z_scored is True
        episode = result.episodes[0]
        assert not math.isclose(
            episode.normalized_reward_by_role["orchestrator"],
            math.tanh(episode.raw_reward_by_role["orchestrator"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_seed_eval_normalizer_round_trips_snapshot():
    snapshot = {
        "orchestrator:easy": {"count": 100, "mean": 0.5, "m2": 25.0},
        "floor_agent:easy": {"count": 100, "mean": 0.5, "m2": 25.0},
    }
    normalizer = _seed_eval_normalizer("easy", snapshot=snapshot)
    assert normalizer.snapshot() == snapshot


def test_fixed_suite_traces_honor_output_dir(monkeypatch):
    recorded: dict[str, Path] = {}
    tmp_dir = _tmp_dir()

    def fake_collect_batch(*args, **kwargs):
        del args
        jsonl_dir = kwargs["jsonl_dir"]
        recorded["jsonl_dir"] = jsonl_dir
        recorded["reward_config"] = kwargs["reward_config"]
        (jsonl_dir / "round_trace.jsonl").write_text(
            '{"episode_id":"ep1","round_id":0}\n',
            encoding="utf-8",
        )
        return [
            SimpleNamespace(
                episode_id="ep1",
                num_rounds=1,
                wall_clock_seconds=0.1,
                done_reason="done",
                total_raw_reward_by_role={"orchestrator": 1.0, "floor_0_agent": 0.5},
                total_normalized_reward_by_role={"orchestrator": 0.9, "floor_0_agent": 0.4},
            )
        ]

    def fake_build_episode_result(**kwargs):
        assert kwargs["log_dir"] == tmp_dir / "logs"
        return EpisodeResult(
            episode_id="episode-result",
            label=kwargs["label"],
            tier=kwargs["tier"],
            seed=kwargs["seed"],
            disaster_family=kwargs["family"],
            rationale_mode=kwargs["rationale_mode"],
            checkpoint_tag=kwargs["label"],
            model_name="StubPolicy",
            trace_schema_version="v1",
            generator_config_hash="",
            reward_schema_version="v1",
            prompt_template_version="2026.04.20",
            total_rounds=1,
            wall_clock_s=0.1,
        )

    monkeypatch.setattr("evaluation.fixed_suite.collect_batch", fake_collect_batch)
    monkeypatch.setattr("evaluation.fixed_suite._build_episode_result", fake_build_episode_result)

    try:
        result = run_fixed_suite(
            lambda: StubPolicy(seed=0),
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire",),
            max_rounds=5,
            label="trace_dir",
            output_dir=tmp_dir,
        )

        assert result.episodes
        assert result.env_side_rationale_wired is True
        assert recorded["jsonl_dir"] == tmp_dir / "logs"
        assert recorded["reward_config"]["rationale_scaling"] == "linear_capped"
        assert (tmp_dir / "logs").exists()
        assert any((tmp_dir / "logs" / name).exists() for name in ("round_trace.jsonl", "action_trace.jsonl", "episode_summary.jsonl"))
        assert recorded["jsonl_dir"] != Path("outputs/logs")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fixed_suite_can_use_scope_aware_policy_factory():
    class ScopeAwareFactory:
        def __init__(self) -> None:
            self.decisions = []

        def for_scope(self, decision):
            self.decisions.append(decision)
            return StubPolicy(seed=0)

    factory = ScopeAwareFactory()
    tmp_dir = _tmp_dir()
    try:
        result = run_fixed_suite(
            factory,
            tiers=("easy",),
            seeds=(42,),
            disaster_families=("fire", "gas"),
            max_rounds=1,
            label="scope",
            output_dir=tmp_dir,
        )

        assert [decision.policy_key for decision in factory.decisions] == [
            "fire_specialist",
            "gas_specialist",
        ]
        assert [episode.scope_policy_key for episode in result.episodes] == [
            "fire_specialist",
            "gas_specialist",
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
