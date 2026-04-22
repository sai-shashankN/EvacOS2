from unittest.mock import MagicMock

import pytest

from curriculum.controller import CurriculumController
from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from training.policy_adapter import StubPolicy
from training.rollout import collect_batch


def _fixed_curriculum() -> CurriculumController:
    curriculum = CurriculumController()
    curriculum.suggest_next_tier = lambda disaster_family: "easy"  # type: ignore[method-assign]
    return curriculum


def test_collect_batch_default_path_calls_cleanup_episode():
    env = EvacEnvironment()
    original_cleanup = env.cleanup_episode
    cleanup_spy = MagicMock(wraps=original_cleanup)
    env.cleanup_episode = cleanup_spy  # type: ignore[method-assign]

    results = collect_batch(
        env,
        StubPolicy(seed=42),
        _fixed_curriculum(),
        num_episodes=1,
        seed_generator=lambda: 42,
        disaster_families=(DisasterType.fire,),
        max_rounds=3,
        is_eval=True,
    )

    assert len(results) == 1
    episode_id = results[0].episode_id
    cleanup_spy.assert_called_once_with(episode_id)
    assert episode_id not in env._episodes
    with pytest.raises(ValueError, match="Unknown episode_id"):
        env.get_internal_state(episode_id)


def test_collect_batch_cleanup_can_be_disabled():
    env = EvacEnvironment()
    original_cleanup = env.cleanup_episode
    cleanup_spy = MagicMock(wraps=original_cleanup)
    env.cleanup_episode = cleanup_spy  # type: ignore[method-assign]

    results = collect_batch(
        env,
        StubPolicy(seed=42),
        _fixed_curriculum(),
        num_episodes=1,
        seed_generator=lambda: 42,
        disaster_families=(DisasterType.fire,),
        max_rounds=3,
        is_eval=True,
        cleanup_env_episodes=False,
    )

    assert len(results) == 1
    episode_id = results[0].episode_id
    cleanup_spy.assert_not_called()
    assert episode_id in env._episodes
    assert env.get_internal_state(episode_id).episode_id == episode_id
