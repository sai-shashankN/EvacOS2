"""Single-agent smoke test for EvacOS-MA.

Runs an easy-tier (task_1_fire_easy) episode on seed 42 using a trivial
wait-only baseline policy. Asserts:
  - Returned score is a finite float within (0, 1).
  - A repeat run with the same seed produces byte-identical observation
    and action traces (deterministic replay).
  - The full episode completes in under 10 seconds locally.
"""

from __future__ import annotations

import time

from evacos_ma.env import EvacEnvironment
from evacos_ma.grader import grade_episode
from evacos_ma.models import ActionType, WaitAction


def _normalize_episode_id(obs_json: dict) -> dict:
    """Replace episode_id with a stable placeholder for comparison."""
    obs_json["episode_id"] = "<episode>"
    return obs_json


def _run_wait_baseline(task_id: str = "task_1_fire_easy", seed: int = 42):
    """Run an episode using only wait actions. Returns (observations, rewards, episode_state)."""
    env = EvacEnvironment()
    episode_id, observation = env.reset(task_id, seed)

    observations = [_normalize_episode_id(observation.model_dump(mode="json"))]
    rewards: list[float] = []

    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            break

        observation, reward, done, info = env.step(
            WaitAction(
                episode_id=episode_id,
                expected_step=ep.step,
                action_type=ActionType.wait,
            )
        )
        observations.append(_normalize_episode_id(observation.model_dump(mode="json")))
        rewards.append(reward.total)

    ep = env.get_internal_state(episode_id)
    graded = grade_episode(ep)
    return observations, rewards, graded


def test_smoke_score_bounded_and_finite():
    """Score must be finite and bounded in (0, 1)."""
    observations, rewards, graded = _run_wait_baseline()
    score = graded["score"]
    assert isinstance(score, float), f"Score is not a float: {type(score)}"
    assert float("-inf") < score < float("inf"), f"Score is not finite: {score}"
    assert 0.0 < score < 1.0, f"Score out of bounded range (0, 1): {score}"


def test_smoke_deterministic_replay():
    """Two runs with the same seed must produce identical observation and reward traces."""
    obs_a, rewards_a, graded_a = _run_wait_baseline(seed=42)
    obs_b, rewards_b, graded_b = _run_wait_baseline(seed=42)

    assert obs_a == obs_b, "Observation traces differ between runs"
    assert rewards_a == rewards_b, "Reward traces differ between runs"
    assert graded_a["score"] == graded_b["score"], "Scores differ between runs"


def test_smoke_completes_quickly():
    """Episode must complete in under 10 seconds."""
    start = time.perf_counter()
    _run_wait_baseline()
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"Episode took {elapsed:.2f}s, exceeding 10s budget"
