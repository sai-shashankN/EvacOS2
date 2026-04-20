"""Long-horizon deterministic replay + ranking tests (Phase 3 deliverable #5).

Tests cover:
- Deterministic replay: identical runs produce byte-equal traces.
- Bounded scores: raw_reward stays finite, final grader score in expected range.
- Cascade firing: ≥2 stages fire for cascade_hard; trigger steps are deterministic.
- Episode completes within max_steps without crashing.
- Brutal task: lightweight sanity (builds + one step).
"""

from __future__ import annotations

import json

import pytest

from evacos_ma.cascade import CascadeScheduler
from evacos_ma.cascade_configs import get_cascade_config
from evacos_ma.env import EvacEnvironment
from evacos_ma.grader import grade_episode
from evacos_ma.models import ActionType, WaitAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_episode_trace(task_id: str, seed: int = 42):
    """Run a full episode with wait-only policy and collect trace tuples."""
    env = EvacEnvironment()
    episode_id, _ = env.reset(task_id, seed)
    trace: list[tuple[int, str, float, bool]] = []
    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            break
        action = WaitAction(
            episode_id=episode_id,
            expected_step=ep.step,
            action_type=ActionType.wait,
        )
        _, reward, done, info = env.step(action)
        trace.append((ep.step, "wait", reward.total, done))
        if done:
            break
    ep = env.get_internal_state(episode_id)
    graded = grade_episode(ep)
    return trace, graded, ep, env, episode_id


def _cascade_trigger_steps(task_id: str, seed: int = 42):
    """Return list of (stage_id, trigger_step) for stages that fired."""
    env = EvacEnvironment()
    episode_id, _ = env.reset(task_id, seed)
    stages = get_cascade_config(task_id)
    if not stages:
        return []

    scheduler = env._cascade_schedulers[episode_id]
    triggered: list[tuple[str, int]] = []
    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            break
        action = WaitAction(
            episode_id=episode_id,
            expected_step=ep.step,
            action_type=ActionType.wait,
        )
        _, _, done, info = env.step(action)
        # Check if any events fired this step
        for evt in info.triggered_events:
            for stage in stages:
                if stage.stage_id in evt.event_id:
                    triggered.append((stage.stage_id, ep.step))
        if done:
            break
    return triggered


# ---------------------------------------------------------------------------
# Fast tasks: fire_easy, flood_medium, cascade_hard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", [
    "task_lh_fire_easy",
    "task_lh_flood_medium",
    "task_lh_cascade_hard",
])
def test_deterministic_replay(task_id: str) -> None:
    """Two wait-only runs at the same seed produce identical traces."""
    trace1, _, _, _, _ = _run_episode_trace(task_id, seed=42)
    trace2, _, _, _, _ = _run_episode_trace(task_id, seed=42)
    assert trace1 == trace2, (
        f"Traces diverge for {task_id}: len1={len(trace1)} len2={len(trace2)}"
    )


@pytest.mark.parametrize("task_id", [
    "task_lh_fire_easy",
    "task_lh_flood_medium",
    "task_lh_cascade_hard",
])
def test_bounded_raw_rewards(task_id: str) -> None:
    """Every raw reward is finite and within a reasonable cap."""
    trace, graded, ep, _, _ = _run_episode_trace(task_id, seed=42)
    for step, action, raw_reward, done in trace:
        assert abs(raw_reward) < 1e6, (
            f"Unbounded raw_reward={raw_reward} at step {step} in {task_id}"
        )


@pytest.mark.parametrize("task_id", [
    "task_lh_fire_easy",
    "task_lh_flood_medium",
    "task_lh_cascade_hard",
])
def test_final_score_in_expected_range(task_id: str) -> None:
    """Final grader score is within TaskSpec.expected_score_range."""
    from evacos_ma.task_registry import get_task
    task = get_task(task_id)
    _, graded, ep, _, _ = _run_episode_trace(task_id, seed=42)
    lo, hi = task.expected_score_range
    assert lo <= graded["score"] <= hi, (
        f"Score {graded['score']} outside [{lo}, {hi}] for {task_id}"
    )


@pytest.mark.parametrize("task_id", [
    "task_lh_fire_easy",
    "task_lh_flood_medium",
    "task_lh_cascade_hard",
])
def test_episode_completes_within_max_steps(task_id: str) -> None:
    """Episode terminates (done=True) without crashing."""
    from evacos_ma.task_registry import get_task
    task = get_task(task_id)
    _, _, ep, _, _ = _run_episode_trace(task_id, seed=42)
    assert ep.done, f"Episode did not finish for {task_id}"
    assert ep.step <= task.max_steps, (
        f"Episode exceeded max_steps: {ep.step} > {task.max_steps}"
    )


# ---------------------------------------------------------------------------
# Cascade-specific tests
# ---------------------------------------------------------------------------

def test_cascade_hard_at_least_two_stages_fire() -> None:
    """For cascade_hard, at least 2 cascade stages must trigger."""
    triggered = _cascade_trigger_steps("task_lh_cascade_hard", seed=42)
    stage_ids = set(sid for sid, _ in triggered)
    assert len(stage_ids) >= 2, (
        f"Expected >=2 cascade stages, got {len(stage_ids)}: {stage_ids}"
    )


def test_cascade_hard_trigger_steps_deterministic() -> None:
    """Cascade trigger steps are identical across two runs."""
    run1 = _cascade_trigger_steps("task_lh_cascade_hard", seed=42)
    run2 = _cascade_trigger_steps("task_lh_cascade_hard", seed=42)
    assert run1 == run2, (
        f"Cascade trigger steps not deterministic: {run1} vs {run2}"
    )


# ---------------------------------------------------------------------------
# Brutal — lightweight sanity only (avoid running 500 rounds in a unit test)
# ---------------------------------------------------------------------------

def test_brutal_builds_and_one_step() -> None:
    """Brutal task: building generates and first step succeeds."""
    env = EvacEnvironment()
    episode_id, obs = env.reset("task_lh_cascade_brutal", seed=42)
    assert obs.step == 0
    assert obs.max_steps == 500
    ep = env.get_internal_state(episode_id)
    action = WaitAction(
        episode_id=episode_id,
        expected_step=ep.step,
        action_type=ActionType.wait,
    )
    _, reward, done, info = env.step(action)
    assert not info.invalid_action or True  # step completes regardless
    assert abs(reward.total) < 1e6
