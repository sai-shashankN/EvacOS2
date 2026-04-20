"""Eval ranking preservation tests (Phase 3 deliverable #5).

Runs 10 episodes (mix of fire_easy and flood_medium across multiple seeds)
with two trivial policies. Verifies that normalization preserves ordering:
for any pair at the same tier, raw_a < raw_b => normalized_a <= normalized_b.

Uses the relaxed ordering check (not strict index equality) because clipping
to [-1, 1] can map different raw values to the same bound.
"""

from __future__ import annotations

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import ActionType, EvacuateFloorAction, WaitAction
from evacos_ma.reward_pipeline import RewardPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1024]
TASKS_MIX = ["task_lh_fire_easy", "task_lh_flood_medium"]


def _run_episode_raw(policy: str, task_id: str, seed: int) -> dict:
    """Run one episode; return raw_total and tier."""
    env = EvacEnvironment()
    episode_id, _ = env.reset(task_id, seed)

    raw_total = 0.0

    while True:
        ep = env.get_internal_state(episode_id)
        if ep.done:
            break

        if policy == "wait":
            action = WaitAction(
                episode_id=episode_id,
                expected_step=ep.step,
                action_type=ActionType.wait,
            )
        else:
            # Policy B: alternate between wait and evacuate_floor
            if ep.step % 2 == 0:
                action = WaitAction(
                    episode_id=episode_id,
                    expected_step=ep.step,
                    action_type=ActionType.wait,
                )
            else:
                best_floor = 0
                best_count = 0
                for floor in ep.building.floors:
                    count = sum(r.occupancy.total for r in floor.rooms)
                    if count > best_count:
                        best_count = count
                        best_floor = floor.floor_id
                action = EvacuateFloorAction(
                    episode_id=episode_id,
                    expected_step=ep.step,
                    action_type=ActionType.evacuate_floor,
                    floor_id=best_floor,
                )

        _, reward, done, info = env.step(action)
        raw_total += reward.total
        if done:
            break

    from evacos_ma.task_registry import get_task
    tier = get_task(task_id).difficulty

    return {
        "policy": policy,
        "task_id": task_id,
        "seed": seed,
        "raw_total": raw_total,
        "tier": tier,
    }


def _check_pairwise_ordering(episodes: list[dict], pipeline: RewardPipeline) -> None:
    """Check that for all same-tier pairs, raw_a < raw_b => norm_a <= norm_b."""
    tiers: dict[str, list[int]] = {}
    for idx, ep in enumerate(episodes):
        tiers.setdefault(ep["tier"], []).append(idx)

    for tier, indices in tiers.items():
        raws = [episodes[i]["raw_total"] for i in indices]
        norms = [pipeline.normalize(r, tier) for r in raws]
        for j in range(len(raws)):
            for k in range(j + 1, len(raws)):
                if raws[j] < raws[k]:
                    assert norms[j] <= norms[k], (
                        f"Ordering violated in tier '{tier}': "
                        f"raw[{j}]={raws[j]} < raw[{k}]={raws[k]} but "
                        f"norm[{j}]={norms[j]} > norm[{k}]={norms[k]}"
                    )
                elif raws[k] < raws[j]:
                    assert norms[k] <= norms[j], (
                        f"Ordering violated in tier '{tier}': "
                        f"raw[{k}]={raws[k]} < raw[{j}]={raws[j]} but "
                        f"norm[{k}]={norms[k]} > norm[{j}]={norms[j]}"
                    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ranking_preserved_mixed_tasks() -> None:
    """10 episodes (mix of easy+medium): pairwise ordering preserved per tier."""
    episodes: list[dict] = []
    for i, seed in enumerate(SEEDS):
        task_id = TASKS_MIX[i % len(TASKS_MIX)]
        episodes.append(_run_episode_raw("wait", task_id, seed))
        episodes.append(_run_episode_raw("alternate", task_id, seed))

    assert len(episodes) == 10

    pipeline = RewardPipeline()
    for ep in episodes:
        pipeline.observe(ep["raw_total"], ep["tier"])

    _check_pairwise_ordering(episodes, pipeline)


def test_ranking_preserved_single_tier_easy() -> None:
    """All-easy episodes: pairwise ordering preserved."""
    episodes: list[dict] = []
    for seed in SEEDS:
        episodes.append(_run_episode_raw("wait", "task_lh_fire_easy", seed))
        episodes.append(_run_episode_raw("alternate", "task_lh_fire_easy", seed))

    pipeline = RewardPipeline()
    for ep in episodes:
        pipeline.observe(ep["raw_total"], ep["tier"])

    _check_pairwise_ordering(episodes, pipeline)


def test_ranking_preserved_single_tier_medium() -> None:
    """All-medium episodes: pairwise ordering preserved."""
    episodes: list[dict] = []
    for seed in SEEDS:
        episodes.append(_run_episode_raw("wait", "task_lh_flood_medium", seed))
        episodes.append(_run_episode_raw("alternate", "task_lh_flood_medium", seed))

    pipeline = RewardPipeline()
    for ep in episodes:
        pipeline.observe(ep["raw_total"], ep["tier"])

    _check_pairwise_ordering(episodes, pipeline)
