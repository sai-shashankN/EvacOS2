import re
from collections import defaultdict

from evacos_ma.env import EvacEnvironment
from evacos_ma.models import DisasterType
from training.policy_adapter import StubPolicy
from training.rollout import collect_episode


def test_floor_groups_are_non_singleton_under_default_procgen_layout():
    """Guard for logs/review_master_FINAL.md:583-588: floor rollout groups must stay non-singleton."""
    env = EvacEnvironment()
    result = collect_episode(
        env,
        StubPolicy(seed=42),
        seed=42,
        tier="easy",
        disaster_family=DisasterType.fire,
        max_rounds=3,
    )

    grouped_samples: dict[str, list] = defaultdict(list)
    for sample in result.samples:
        grouped_samples[sample.group_id].append(sample)

    floor_groups = {
        group_id: group
        for group_id, group in grouped_samples.items()
        if re.fullmatch(r"ep_.+_r_\d+_floor", group_id)
    }

    assert floor_groups, "Expected at least one floor group matching ep_*_r_*_floor."
    observed_sizes = {group_id: len(group) for group_id, group in floor_groups.items()}
    assert all(size >= 2 for size in observed_sizes.values()), (
        "Floor groups regressed to singleton size; "
        f"observed sizes were {observed_sizes!r}"
    )
