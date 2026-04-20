"""Tests for procedural generator tier scaling."""

import pytest

from evacos_ma.models import DisasterType
from procgen import generate_instance


class TestGeneratorTierScaling:
    """Tier knobs produce monotone differences across tiers."""

    def test_easy_more_exits_than_brutal(self):
        """Easy has strictly more exits than brutal (aggregate over 10 seeds)."""
        easy_exits = 0
        brutal_exits = 0
        for seed in range(10):
            easy = generate_instance(seed, "easy", DisasterType.fire)
            brutal = generate_instance(seed, "brutal", DisasterType.fire)
            easy_exits += sum(len(f.exits) for f in easy.building.floors)
            brutal_exits += sum(len(f.exits) for f in brutal.building.floors)
        # Easy should have more exits total (3 per building vs 1 per building)
        assert easy_exits > brutal_exits

    def test_brutal_more_hazards_than_easy(self):
        """Brutal has strictly more scheduled hazard events than easy."""
        easy_events = 0
        brutal_events = 0
        for seed in range(10):
            easy = generate_instance(seed, "easy", DisasterType.fire)
            brutal = generate_instance(seed, "brutal", DisasterType.fire)
            easy_events += len(easy.scheduled_events)
            brutal_events += len(brutal.scheduled_events)
        # Brutal cascade_aggression=1.0 -> more waves than easy cascade_aggression=0.25
        assert brutal_events > easy_events

    def test_rooms_per_floor_monotone_decreasing(self):
        """Mean rooms_per_floor is monotone decreasing easy -> brutal over 20 seeds."""
        tier_means = {}
        for tier in ["easy", "medium", "hard", "brutal"]:
            total_rooms = 0
            for seed in range(20):
                inst = generate_instance(seed, tier, DisasterType.fire)
                total_rooms += sum(len(f.rooms) for f in inst.building.floors)
            tier_means[tier] = total_rooms / 20 / 5  # per floor

        # Monotone decreasing
        assert tier_means["easy"] >= tier_means["medium"]
        assert tier_means["medium"] >= tier_means["hard"]
        assert tier_means["hard"] >= tier_means["brutal"]
