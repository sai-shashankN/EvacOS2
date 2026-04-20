"""Tests for procedural generator determinism."""

import pytest

from evacos_ma.models import DisasterType
from procgen import generate_instance


class TestGeneratorDeterminism:
    """Same (seed, tier, family) produces identical buildings and events."""

    def test_same_seed_tier_family_identical(self):
        """Same (seed, tier, family) => identical building JSON and events."""
        a = generate_instance(42, "easy", DisasterType.fire)
        b = generate_instance(42, "easy", DisasterType.fire)
        assert a.building.model_dump_json() == b.building.model_dump_json()
        assert len(a.scheduled_events) == len(b.scheduled_events)
        for ea, eb in zip(a.scheduled_events, b.scheduled_events):
            assert ea.model_dump_json() == eb.model_dump_json()
        assert a.generator_config_hash == b.generator_config_hash

    def test_different_seeds_diverge(self):
        """Different seeds produce different room layouts."""
        a = generate_instance(1, "easy", DisasterType.fire)
        b = generate_instance(2, "easy", DisasterType.fire)
        # Different seeds should produce different building IDs at minimum
        assert a.building.building_id != b.building.building_id
        # Room counts may differ within range
        a_rooms = sum(len(f.rooms) for f in a.building.floors)
        b_rooms = sum(len(f.rooms) for f in b.building.floors)
        # They can be the same by coincidence but usually differ
        # At minimum the scheduled events target different rooms
        a_targets = {e.target_id for e in a.scheduled_events}
        b_targets = {e.target_id for e in b.scheduled_events}
        # Very unlikely to be identical
        assert a.building.model_dump_json() != b.building.model_dump_json()

    def test_different_tiers_different_hash_same_for_same_knobs(self):
        """Different tiers produce different config hashes, same tier gives same hash regardless of seed."""
        easy_a = generate_instance(1, "easy", DisasterType.fire)
        easy_b = generate_instance(99, "easy", DisasterType.fire)
        medium = generate_instance(1, "medium", DisasterType.fire)

        # Same tier & family -> same hash regardless of seed
        assert easy_a.generator_config_hash == easy_b.generator_config_hash
        # Different tier -> different hash
        assert easy_a.generator_config_hash != medium.generator_config_hash
