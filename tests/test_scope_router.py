from __future__ import annotations

from dataclasses import dataclass

from evacos_ma.models import DisasterType
from training.scope_router import GENERALIST_POLICY_KEY, route_scope


def test_routes_single_family_fire_to_fire_specialist():
    decision = route_scope({"disaster_family": "fire", "tier": "easy", "severity": "0.7"})

    assert decision.policy_key == "fire_specialist"
    assert decision.disaster_family == "fire"
    assert decision.reason == "single_family_fire"
    assert decision.tier == "easy"
    assert decision.severity == 0.7
    assert decision.uses_specialist is True


def test_routes_single_family_flood_and_gas_specialists():
    assert route_scope({"disaster_type": DisasterType.flood}).policy_key == "flood_specialist"
    assert route_scope({"disaster_type": "gas_leak"}).policy_key == "gas_specialist"


def test_routes_unknown_family_to_generalist():
    decision = route_scope({"disaster_family": "meteor"})

    assert decision.policy_key == GENERALIST_POLICY_KEY
    assert decision.disaster_family == "meteor"
    assert decision.reason == "unsupported_disaster_family"
    assert decision.uses_specialist is False


def test_routes_multi_family_or_cascade_to_generalist():
    multi = route_scope({"disaster_families": ["fire", "gas"]})
    cascade = route_scope({"disaster_family": "fire", "cascade_hint": {"next": "explosion"}})

    assert multi.policy_key == GENERALIST_POLICY_KEY
    assert multi.reason == "multi_disaster_or_cascade"
    assert cascade.policy_key == GENERALIST_POLICY_KEY
    assert cascade.reason == "multi_disaster_or_cascade"


def test_routes_object_metadata():
    @dataclass
    class Metadata:
        disaster_family: str
        tier: str

    decision = route_scope(Metadata(disaster_family="flood", tier="easy"))

    assert decision.policy_key == "flood_specialist"
    assert decision.tier == "easy"
