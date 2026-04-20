from __future__ import annotations

import json

from curriculum import CurriculumController
from evacos_ma.models import DisasterType


def test_curriculum_promotes_after_enough_high_rewards() -> None:
    controller = CurriculumController()
    for seed in range(1100, 1130):
        controller.record_outcome("easy", DisasterType.fire, 0.9, seed)
    assert controller.suggest_next_tier("fire") == "medium"


def test_snapshot_round_trip_keeps_family_canonicalization_and_decisions() -> None:
    controller = CurriculumController()
    for seed in range(1100, 1130):
        controller.record_outcome("easy", DisasterType.fire, 0.9, seed)

    payload = json.loads(json.dumps(controller.snapshot()))
    restored = CurriculumController()
    restored.load_snapshot(payload)

    assert restored.snapshot() == payload
    assert restored.suggest_next_tier(DisasterType.fire) == "medium"
    assert restored.suggest_next_tier("fire") == "medium"
    assert restored.suggest_next_tier("DisasterType.fire") == "medium"
