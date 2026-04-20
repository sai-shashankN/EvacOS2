from __future__ import annotations

from curriculum import CurriculumController
from evacos_ma.models import DisasterType


def test_eval_holdout_seeds_do_not_update_curriculum_state() -> None:
    controller = CurriculumController()
    for seed in (42, 123, 456, 789, 1024):
        controller.record_outcome("easy", DisasterType.fire, 1.0, seed)
    assert controller.snapshot()["stats"] == {}
    assert controller.suggest_next_tier("fire") == "easy"
