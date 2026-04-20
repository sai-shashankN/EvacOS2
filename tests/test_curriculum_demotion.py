from __future__ import annotations

from curriculum import CurriculumController


def test_curriculum_demotes_when_current_tier_underperforms() -> None:
    controller = CurriculumController()
    controller._current_tier["fire"] = "medium"
    for seed in range(1100, 1130):
        controller.record_outcome("medium", "fire", 0.0, seed)
    assert controller.suggest_next_tier("fire") == "easy"
