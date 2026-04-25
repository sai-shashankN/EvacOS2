from __future__ import annotations

from evacos_ma.models import DisasterType
from scripts.run_oracle_canary import run_oracle_canary


def test_easy_fire_oracle_canary_saves_civilians() -> None:
    summary = run_oracle_canary(
        seeds=[42],
        max_rounds=8,
        task_id="procgen_easy_fire",
        tier="easy",
        disaster_family=DisasterType.fire,
    )

    assert summary["pass"] is True
    assert summary["total_saved"] > 0
    assert summary["save_rate"] > 0.0
