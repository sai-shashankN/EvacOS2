from __future__ import annotations

from procgen._oracle import run_oracle
from tests.procgen_helpers import make_five_floor_building


def test_oracle_save_rate_drops_for_impaired_cohort_with_stairwell_only_egress() -> None:
    building = make_five_floor_building(with_elevator=False, impaired_on_top=2, exits_on_floors=(0,))
    rate = run_oracle(building, [])
    assert rate < 1.0


def test_oracle_recovers_full_save_rate_when_elevator_exists_for_impaired_cohort() -> None:
    building = make_five_floor_building(with_elevator=True, impaired_on_top=1, exits_on_floors=(0,))
    rate = run_oracle(building, [])
    assert rate == 1.0
