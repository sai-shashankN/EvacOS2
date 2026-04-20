from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import procgen.validator as validator


def test_regenerate_until_valid_marks_original_seed_invalid_once(monkeypatch) -> None:
    tmp_path = (Path("outputs/test_tmp") / f"regenerate_{uuid.uuid4().hex}").resolve()
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    log_dir = Path("outputs/logs")

    class DummyInstance:
        pass

    monkeypatch.setattr(validator, "generate_instance", lambda seed, tier, disaster_family: DummyInstance())
    monkeypatch.setattr(
        validator,
        "validate",
        lambda instance: validator.ValidationReport(
            valid=False,
            oracle_save_rate=0.0,
            min_path_length_per_floor={},
            earliest_blockage_round=None,
            reasons=["invalid"],
        ),
    )

    result = validator.regenerate_until_valid(7, "easy", "fire", max_attempts=2)
    assert result is None
    lines = (log_dir / "invalid_seeds.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"seed": 7, "tier": "easy", "disaster_family": "fire"}
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_regenerate_until_valid_returns_first_valid_attempt(monkeypatch) -> None:
    attempts: list[int] = []

    class DummyInstance:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    monkeypatch.setattr(validator, "generate_instance", lambda seed, tier, disaster_family: DummyInstance(seed))

    def fake_validate(instance):
        attempts.append(instance.seed)
        return validator.ValidationReport(
            valid=instance.seed == 9,
            oracle_save_rate=0.75,
            min_path_length_per_floor={0: 1},
            earliest_blockage_round=None,
            reasons=[] if instance.seed == 9 else ["invalid"],
        )

    monkeypatch.setattr(validator, "validate", fake_validate)
    instance, report = validator.regenerate_until_valid(7, "easy", "fire", max_attempts=5)
    assert instance.seed == 9
    assert report.valid is True
    assert attempts == [7, 8, 9]
