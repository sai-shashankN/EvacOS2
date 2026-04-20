from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.slow
def test_procgen_benchmark_script_finishes_under_90_seconds() -> None:
    script = Path("scripts/bench_procgen.py")
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    assert "elapsed_seconds" in result.stdout
    assert elapsed < 90
