"""Evaluate the 3B gas floor specialist."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_profiles import run_profile


if __name__ == "__main__":
    run_profile("3b-gas")
