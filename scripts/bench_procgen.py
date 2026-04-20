from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from procgen import regenerate_until_valid


def main() -> None:
    tiers = ("easy", "medium", "hard", "brutal")
    families = ("fire", "flood", "gas", "structural", "active_threat", "multi_cascade")
    seeds = range(5)
    started = time.perf_counter()
    total_instances = 0

    for tier in tiers:
        for family in families:
            for seed in seeds:
                result = regenerate_until_valid(seed, tier, family, max_attempts=5)
                if result is None:
                    raise RuntimeError(f"benchmark failed to validate seed={seed} tier={tier} family={family}")
                total_instances += 1

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "instances": total_instances,
                "elapsed_seconds": round(elapsed, 3),
                "avg_seconds_per_instance": round(elapsed / max(total_instances, 1), 4),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
