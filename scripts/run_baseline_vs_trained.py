"""CLI wrapper for baseline-vs-trained comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evacos_ma.models import DisasterType

from evaluation.baseline_vs_trained import run_comparison


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-checkpoint")
    parser.add_argument("--tiers", default="easy,medium")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--families", default="fire,flood,gas,structural,active_threat,multi_cascade")
    parser.add_argument("--rationale-mode", default="linear_capped")
    parser.add_argument("--output-csv", default="outputs/evals/baseline_vs_trained.csv")
    parser.add_argument("--skip-trained", action="store_true")
    args = parser.parse_args()

    run_comparison(
        trained_checkpoint=Path(args.trained_checkpoint) if args.trained_checkpoint else None,
        tiers=_split_csv(args.tiers),
        seeds=[int(item) for item in _split_csv(args.seeds)],
        disaster_families=[DisasterType(item) for item in _split_csv(args.families)],
        rationale_mode=args.rationale_mode,
        output_csv=Path(args.output_csv),
        skip_trained=args.skip_trained,
    )


if __name__ == "__main__":
    main()
