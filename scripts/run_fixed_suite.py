"""CLI wrapper for the fixed-suite harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evacos_ma.models import DisasterType
from training.policy_adapter import StubPolicy, hf_policy_factory

from evaluation.fixed_suite import run_fixed_suite


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="easy,medium")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--families", default="fire,flood,gas,structural,active_threat,multi_cascade")
    parser.add_argument("--rationale-mode", default="linear_capped")
    parser.add_argument("--label", default="trained")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default="outputs/evals")
    parser.add_argument("--max-rounds", type=int, default=500)
    args = parser.parse_args()

    tiers = _split_csv(args.tiers)
    seeds = [int(item) for item in _split_csv(args.seeds)]
    families = [DisasterType(item) for item in _split_csv(args.families)]

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        policy_factory = lambda: hf_policy_factory("Qwen/Qwen2.5-1.5B-Instruct", lora_adapter_path=str(checkpoint))
    else:
        policy_factory = lambda: StubPolicy(seed=0)

    run_fixed_suite(
        policy_factory,
        tiers=tiers,
        seeds=seeds,
        disaster_families=families,
        max_rounds=args.max_rounds,
        rationale_mode=args.rationale_mode,
        label=args.label,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
