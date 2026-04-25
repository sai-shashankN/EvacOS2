"""CLI wrapper for baseline-vs-trained comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evacos_ma.models import DisasterType
from training.checkpoint import load_checkpoint

from evaluation.baseline_vs_trained import run_comparison


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_latest_checkpoint_snapshot(config_path: Path = Path("training/config.yaml")) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint_root = Path(str(data.get("checkpoint", {}).get("root_dir", "outputs/checkpoints")))
    bundle = load_checkpoint(checkpoint_root)
    if bundle is None:
        raise FileNotFoundError(f"No checkpoint with normalizer snapshot found under {checkpoint_root}")
    return bundle.normalizer_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-checkpoint")
    parser.add_argument("--tiers", default="easy,medium")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--families", default="fire,flood,gas,structural,active_threat,multi_cascade")
    parser.add_argument("--rationale-mode", default="linear_capped")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Bounded rounds per eval episode for smoke/gate comparisons.",
    )
    parser.add_argument("--output-csv", default="outputs/evals/baseline_vs_trained.csv")
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--use-latest-checkpoint-normalizer", action="store_true")
    args = parser.parse_args()

    trained_normalizer_snapshot = None
    if args.use_latest_checkpoint_normalizer:
        trained_normalizer_snapshot = _load_latest_checkpoint_snapshot()

    result = run_comparison(
        trained_checkpoint=Path(args.trained_checkpoint) if args.trained_checkpoint else None,
        tiers=_split_csv(args.tiers),
        seeds=[int(item) for item in _split_csv(args.seeds)],
        disaster_families=[DisasterType(item) for item in _split_csv(args.families)],
        max_rounds=args.max_rounds,
        rationale_mode=args.rationale_mode,
        output_csv=Path(args.output_csv),
        skip_trained=args.skip_trained,
        trained_normalizer_snapshot=trained_normalizer_snapshot,
    )
    print("Baseline eval: cold normalizer (tanh fallback).")
    if result.trained_json is not None and args.use_latest_checkpoint_normalizer:
        print("Trained eval: z-scored against snapshot from training/config.yaml checkpoint.root_dir latest/.")
    elif result.trained_json is not None:
        print("Trained eval: cold normalizer (no checkpoint snapshot supplied).")


if __name__ == "__main__":
    main()
