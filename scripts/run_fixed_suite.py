"""CLI wrapper for the fixed-suite harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evacos_ma.models import DisasterType
from training.checkpoint import load_checkpoint
from training.policy_adapter import StubPolicy, hf_policy_factory

from evaluation.fixed_suite import run_fixed_suite


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_snapshot_from_path(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_checkpoint_snapshot(config_path: Path = Path("training/config.yaml")) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint_root = Path(str(data.get("checkpoint", {}).get("root_dir", "outputs/checkpoints")))
    bundle = load_checkpoint(checkpoint_root)
    if bundle is None:
        raise FileNotFoundError(f"No checkpoint with normalizer snapshot found under {checkpoint_root}")
    return bundle.normalizer_snapshot


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--normalizer-snapshot")
    group.add_argument("--use-latest-checkpoint-normalizer", action="store_true")
    args = parser.parse_args()

    tiers = _split_csv(args.tiers)
    seeds = [int(item) for item in _split_csv(args.seeds)]
    families = [DisasterType(item) for item in _split_csv(args.families)]

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        policy_factory = lambda: hf_policy_factory("Qwen/Qwen2.5-1.5B-Instruct", lora_adapter_path=str(checkpoint))
    else:
        policy_factory = lambda: StubPolicy(seed=0)

    normalizer_snapshot = None
    if args.normalizer_snapshot:
        normalizer_snapshot = _load_snapshot_from_path(Path(args.normalizer_snapshot))
    elif args.use_latest_checkpoint_normalizer:
        normalizer_snapshot = _load_latest_checkpoint_snapshot()

    run_fixed_suite(
        policy_factory,
        tiers=tiers,
        seeds=seeds,
        disaster_families=families,
        max_rounds=args.max_rounds,
        rationale_mode=args.rationale_mode,
        label=args.label,
        output_dir=Path(args.output_dir),
        normalizer_snapshot=normalizer_snapshot,
    )


if __name__ == "__main__":
    main()
