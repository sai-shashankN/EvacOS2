"""Opinionated evaluation profiles for EvacOS2 specialist and orchestrator runs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.demo_bundle import DemoBundleResult, build_demo_bundle
from training.checkpoint import load_checkpoint


DEFAULT_TIERS = "easy,medium,hard,brutal"
DEFAULT_SEEDS = "42,123,456,789,1024"


@dataclass(frozen=True)
class EvalProfile:
    """Named eval surface with safe defaults for one model lane."""

    name: str
    description: str
    families: tuple[str, ...]
    output_dir: Path
    config_path: Path


PROFILES: dict[str, EvalProfile] = {
    "3b-fire": EvalProfile(
        name="3b-fire",
        description="Evaluate the 3B fire floor specialist.",
        families=("fire",),
        output_dir=Path("outputs/evals/3b-fire-specialist"),
        config_path=Path("training/config.remote-unsloth-3b-fire-floor-specialist.yaml"),
    ),
    "3b-flood": EvalProfile(
        name="3b-flood",
        description="Evaluate the 3B flood floor specialist.",
        families=("flood",),
        output_dir=Path("outputs/evals/3b-flood-specialist"),
        config_path=Path("training/config.remote-unsloth-3b-flood-floor-specialist.yaml"),
    ),
    "3b-gas": EvalProfile(
        name="3b-gas",
        description="Evaluate the 3B gas floor specialist.",
        families=("gas",),
        output_dir=Path("outputs/evals/3b-gas-specialist"),
        config_path=Path("training/config.remote-unsloth-3b-gas-floor-specialist.yaml"),
    ),
    "7b-orchestrator": EvalProfile(
        name="7b-orchestrator",
        description="Evaluate the shared 7B orchestrator over routed frozen 3B specialists.",
        families=("fire", "flood", "gas"),
        output_dir=Path("outputs/evals/7b-orchestrator-routed-specialists"),
        config_path=Path("training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml"),
    ),
}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_latest_checkpoint_snapshot(config_path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runner image
        raise ImportError(
            "--use-latest-checkpoint-normalizer requires PyYAML to read the config."
        ) from exc

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    checkpoint_root = Path(
        str(data.get("checkpoint", {}).get("root_dir", "outputs/checkpoints"))
    )
    bundle = load_checkpoint(checkpoint_root)
    if bundle is None:
        raise FileNotFoundError(
            f"No checkpoint with normalizer snapshot found under {checkpoint_root}"
        )
    return bundle.normalizer_snapshot


def build_parser(profile: EvalProfile) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument(
        "--trained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the trained checkpoint directory or its lora_adapter root. "
            "Use --skip-trained for a baseline-only smoke bundle."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=profile.output_dir,
        help="Where to write JSON, CSV, scorecard, and plots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=profile.config_path,
        help="Training config used if checkpoint metadata is unavailable.",
    )
    parser.add_argument(
        "--tiers",
        default=DEFAULT_TIERS,
        help="Comma-separated difficulty tiers to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Comma-separated held-out eval seeds.",
    )
    parser.add_argument(
        "--families",
        default=",".join(profile.families),
        help="Comma-separated disaster families. Defaults to the profile lane.",
    )
    parser.add_argument("--rationale-mode", default="linear_capped")
    parser.add_argument("--training-metrics-path", type=Path, default=None)
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument(
        "--use-latest-checkpoint-normalizer",
        action="store_true",
        help="Seed trained eval normalization from config.checkpoint.root_dir/latest.",
    )
    return parser


def run_profile(profile_name: str, argv: Sequence[str] | None = None) -> DemoBundleResult:
    profile = PROFILES[profile_name]
    parser = build_parser(profile)
    args = parser.parse_args(argv)

    trained_normalizer_snapshot = None
    if args.use_latest_checkpoint_normalizer:
        trained_normalizer_snapshot = _load_latest_checkpoint_snapshot(args.config)

    result = build_demo_bundle(
        trained_checkpoint=args.trained_checkpoint,
        tiers=tuple(_split_csv(args.tiers)),
        seeds=tuple(int(item) for item in _split_csv(args.seeds)),
        disaster_families=tuple(_split_csv(args.families)),
        rationale_mode=args.rationale_mode,
        output_dir=args.output_dir,
        skip_trained=args.skip_trained,
        training_metrics_path=args.training_metrics_path,
        trained_normalizer_snapshot=trained_normalizer_snapshot,
        config_path=args.config,
    )
    _print_result(profile, result)
    return result


def _print_result(profile: EvalProfile, result: DemoBundleResult) -> None:
    print(f"Eval profile: {profile.name}")
    print(f"Output dir: {result.output_dir}")
    print(f"Scorecard: {result.scorecard_md}")
    print(f"Summary: {result.summary_md}")
    print(f"Comparison CSV: {result.comparison_csv}")
    if result.trained_json is not None:
        print(f"Trained fixed suite: {result.trained_json}")
    else:
        print("Trained fixed suite: skipped")
    for plot_path in result.plot_paths:
        print(f"Plot: {plot_path}")

