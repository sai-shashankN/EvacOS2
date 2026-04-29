from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_COLUMNS = (
    "floor_agent_group_raw_reward_std_mean",
    "floor_agent_advantage_std",
)


def _parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def check_csv(
    path: Path,
    *,
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    last_n: int = 10,
    min_mean: float = 1e-8,
) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            return [f"{path}: missing required contrast columns: {', '.join(missing)}"]
        rows = list(reader)

    if not rows:
        return [f"{path}: metrics CSV has no data rows"]

    window = rows[-last_n:] if last_n > 0 else rows
    errors: list[str] = []
    for column in columns:
        values = [
            parsed
            for row in window
            if (parsed := _parse_float(row.get(column, ""))) is not None
        ]
        if not values:
            errors.append(f"{path}: {column} has no numeric values in final {len(window)} rows")
            continue
        mean_value = sum(values) / len(values)
        if mean_value <= min_mean:
            errors.append(
                f"{path}: {column} final-window mean {mean_value:.6g} <= {min_mean:.6g}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if GRPO contrast metrics are missing or flat in metrics CSV files."
    )
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help=(
            "Metric columns that must have non-zero final-window means. "
            "Defaults to floor-agent GRPO columns; pass orchestrator columns "
            "for orchestrator-only runs with frozen floor specialists."
        ),
    )
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--min-mean", type=float, default=1e-8)
    args = parser.parse_args()

    all_errors: list[str] = []
    for csv_path in args.csv_paths:
        if not csv_path.exists():
            all_errors.append(f"{csv_path}: file does not exist")
            continue
        all_errors.extend(
            check_csv(
                csv_path,
                columns=tuple(args.columns),
                last_n=args.last_n,
                min_mean=args.min_mean,
            )
        )

    if all_errors:
        for error in all_errors:
            print(error)
        return 1

    print(f"GRPO contrast check passed for {len(args.csv_paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
