"""Simple heuristic rubric for override examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_belief_id(arguments: dict[str, Any]) -> str | None:
    for key in ("belief_id", "target_belief_id"):
        value = arguments.get(key)
        if value:
            return str(value)
    return None


def build_oversight_rubric(
    *,
    log_dir: Path = Path("outputs/logs"),
    output_json: Path = Path("outputs/evals/oversight_rubric.json"),
    output_markdown: Path = Path("outputs/evals/oversight_examples.md"),
    top_k: int = 3,
) -> list[dict[str, Any]]:
    action_rows = _read_jsonl(log_dir / "action_trace.jsonl")
    rationale_rows = _read_jsonl(log_dir / "rationale_audit.jsonl")
    rationale_by_action = {row.get("action_id"): row for row in rationale_rows}
    scored: list[dict[str, Any]] = []

    for row in action_rows:
        if row.get("action_type") != "override_floor_agent":
            continue
        rationale = rationale_by_action.get(row.get("action_id"), {})
        arguments = row.get("arguments", {}) or {}
        belief_id = _find_belief_id(arguments)
        delta = float(rationale.get("counterfactual_delta", 0.0))
        rationale_text = str(arguments.get("rationale", ""))
        mentions_belief = 1.0 if belief_id and belief_id in rationale_text else 0.0
        groundedness = 1.0 if belief_id else 0.0
        rationale_quality = 0.5 * mentions_belief + 0.5 * (1.0 if delta > 0 else 0.0)
        scored.append(
            {
                **row,
                "belief_id": belief_id,
                "groundedness": groundedness,
                "counterfactual_delta": delta,
                "rationale_quality": rationale_quality,
            }
        )

    scored.sort(
        key=lambda entry: (
            -float(entry.get("rationale_quality", 0.0)),
            -float(entry.get("counterfactual_delta", 0.0)),
            str(entry.get("episode_id", "")),
            int(entry.get("round_id", 0)),
        )
    )
    payload = {"top_examples": scored[:top_k], "count": len(scored)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Oversight Examples", ""]
    if not scored:
        lines.append("No override examples were found in `outputs/logs/action_trace.jsonl`.")
    else:
        for index, example in enumerate(scored[:top_k], start=1):
            lines.extend(
                [
                    f"## Example {index}",
                    f"- episode_id: `{example.get('episode_id')}`",
                    f"- round_id: `{example.get('round_id')}`",
                    f"- groundedness: `{example.get('groundedness')}`",
                    f"- counterfactual_delta: `{example.get('counterfactual_delta')}`",
                    f"- rationale_quality: `{example.get('rationale_quality')}`",
                    f"- belief_id: `{example.get('belief_id')}`",
                    "",
                ]
            )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scored
