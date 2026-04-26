from __future__ import annotations

from pathlib import Path

from scripts.check_grpo_contrast import check_csv


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_check_grpo_contrast_accepts_nonzero_final_window(tmp_path: Path):
    csv_path = _write_csv(
        tmp_path / "metrics.csv",
        "\n".join(
            [
                "step,floor_agent_group_raw_reward_std_mean,floor_agent_advantage_std",
                "0,0.0,0.0",
                "1,0.3,0.4",
                "2,0.2,0.5",
            ]
        ),
    )

    assert check_csv(csv_path, last_n=2, min_mean=1e-8) == []


def test_check_grpo_contrast_rejects_missing_columns(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "metrics.csv", "step,invalid_action_rate\n0,0.1\n")

    assert "missing required contrast columns" in check_csv(csv_path)[0]


def test_check_grpo_contrast_rejects_flat_final_window(tmp_path: Path):
    csv_path = _write_csv(
        tmp_path / "metrics.csv",
        "\n".join(
            [
                "step,floor_agent_group_raw_reward_std_mean,floor_agent_advantage_std",
                "0,0.2,0.2",
                "1,0.0,0.0",
                "2,0.0,0.0",
            ]
        ),
    )

    errors = check_csv(csv_path, last_n=2, min_mean=1e-8)

    assert any("floor_agent_group_raw_reward_std_mean" in error for error in errors)
    assert any("floor_agent_advantage_std" in error for error in errors)
