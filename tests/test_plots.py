from pathlib import Path

from evaluation.plots import make_all_plots, make_reward_curve


def test_plot_functions_skip_gracefully_when_inputs_missing(tmp_path: Path):
    results = make_all_plots(tmp_path)
    assert results == [None, None, None, None, None, None]


def test_reward_curve_writes_png(tmp_path: Path):
    metrics_dir = Path("outputs/training")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.csv"
    original = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else None
    metrics_path.write_text(
        "\n".join(
            [
                "step,wall_seconds,tier_mix,mean_raw_reward_orch,mean_raw_reward_floor,mean_norm_reward_orch,mean_norm_reward_floor,invalid_action_rate,override_rate,override_win_rate,rationale_bonus_mean,episodes_seen",
                "1,1.0,easy,0.1,0.2,0.3,0.4,0.0,0.0,0.0,0.0,1",
                "2,2.0,easy,0.2,0.3,0.4,0.5,0.0,0.0,0.0,0.0,2",
                "3,3.0,easy,0.3,0.4,0.5,0.6,0.0,0.0,0.0,0.0,3",
            ]
        ),
        encoding="utf-8",
    )
    try:
        output = make_reward_curve(tmp_path)
        assert output is not None and output.exists()
        assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        if original is None:
            metrics_path.unlink(missing_ok=True)
        else:
            metrics_path.write_text(original, encoding="utf-8")
