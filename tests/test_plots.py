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
                "step,wall_seconds,tier_mix,mean_raw_reward_orch,mean_raw_reward_floor,mean_norm_reward_orch,mean_norm_reward_floor,invalid_action_rate,override_rate,override_win_rate,rationale_bonus_mean,episodes_seen,loss,policy_loss,kl_loss,ratio_mean,ratio_std,clip_fraction,kl_max,mask_coverage,mean_advantage,advantage_std",
                "1,1.0,easy,0.1,0.2,0.3,0.4,0.0,0.0,0.0,0.0,1,1.0,0.8,0.2,1.0,0.0,0.0,0.2,0.5,0.0,1.0",
                "2,2.0,easy,0.2,0.3,0.4,0.5,0.0,0.1,1.0,0.2,2,0.9,0.7,0.2,1.0,0.0,0.0,0.2,0.5,0.1,0.9",
                "3,3.0,easy,0.3,0.4,0.5,0.6,0.0,0.2,0.5,0.3,3,0.8,0.6,0.2,1.0,0.0,0.0,0.2,0.5,0.2,0.8",
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


def test_reward_curve_accepts_custom_metrics_path(tmp_path: Path):
    metrics_path = tmp_path / "split-metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "step,wall_seconds,tier_mix,mean_raw_reward_orch,mean_raw_reward_floor,mean_norm_reward_orch,mean_norm_reward_floor,invalid_action_rate,override_rate,override_win_rate,rationale_bonus_mean,episodes_seen,loss,policy_loss,kl_loss,ratio_mean,ratio_std,clip_fraction,kl_max,mask_coverage,mean_advantage,advantage_std",
                "1,1.0,easy,0.1,0.2,0.3,0.4,0.0,0.0,0.0,0.0,1,1.0,0.8,0.2,1.0,0.0,0.0,0.2,0.5,0.0,1.0",
                "2,2.0,easy,0.2,0.3,0.4,0.5,0.0,0.1,1.0,0.2,2,0.9,0.7,0.2,1.0,0.0,0.0,0.2,0.5,0.1,0.9",
            ]
        ),
        encoding="utf-8",
    )

    output = make_reward_curve(tmp_path, metrics_path=metrics_path)

    assert output is not None and output.exists()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
