from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_fire_bootstrap_packages_artifacts_after_timeout_exit():
    script = (ROOT / "remote_fire_bootstrap.sh").read_text(encoding="utf-8")

    timeout_index = script.index("timeout 3600s python -m training.train")
    report_index = script.index("fire_3b_hour_report.json")
    tar_index = script.index("tar -czf /root/evacos2_fire_3b_artifacts.tgz")

    assert "set +e\ntimeout 3600s python -m training.train" in script
    assert "TRAIN_EXIT=$?\nset -e" in script
    assert timeout_index < report_index < tar_index
    assert 'exit "$TRAIN_EXIT"' in script


def test_specialist_queue_waits_on_broad_fire_training_patterns():
    script = (ROOT / "remote_specialist_queue.sh").read_text(encoding="utf-8")

    assert "fire_training_active()" in script
    assert 'pgrep -f "remote_fire_unsloth_train_call.sh"' in script
    assert 'pgrep -f "remote_fire_train_call.sh"' in script
    assert 'pgrep -f "config.remote-unsloth-3b-fire-floor-specialist-750.yaml"' in script
    assert 'pgrep -f "config.remote-unsloth-3b-fire-floor-specialist.yaml"' in script
    assert 'pgrep -f "config.fire-hour.yaml"' in script
    assert "while fire_training_active; do" in script


def test_specialist_queue_uses_750_step_configs_for_real_runs():
    script = (ROOT / "remote_specialist_queue.sh").read_text(encoding="utf-8")

    assert 'name="remote-unsloth-3b-${family}-floor-specialist-750"' in script
    assert "config.remote-unsloth-3b-${family}-floor-specialist-750.yaml" in script
    assert "${family}_3b_750step_report.json" in script
    assert "set_max_steps \"$config\" 100" not in script
    assert "starting $family specialist 750-step run" in script


def test_fire_hour_supervisor_targets_750_step_real_run_artifacts():
    script = (ROOT / "remote_fire_hour_supervisor.sh").read_text(encoding="utf-8")

    assert "config.remote-unsloth-3b-fire-floor-specialist-750.yaml" in script
    assert "remote-unsloth-3b-fire-floor-specialist-750-metrics.csv" in script
    assert "remote-unsloth-3b-fire-floor-specialist-750" in script
    assert "fire_3b_750step_report.json" in script
    assert 'TARGET_STEPS="${TARGET_STEPS:-750}"' in script
    assert 'TARGET_SECONDS="${TARGET_SECONDS:-21600}"' in script
    assert 'pgrep -f "remote_fire_unsloth_train_call.sh"' in script
    assert "timeout 3300s bash /root/remote_fire_unsloth_train_call.sh" not in script
    assert "remote-unsloth-3b-fire-floor-specialist-metrics.csv" not in script
    assert "fire_3b_hour_report.json" not in script


def test_fire_easy_proof_script_has_self_guarded_artifacts():
    script = (ROOT / "scripts" / "remote_fire_easy_proof300_train_only.sh").read_text(
        encoding="utf-8"
    )

    assert "config.remote-unsloth-3b-fire-floor-specialist-easy-proof-300.yaml" in script
    assert "remote-unsloth-3b-fire-floor-specialist-easy-proof-300-metrics.csv" in script
    assert "fire_easy_proof300_watchdog.jsonl" in script
    assert "EVACOS_STALE_AFTER_SECONDS" in script
    assert "EVACOS_MAX_WALL_SECONDS" in script
    assert "EVACOS_TRAIN_TIMEOUT_SECONDS" in script
    assert "EVACOS_MAX_INVALID_RATE" in script
    assert 'timeout "${TRAIN_TIMEOUT_SECONDS}s"' in script
    assert "metrics_stale>" in script
    assert "wall_time>" in script
    assert "MISSING_REQUIRED=" in script
    assert "fire_easy_proof300_artifacts.tgz" in script
    assert "outputs/oracle_canary/easy_fire_proof300_preflight.json" in script
