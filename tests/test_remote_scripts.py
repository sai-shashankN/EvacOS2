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
    assert 'pgrep -f "config.remote-unsloth-3b-fire-floor-specialist.yaml"' in script
    assert 'pgrep -f "config.fire-hour.yaml"' in script
    assert "while fire_training_active; do" in script
