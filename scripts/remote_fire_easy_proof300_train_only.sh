#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-4}"

WORKDIR="/workspace/EvacOS2"
CONFIG="training/config.remote-unsloth-3b-fire-floor-specialist-easy-proof-300.yaml"
RUN_NAME="remote-unsloth-3b-fire-floor-specialist-easy-proof-300"
METRICS="outputs/training/${RUN_NAME}-metrics.csv"
RUN_DIR="outputs/training/${RUN_NAME}"
LOG_DIR="outputs/logs/${RUN_NAME}"
REPORT="outputs/training/fire_easy_proof300_report.json"
WATCHDOG_REPORT="outputs/training/fire_easy_proof300_watchdog.jsonl"
ARTIFACT_DIR="/root/evacos2_fire_easy_proof300_artifacts"
ARTIFACT_TGZ="$ARTIFACT_DIR/fire_easy_proof300_artifacts.tgz"
TRAIN_LOG="$ARTIFACT_DIR/fire_easy_proof300_train.log"
PID_FILE="/root/fire_easy_proof300.pid"

STALE_AFTER_SECONDS="${EVACOS_STALE_AFTER_SECONDS:-900}"
MAX_WALL_SECONDS="${EVACOS_MAX_WALL_SECONDS:-5400}"
TRAIN_TIMEOUT_SECONDS="${EVACOS_TRAIN_TIMEOUT_SECONDS:-$MAX_WALL_SECONDS}"
MAX_DISK_PERCENT="${EVACOS_MAX_DISK_PERCENT:-80}"
MAX_INVALID_RATE="${EVACOS_MAX_INVALID_RATE:-0.55}"
MIN_INVALID_RATE_STEP="${EVACOS_MIN_INVALID_RATE_STEP:-50}"

mkdir -p "$ARTIFACT_DIR"
cd "$WORKDIR"
source .venv/bin/activate
mkdir -p outputs/training outputs/logs outputs/oracle_canary "$RUN_DIR" "$LOG_DIR"

python - <<'PY'
from pathlib import Path
from training.config_schema import TrainingConfig
from training.train import _load_yaml_config, _validate_config_path_identity

config_path = Path("training/config.remote-unsloth-3b-fire-floor-specialist-easy-proof-300.yaml")
config = TrainingConfig(**_load_yaml_config(config_path))
_validate_config_path_identity(config_path, config)
assert config.rollout.disaster_families == ["fire"]
assert config.rollout.expanded_tier_schedule() == ["easy"] * 300
print(
    "CONFIG_OK",
    "max_steps=", config.max_steps,
    "eval_every=", config.eval.every_steps,
    "checkpoint_every=", config.checkpoint.every_steps,
    "trainable=", ",".join(config.roles.trainable),
)
PY

python scripts/run_oracle_canary.py \
  --task-id procgen_easy_fire \
  --tier easy \
  --disaster-family fire \
  --seeds 42,123,456,789 \
  --max-rounds 20 \
  --output-json outputs/oracle_canary/easy_fire_proof300_preflight.json

cat > /tmp/fire_easy_proof300_watchdog.py <<'PY'
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

pid = int(os.environ["TRAIN_PID"])
metrics = Path(os.environ["METRICS"])
report = Path(os.environ["WATCHDOG_REPORT"])
stale_after = int(os.environ.get("STALE_AFTER_SECONDS", "900"))
max_wall_seconds = int(os.environ.get("MAX_WALL_SECONDS", "5400"))
max_disk_percent = int(os.environ.get("MAX_DISK_PERCENT", "80"))
max_invalid_rate = float(os.environ.get("MAX_INVALID_RATE", "0.55"))
min_invalid_rate_step = int(os.environ.get("MIN_INVALID_RATE_STEP", "50"))

report.parent.mkdir(parents=True, exist_ok=True)
last_size = metrics.stat().st_size if metrics.exists() else 0
last_growth = time.time()
started_at = time.time()


def alive() -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def latest_metrics() -> dict[str, str]:
    if not metrics.exists():
        return {}
    try:
        with metrics.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    return rows[-1] if rows else {}


def stop(reason: str, row: dict[str, str]) -> None:
    event = {"time": time.time(), "event": "stop", "reason": reason, "latest": row}
    with report.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    time.sleep(20)
    if alive():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    sys.exit(0)


while alive():
    now = time.time()
    row = latest_metrics()
    current_size = metrics.stat().st_size if metrics.exists() else 0
    if current_size > last_size:
        last_size = current_size
        last_growth = now

    disk = shutil.disk_usage(str(Path.cwd()))
    disk_percent = round(100 * (disk.used / max(disk.total, 1)), 2)
    event = {
        "time": now,
        "event": "pulse",
        "metrics_exists": metrics.exists(),
        "metrics_size": current_size,
        "seconds_since_growth": round(now - last_growth, 1),
        "disk_percent": disk_percent,
        "latest": row,
    }
    with report.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    if disk_percent > max_disk_percent:
        stop(f"disk_percent>{max_disk_percent}", row)

    if now - started_at > max_wall_seconds:
        stop(f"wall_time>{max_wall_seconds}s", row)

    if now - last_growth > stale_after:
        stop(f"metrics_stale>{stale_after}s", row)

    try:
        step = int(float(row.get("step", "-1")))
        invalid = float(row.get("invalid_action_rate", "0"))
    except Exception:
        step = -1
        invalid = 0.0
    if step >= min_invalid_rate_step and invalid > max_invalid_rate:
        stop(f"invalid_action_rate>{max_invalid_rate}", row)

    time.sleep(300)

with report.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.time(), "event": "train_process_exited"}) + "\n")
PY

echo "TRAIN_START $(date -Is)" | tee "$TRAIN_LOG"
set +e
timeout "${TRAIN_TIMEOUT_SECONDS}s" \
  python -u -c "from pathlib import Path; from training.train import run_training; run_training(Path('$CONFIG'))" \
  >> "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
TRAIN_PID="$TRAIN_PID" \
METRICS="$METRICS" \
WATCHDOG_REPORT="$WATCHDOG_REPORT" \
STALE_AFTER_SECONDS="$STALE_AFTER_SECONDS" \
MAX_WALL_SECONDS="$MAX_WALL_SECONDS" \
MAX_DISK_PERCENT="$MAX_DISK_PERCENT" \
MAX_INVALID_RATE="$MAX_INVALID_RATE" \
MIN_INVALID_RATE_STEP="$MIN_INVALID_RATE_STEP" \
python /tmp/fire_easy_proof300_watchdog.py &
WATCHDOG_PID=$!
wait "$TRAIN_PID"
TRAIN_EXIT=$?
kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT $(date -Is)" | tee -a "$TRAIN_LOG"
echo "$TRAIN_EXIT" > "$ARTIFACT_DIR/train_exit_code.txt"

python - <<'PY'
from pathlib import Path
import csv
import json

metrics = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-easy-proof-300-metrics.csv")
jsonl_dir = Path("outputs/logs/remote-unsloth-3b-fire-floor-specialist-easy-proof-300")
rows = []
if metrics.exists():
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

watch = [
    "step",
    "invalid_action_rate",
    "mean_norm_reward_floor",
    "mean_raw_reward_floor",
    "raw_reward_std_floor",
    "norm_reward_std_floor",
    "candidates_per_floor_prompt",
    "sampling_temperature",
    "policy_loss",
    "floor_agent_policy_loss",
    "advantage_std",
    "floor_agent_advantage_std",
    "group_raw_reward_std_mean",
    "floor_agent_group_raw_reward_std_mean",
    "wait_rate",
    "active_empty_args_rate",
    "valid_but_hollow_action_rate",
    "floor_route_action_rate",
    "floor_route_exit_rate",
    "floor_route_stairwell_rate",
    "floor_route_room_rate",
    "floor_route_missing_target_rate",
]

summary = {
    "metrics_path": str(metrics),
    "rows": len(rows),
    "jsonl_dir": str(jsonl_dir),
}
if rows:
    summary["first"] = {k: rows[0].get(k) for k in watch if k in rows[0]}
    summary["last"] = {k: rows[-1].get(k) for k in watch if k in rows[-1]}
    for key in watch[1:]:
        vals = []
        for row in rows[-20:]:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            summary[f"last20_avg_{key}"] = sum(vals) / len(vals)

ckpt = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-easy-proof-300/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
summary["latest_checkpoint_files"] = (
    [str(p.relative_to(ckpt)) for p in ckpt.rglob("*")][:100] if ckpt.exists() else []
)
Path("outputs/training/fire_easy_proof300_report.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

PACKAGE_CHECK="$ARTIFACT_DIR/packaging_check.txt"
PACKAGE_EXIT=0
{
  echo "TRAIN_EXIT=$TRAIN_EXIT"
  echo "CONFIG=$CONFIG"
  echo "METRICS=$METRICS"
  echo "ARTIFACT_TGZ=$ARTIFACT_TGZ"
} > "$PACKAGE_CHECK"

if [[ "$TRAIN_EXIT" -eq 0 ]]; then
  for required in \
    "$CONFIG" \
    "$METRICS" \
    "$LOG_DIR" \
    "$RUN_DIR" \
    "$REPORT" \
    "$WATCHDOG_REPORT" \
    outputs/oracle_canary/easy_fire_proof300_preflight.json \
    "$TRAIN_LOG" \
    "$ARTIFACT_DIR/train_exit_code.txt"; do
    if [[ ! -e "$required" ]]; then
      echo "MISSING_REQUIRED=$required" | tee -a "$PACKAGE_CHECK"
      PACKAGE_EXIT=1
    fi
  done
fi

TAR_PATHS=("$CONFIG" "$TRAIN_LOG" "$ARTIFACT_DIR/train_exit_code.txt" "$PACKAGE_CHECK")
for maybe_path in \
  "$METRICS" \
  "$LOG_DIR" \
  "$RUN_DIR" \
  "$REPORT" \
  "$WATCHDOG_REPORT" \
  outputs/oracle_canary/easy_fire_proof300_preflight.json; do
  if [[ -e "$maybe_path" ]]; then
    TAR_PATHS+=("$maybe_path")
  fi
done

tar -czf "$ARTIFACT_TGZ" "${TAR_PATHS[@]}" 2>/tmp/fire_easy_proof300_tar_warnings.log
cp /tmp/fire_easy_proof300_tar_warnings.log "$ARTIFACT_DIR/" || true
tar -tzf "$ARTIFACT_TGZ" >/tmp/fire_easy_proof300_tar_listing.txt
cp /tmp/fire_easy_proof300_tar_listing.txt "$ARTIFACT_DIR/" || true
echo "ARTIFACT_TGZ=$ARTIFACT_TGZ"
if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  exit "$TRAIN_EXIT"
fi
exit "$PACKAGE_EXIT"
