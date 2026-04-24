#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/EvacOS2
CONFIG="$ROOT/training/config.remote-unsloth-3b-fire-floor-specialist.yaml"
LAUNCHER=/root/remote_fire_unsloth_train_call.sh
METRICS="$ROOT/outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv"
CKPT_ROOT="$ROOT/outputs/training/remote-unsloth-3b-fire-floor-specialist"
SUP_LOG=/root/fire_hour_supervisor.log
TRAIN_LOG=/root/fire_unsloth_train.log
ARTIFACT=/root/evacos2_fire_3b_artifacts.tgz
REPORT="$ROOT/outputs/training/fire_3b_hour_report.json"

TARGET_STEPS="${TARGET_STEPS:-200}"
TARGET_SECONDS="${TARGET_SECONDS:-3600}"
BUFFER_SECONDS="${BUFFER_SECONDS:-120}"

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export LD_LIBRARY_PATH="/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$SUP_LOG"
}

last_step() {
  "$ROOT/.venv/bin/python" - <<'PY'
import csv
from pathlib import Path
p = Path("/workspace/EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv")
if not p.exists():
    print(-1)
    raise SystemExit
last = -1
with p.open(newline="") as f:
    for row in csv.DictReader(f):
        try:
            last = int(float(row.get("step", -1)))
        except Exception:
            pass
print(last)
PY
}

write_report() {
  "$ROOT/.venv/bin/python" - <<'PY'
import csv, json
from pathlib import Path

root = Path("/workspace/EvacOS2")
metrics = root / "outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv"
ckpt_root = root / "outputs/training/remote-unsloth-3b-fire-floor-specialist"
report_path = root / "outputs/training/fire_3b_hour_report.json"

rows = []
if metrics.exists():
    with metrics.open(newline="") as f:
        rows = list(csv.DictReader(f))

def as_float(row, key, default=None):
    try:
        return float(row.get(key, ""))
    except Exception:
        return default

first = rows[0] if rows else {}
last = rows[-1] if rows else {}
floor_rewards = [as_float(r, "mean_norm_reward_floor") for r in rows]
floor_rewards = [x for x in floor_rewards if x is not None]
invalid_rates = [as_float(r, "invalid_action_rate") for r in rows]
invalid_rates = [x for x in invalid_rates if x is not None]
ckpts = sorted([p.name for p in ckpt_root.glob("ckpt_*")]) if ckpt_root.exists() else []

report = {
    "run": "remote-unsloth-3b-fire-floor-specialist",
    "metrics_csv": str(metrics),
    "checkpoint_root": str(ckpt_root),
    "rows": len(rows),
    "first_step": first.get("step"),
    "last_step": last.get("step"),
    "first_wall_seconds": first.get("wall_seconds"),
    "last_wall_seconds": last.get("wall_seconds"),
    "first_mean_norm_reward_floor": first.get("mean_norm_reward_floor"),
    "last_mean_norm_reward_floor": last.get("mean_norm_reward_floor"),
    "best_mean_norm_reward_floor": max(floor_rewards) if floor_rewards else None,
    "last_invalid_action_rate": last.get("invalid_action_rate"),
    "mean_invalid_action_rate": (sum(invalid_rates) / len(invalid_rates)) if invalid_rates else None,
    "latest_checkpoint": str(ckpt_root / "latest") if (ckpt_root / "latest").exists() else None,
    "checkpoints": ckpts[-8:],
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
}

START_EPOCH="$(date +%s)"
DEADLINE=$((START_EPOCH + TARGET_SECONDS))
log "supervisor started; target_seconds=$TARGET_SECONDS target_steps=$TARGET_STEPS"

while pgrep -f "timeout 3300s bash /root/remote_fire_unsloth_train_call.sh" >/dev/null; do
  step="$(last_step || echo -1)"
  log "current launch still running; latest_step=$step"
  sleep 60
done

step="$(last_step || echo -1)"
now="$(date +%s)"
remaining=$((DEADLINE - now - BUFFER_SECONDS))
log "current launch ended; latest_step=$step remaining_seconds=$remaining"

if [ "$remaining" -gt 300 ] && [ "$step" -lt $((TARGET_STEPS - 1)) ]; then
  "$ROOT/.venv/bin/python" - <<PY
from pathlib import Path
p = Path("$CONFIG")
text = p.read_text(encoding="utf-8")
lines = []
done = False
for line in text.splitlines():
    if line.strip().startswith("max_steps:"):
        lines.append("max_steps: $TARGET_STEPS")
        done = True
    else:
        lines.append(line)
if not done:
    lines.insert(0, "max_steps: $TARGET_STEPS")
p.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
PY
  log "resuming from latest checkpoint with max_steps=$TARGET_STEPS for up to ${remaining}s"
  timeout "$remaining"s bash "$LAUNCHER" >> "$TRAIN_LOG" 2>&1 || log "resume command exited non-zero or timed out"
else
  log "not resuming; run is close to target duration or target steps reached"
fi

log "writing report"
write_report | tee -a "$SUP_LOG"

log "packing artifacts at $ARTIFACT"
cd /workspace
tar -czf "$ARTIFACT" \
  EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist \
  EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv \
  EvacOS2/outputs/training/fire_3b_hour_report.json \
  EvacOS2/outputs/logs/remote-unsloth-3b-fire-floor-specialist \
  EvacOS2/training/config.remote-unsloth-3b-fire-floor-specialist.yaml \
  fire_unsloth_train.log \
  fire_hour_supervisor.log
log "artifact ready: $ARTIFACT"
