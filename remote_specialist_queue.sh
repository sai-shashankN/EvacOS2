#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/EvacOS2
PY="$ROOT/.venv/bin/python"
ARTIFACT_DIR=/root/evacos2_specialist_artifacts
QUEUE_LOG=/root/specialist_queue.log

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export LD_LIBRARY_PATH="/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ARTIFACT_DIR"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"
}

set_max_steps() {
  local config="$1"
  local steps="$2"
  "$PY" - "$config" "$steps" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
steps = sys.argv[2]
text = path.read_text(encoding="utf-8")
lines = []
changed = False
for line in text.splitlines():
    if line.strip().startswith("max_steps:"):
        lines.append(f"max_steps: {steps}")
        changed = True
    else:
        lines.append(line)
if not changed:
    lines.insert(0, f"max_steps: {steps}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

write_report() {
  local name="$1"
  local metrics="$2"
  local ckpt_root="$3"
  local report="$4"
  "$PY" - "$name" "$metrics" "$ckpt_root" "$report" <<'PY'
import csv
import json
import sys
from pathlib import Path

name, metrics_s, ckpt_root_s, report_s = sys.argv[1:5]
metrics = Path(metrics_s)
ckpt_root = Path(ckpt_root_s)
report = Path(report_s)

rows = []
if metrics.exists():
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

def fl(row, key):
    try:
        return float(row.get(key, ""))
    except Exception:
        return None

first = rows[0] if rows else {}
last = rows[-1] if rows else {}
floor_norm = [fl(r, "mean_norm_reward_floor") for r in rows]
floor_norm = [v for v in floor_norm if v is not None]
floor_raw = [fl(r, "mean_raw_reward_floor") for r in rows]
floor_raw = [v for v in floor_raw if v is not None]
invalid = [fl(r, "invalid_action_rate") for r in rows]
invalid = [v for v in invalid if v is not None]
recent = rows[-10:]
recent_norm = [fl(r, "mean_norm_reward_floor") for r in recent]
recent_norm = [v for v in recent_norm if v is not None]

payload = {
    "run": name,
    "metrics_csv": str(metrics),
    "checkpoint_root": str(ckpt_root),
    "rows": len(rows),
    "first_step": first.get("step"),
    "last_step": last.get("step"),
    "first_wall_seconds": first.get("wall_seconds"),
    "last_wall_seconds": last.get("wall_seconds"),
    "first_mean_norm_reward_floor": first.get("mean_norm_reward_floor"),
    "last_mean_norm_reward_floor": last.get("mean_norm_reward_floor"),
    "best_mean_norm_reward_floor": max(floor_norm) if floor_norm else None,
    "recent10_mean_norm_reward_floor": (
        sum(recent_norm) / len(recent_norm) if recent_norm else None
    ),
    "first_mean_raw_reward_floor": first.get("mean_raw_reward_floor"),
    "last_mean_raw_reward_floor": last.get("mean_raw_reward_floor"),
    "best_mean_raw_reward_floor": max(floor_raw) if floor_raw else None,
    "last_invalid_action_rate": last.get("invalid_action_rate"),
    "mean_invalid_action_rate": (
        sum(invalid) / len(invalid) if invalid else None
    ),
    "latest_checkpoint_exists": (ckpt_root / "latest").exists(),
    "latest_checkpoint": str(ckpt_root / "latest"),
    "checkpoints": sorted(p.name for p in ckpt_root.glob("ckpt_*"))[-8:]
    if ckpt_root.exists()
    else [],
}
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

package_run() {
  local name="$1"
  local metrics="$2"
  local ckpt_root="$3"
  local jsonl_dir="$4"
  local config="$5"
  local train_log="$6"
  local report="$7"
  local artifact="$ARTIFACT_DIR/${name}.tgz"

  log "writing report for $name"
  write_report "$name" "$metrics" "$ckpt_root" "$report" | tee -a "$QUEUE_LOG"

  log "packing $name -> $artifact"
  tar -czf "$artifact" \
    -C /workspace EvacOS2/"${metrics#/workspace/EvacOS2/}" \
    -C /workspace EvacOS2/"${ckpt_root#/workspace/EvacOS2/}" \
    -C /workspace EvacOS2/"${jsonl_dir#/workspace/EvacOS2/}" \
    -C /workspace EvacOS2/"${config#/workspace/EvacOS2/}" \
    -C /workspace EvacOS2/"${report#/workspace/EvacOS2/}" \
    -C /root "$(basename "$train_log")" \
    -C /root "$(basename "$QUEUE_LOG")"
  log "artifact ready for $name: $artifact"
}

run_specialist() {
  local family="$1"
  local name="remote-unsloth-3b-${family}-floor-specialist-750"
  local config="$ROOT/training/config.remote-unsloth-3b-${family}-floor-specialist-750.yaml"
  local ckpt_root="$ROOT/outputs/training/${name}"
  local metrics="$ROOT/outputs/training/${name}-metrics.csv"
  local jsonl_dir="$ROOT/outputs/logs/${name}"
  local train_log="/root/${family}_unsloth_train.log"
  local report="$ROOT/outputs/training/${family}_3b_750step_report.json"

  log "starting $family specialist 750-step run"
  rm -rf "$ckpt_root" "$metrics" "$jsonl_dir"
  cd "$ROOT"
  source .venv/bin/activate
  "$PY" - "$config" > "$train_log" 2>&1 <<'PY'
from pathlib import Path
import sys
from training.train import run_training

run_training(Path(sys.argv[1]))
PY
  log "$family specialist training completed"
  package_run "$name" "$metrics" "$ckpt_root" "$jsonl_dir" "$config" "$train_log" "$report"
}

fire_training_active() {
  pgrep -f "remote_fire_unsloth_train_call.sh" >/dev/null && return 0
  pgrep -f "remote_fire_train_call.sh" >/dev/null && return 0
  pgrep -f "config.remote-unsloth-3b-fire-floor-specialist-750.yaml" >/dev/null && return 0
  pgrep -f "config.remote-unsloth-3b-fire-floor-specialist.yaml" >/dev/null && return 0
  pgrep -f "config.fire-hour.yaml" >/dev/null && return 0
  return 1
}

wait_for_fire() {
  local metrics="$ROOT/outputs/training/remote-unsloth-3b-fire-floor-specialist-750-metrics.csv"
  log "waiting for currently-running fire specialist to finish at its loaded 750-step cap"
  while fire_training_active; do
    local step="-1"
    if [ -f "$metrics" ]; then
      step="$("$PY" - <<'PY'
import csv
from pathlib import Path
p=Path('/workspace/EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist-750-metrics.csv')
if not p.exists():
    p=Path('/workspace/EvacOS2/outputs/training/remote-unsloth-3b-fire-floor-specialist-metrics.csv')
rows=list(csv.DictReader(p.open(newline='', encoding='utf-8')))
print(rows[-1].get('step') if rows else -1)
PY
)"
    fi
    log "fire still running; latest_step=$step"
    sleep 60
  done
}

package_fire() {
  local name="remote-unsloth-3b-fire-floor-specialist-750"
  local metrics="$ROOT/outputs/training/${name}-metrics.csv"
  local ckpt_root="$ROOT/outputs/training/${name}"
  local jsonl_dir="$ROOT/outputs/logs/${name}"
  local config="$ROOT/training/config.remote-unsloth-3b-fire-floor-specialist-750.yaml"
  local train_log="/root/fire_unsloth_train.log"
  local report="$ROOT/outputs/training/fire_3b_750step_report.json"
  package_run "$name" "$metrics" "$ckpt_root" "$jsonl_dir" "$config" "$train_log" "$report"
}

log "queue started"
wait_for_fire
log "fire process ended; packaging fire before flood/gas"
package_fire
run_specialist flood
run_specialist gas
log "all specialist runs complete"
