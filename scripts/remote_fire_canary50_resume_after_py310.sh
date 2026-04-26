#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-4}"

WORKDIR="/workspace/EvacOS2"
CONFIG="training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml"
METRICS="outputs/training/remote-unsloth-3b-fire-floor-specialist-canary-50-metrics.csv"
ARTIFACT_DIR="/root/evacos2_fire_canary50_artifacts"
ARTIFACT_TGZ="$ARTIFACT_DIR/fire_canary50_artifacts.tgz"

mkdir -p "$ARTIFACT_DIR"
cd "$WORKDIR"
source .venv/bin/activate
mkdir -p outputs/training outputs/logs outputs/oracle_canary

python -m pip install --ignore-requires-python -e .
python -m pip install pytest

python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "no_version"))
PY

python -m pytest tests/test_specialist_configs.py tests/test_policy_adapter.py tests/test_round_protocol_apply.py tests/test_arbitration.py tests/test_prompts.py tests/test_metrics.py tests/test_oracle_canary.py -q --basetemp .pytest_tmp_remote_canary
python scripts/run_oracle_canary.py \
  --task-id procgen_easy_fire \
  --tier easy \
  --disaster-family fire \
  --seeds 42,123,456 \
  --max-rounds 20 \
  --output-json outputs/oracle_canary/easy_fire_route_fix_remote.json

set +e
echo "TRAIN_START $(date -Is)"
python -u -c "from pathlib import Path; from training.train import run_training; run_training(Path('$CONFIG'))"
TRAIN_EXIT=$?
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT $(date -Is)"

python - <<'PY'
from pathlib import Path
import csv
import json

metrics = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-canary-50-metrics.csv")
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
    "floor_route_legacy_egress_alias_rate",
    "floor_route_missing_target_rate",
]

summary = {"metrics_path": str(metrics), "rows": len(rows)}
if rows:
    summary["first"] = {k: rows[0].get(k) for k in watch if k in rows[0]}
    summary["last"] = {k: rows[-1].get(k) for k in watch if k in rows[-1]}
    for key in watch[1:]:
        vals = []
        for row in rows[-10:]:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        if vals:
            summary[f"last10_avg_{key}"] = sum(vals) / len(vals)

ckpt = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-canary-50/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
summary["latest_checkpoint_files"] = (
    [str(p.relative_to(ckpt)) for p in ckpt.rglob("*")][:80] if ckpt.exists() else []
)
Path("outputs/training/fire_canary50_report.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

tar -czf "$ARTIFACT_TGZ" \
  "$CONFIG" \
  "$METRICS" \
  outputs/logs/remote-unsloth-3b-fire-floor-specialist-canary-50 \
  outputs/training/remote-unsloth-3b-fire-floor-specialist-canary-50 \
  outputs/training/fire_canary50_report.json \
  outputs/oracle_canary/easy_fire_route_fix_remote.json \
  2>/tmp/fire_canary50_tar_warnings.log || true

cp /tmp/fire_canary50_tar_warnings.log "$ARTIFACT_DIR/" || true
echo "$TRAIN_EXIT" > "$ARTIFACT_DIR/train_exit_code.txt"
exit "$TRAIN_EXIT"
