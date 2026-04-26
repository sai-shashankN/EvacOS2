#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export PYTORCH_ALLOC_CONF=expandable_segments:True
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-4}"

REPO_ZIP="/root/evacos2_remote_upload.zip"
WORKDIR="/workspace/EvacOS2"
CONFIG="training/config.remote-unsloth-3b-fire-floor-specialist-signal-canary-10.yaml"
METRICS="outputs/training/remote-unsloth-3b-fire-floor-specialist-signal-canary-10-metrics.csv"
JSONL_DIR="outputs/logs/remote-unsloth-3b-fire-floor-specialist-signal-canary-10"
CHECKPOINT_DIR="outputs/training/remote-unsloth-3b-fire-floor-specialist-signal-canary-10"
ARTIFACT_DIR="/root/evacos2_fire_signal_canary10_artifacts"
ARTIFACT_TGZ="$ARTIFACT_DIR/fire_signal_canary10_artifacts.tgz"
REPORT="$ARTIFACT_DIR/fire_signal_canary10_report.json"

mkdir -p /workspace /workspace/hf_cache "$ARTIFACT_DIR"

apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev \
  git curl ca-certificates build-essential unzip

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
python3 - <<'PY'
import zipfile
zipfile.ZipFile('/root/evacos2_remote_upload.zip').extractall('/workspace/EvacOS2')
PY

cd "$WORKDIR"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
mkdir -p outputs/training outputs/logs

python -m pip install --upgrade pip setuptools wheel
if python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(1)
import torch
if not torch.cuda.is_available():
    sys.exit(2)
print("using preinstalled torch", torch.__version__, torch.version.cuda)
PY
then
  echo "Preinstalled torch is usable; skipping torch reinstall."
else
  python -m pip install "torch==2.10.0"
fi
if python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torchvision") is not None else 1)
PY
then
  echo "torchvision is available."
else
  python -m pip install --no-deps "torchvision==0.25.0"
fi
python -m pip install \
  "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" \
  accelerate bitsandbytes datasets "fsspec==2025.9.0" \
  "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" \
  "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib pytest
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes
# Vast base images can lag local Python metadata; the training path is 3.10-safe.
python -m pip install --ignore-requires-python -e .

python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "no_version"))
PY

python -m pytest \
  tests/test_group_for_grpo.py \
  tests/test_config_schema.py \
  tests/test_train_build_grpo_trainer.py \
  -q --basetemp .pytest_tmp_remote_signal_canary10

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

metrics = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-signal-canary-10-metrics.csv")
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
    "floor_route_missing_target_rate",
    "watchdog_status",
    "watchdog_reason",
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

ckpt = Path("outputs/training/remote-unsloth-3b-fire-floor-specialist-signal-canary-10/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
summary["latest_checkpoint_files"] = (
    [str(p.relative_to(ckpt)) for p in ckpt.rglob("*")][:80] if ckpt.exists() else []
)
Path("/root/evacos2_fire_signal_canary10_artifacts/fire_signal_canary10_report.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

cp -f "$METRICS" "$ARTIFACT_DIR/" 2>/dev/null || true
cp -rf "$JSONL_DIR" "$ARTIFACT_DIR/logs" 2>/dev/null || true
cp -rf "$CHECKPOINT_DIR" "$ARTIFACT_DIR/checkpoints" 2>/dev/null || true
cp -f "$CONFIG" "$ARTIFACT_DIR/" 2>/dev/null || true
echo "$TRAIN_EXIT" > "$ARTIFACT_DIR/train_exit_code.txt"
tar -C "$ARTIFACT_DIR" -czf "$ARTIFACT_TGZ" .

exit "$TRAIN_EXIT"
