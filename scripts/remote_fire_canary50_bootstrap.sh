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
CONFIG="training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml"
METRICS="outputs/training/remote-unsloth-3b-fire-floor-specialist-canary-50-metrics.csv"
ARTIFACT_DIR="/root/evacos2_fire_canary50_artifacts"
ARTIFACT_TGZ="$ARTIFACT_DIR/fire_canary50_artifacts.tgz"

mkdir -p /workspace /workspace/hf_cache "$ARTIFACT_DIR"

apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev python3.10-dev \
  git curl ca-certificates build-essential unzip

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
python3 - <<'PY'
import zipfile
zipfile.ZipFile('/root/evacos2_remote_upload.zip').extractall('/workspace/EvacOS2')
PY

cd "$WORKDIR"
python3 -m venv .venv
source .venv/bin/activate
mkdir -p outputs/training outputs/logs outputs/oracle_canary

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.10.0" "torchvision==0.25.0"
python -m pip install \
  "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" \
  accelerate bitsandbytes datasets "fsspec==2025.9.0" \
  "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" \
  "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib pytest
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes
# Vast CUDA base images may still ship Python 3.10. The repo metadata targets
# Python >=3.11 for local/dev parity, but the training code path is 3.10-safe.
python -m pip install --ignore-requires-python -e .

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
    "advantage_std_floor",
    "wait_rate",
    "active_empty_args_rate",
    "floor_agent_valid_but_hollow_action_rate",
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
