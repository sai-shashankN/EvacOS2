#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-8}"

DISASTER_FAMILY="${DISASTER_FAMILY:-fire}"
case "$DISASTER_FAMILY" in
  fire|flood|gas) ;;
  *)
    echo "DISASTER_FAMILY must be one of fire, flood, gas; got $DISASTER_FAMILY" >&2
    exit 2
    ;;
esac

case "$DISASTER_FAMILY" in
  fire) STEPS=400 ;;
  flood) STEPS=500 ;;
  gas) STEPS=700 ;;
esac

REPO_URL="${EVACOS_REPO_URL:-https://github.com/sai-shashankN/EvacOS2.git}"
REPO_REF="${EVACOS_REPO_REF:-main}"
if [[ "${EVACOS_USE_EXISTING_SOURCE:-0}" == "1" ]]; then
  WORKDIR="${EVACOS_WORKDIR:-$(pwd)}"
else
  WORKDIR="${EVACOS_WORKDIR:-/workspace/EvacOS2}"
fi
RUN_NAME="remote-unsloth-3b-${DISASTER_FAMILY}-floor-specialist-quality-${STEPS}"
CONFIG="training/config.remote-unsloth-3b-${DISASTER_FAMILY}-floor-specialist-quality-${STEPS}.yaml"
METRICS="outputs/training/${RUN_NAME}-metrics.csv"
JSONL_DIR="outputs/logs/${RUN_NAME}"
CHECKPOINT_DIR="outputs/training/${RUN_NAME}"
ARTIFACT_DIR="/workspace/evacos2_${DISASTER_FAMILY}_quality_${STEPS}_artifacts"
REPORT="$ARTIFACT_DIR/${DISASTER_FAMILY}_quality_${STEPS}_report.json"

echo "HF_H200_JOB_START $(date -Is)"
echo "DISASTER_FAMILY=$DISASTER_FAMILY STEPS=$STEPS CONFIG=$CONFIG"
echo "REPO_URL=$REPO_URL REPO_REF=$REPO_REF"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN secret is required for artifact upload." >&2
  exit 2
fi

mkdir -p /workspace "$HF_HOME" "$ARTIFACT_DIR"

apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates build-essential

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" \
  accelerate bitsandbytes datasets "fsspec==2025.9.0" \
  "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" \
  "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib pytest \
  "huggingface_hub>=0.34.0,<1.0"
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes

python - <<'PY'
import subprocess
import time

deadline = time.time() + 300
last_error = None
while time.time() < deadline:
    try:
        import torch

        if torch.cuda.is_available():
            print(
                "cuda_ready",
                torch.__version__,
                torch.version.cuda,
                torch.cuda.get_device_name(0),
            )
            raise SystemExit(0)
        last_error = "torch.cuda.is_available() returned False"
    except SystemExit:
        raise
    except Exception as exc:
        last_error = repr(exc)
    subprocess.run(["nvidia-smi"], check=False)
    print(f"cuda_not_ready_yet: {last_error}; sleeping 15s", flush=True)
    time.sleep(15)

raise SystemExit(f"CUDA did not become available within 300s: {last_error}")
PY

if [[ "${EVACOS_USE_EXISTING_SOURCE:-0}" == "1" ]]; then
  echo "Using existing source tree at $WORKDIR"
else
  rm -rf "$WORKDIR"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$WORKDIR" || {
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
    git checkout "$REPO_REF"
  }
fi
cd "$WORKDIR"

python -m pip install --ignore-requires-python -e .

python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "no_version"))
PY

python -m pytest tests/test_config_schema.py tests/test_specialist_configs.py tests/test_check_grpo_contrast.py -q --basetemp .pytest_tmp_hf_h200

set +e
echo "TRAIN_START $(date -Is)"
python -u -c "from pathlib import Path; from training.train import run_training; run_training(Path('$CONFIG'))"
TRAIN_EXIT=$?
set -e
echo "TRAIN_EXIT=$TRAIN_EXIT $(date -Is)"

set +e
python scripts/check_grpo_contrast.py "$METRICS"
CONTRAST_EXIT=$?
set -e
echo "CONTRAST_EXIT=$CONTRAST_EXIT $(date -Is)"

python - <<PY
from pathlib import Path
import csv
import json

metrics = Path("$METRICS")
rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8"))) if metrics.exists() else []
watch = [
    "step",
    "wall_seconds",
    "invalid_action_rate",
    "mean_norm_reward_floor",
    "mean_raw_reward_floor",
    "floor_agent_advantage_std",
    "floor_agent_group_raw_reward_std_mean",
    "floor_route_action_rate",
    "floor_route_missing_target_rate",
]
summary = {
    "run_name": "$RUN_NAME",
    "config": "$CONFIG",
    "disaster_family": "$DISASTER_FAMILY",
    "planned_steps": $STEPS,
    "train_exit": $TRAIN_EXIT,
    "contrast_exit": $CONTRAST_EXIT,
    "metrics_path": str(metrics),
    "rows": len(rows),
}
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
ckpt = Path("$CHECKPOINT_DIR/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
Path("$REPORT").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

cp -f "$METRICS" "$ARTIFACT_DIR/" 2>/dev/null || true
cp -rf "$JSONL_DIR" "$ARTIFACT_DIR/logs" 2>/dev/null || true
cp -rf "$CHECKPOINT_DIR" "$ARTIFACT_DIR/checkpoints" 2>/dev/null || true
cp -f "$CONFIG" "$ARTIFACT_DIR/" 2>/dev/null || true
echo "$TRAIN_EXIT" > "$ARTIFACT_DIR/train_exit_code.txt"
echo "$CONTRAST_EXIT" > "$ARTIFACT_DIR/contrast_exit_code.txt"

python - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repo_id = os.environ.get("HF_ARTIFACT_REPO") or f"{user}/evacos2-h200-specialist-artifacts"
run_name = "$RUN_NAME"
artifact_dir = Path("$ARTIFACT_DIR")
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(artifact_dir),
    path_in_repo=f"runs/{run_name}",
    commit_message=f"Upload {run_name} artifacts",
)
print(f"ARTIFACT_REPO={repo_id}")
print(f"ARTIFACT_PATH=runs/{run_name}")
PY

echo "HF_H200_JOB_END $(date -Is)"
exit "$TRAIN_EXIT"
