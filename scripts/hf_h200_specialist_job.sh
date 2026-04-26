#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

DISASTER_FAMILY="${DISASTER_FAMILY:-fire}"
case "$DISASTER_FAMILY" in
  fire|flood|gas) ;;
  *)
    echo "DISASTER_FAMILY must be one of fire, flood, gas; got $DISASTER_FAMILY" >&2
    exit 2
    ;;
esac

case "$DISASTER_FAMILY" in
  fire) DEFAULT_STEPS=400 ;;
  flood) DEFAULT_STEPS=500 ;;
  gas) DEFAULT_STEPS=700 ;;
esac
STEPS="${HF_SPECIALIST_STEPS:-$DEFAULT_STEPS}"
RUN_LABEL="${HF_SPECIALIST_RUN_LABEL:-quality}"
BASE_CONFIG="training/config.remote-unsloth-3b-${DISASTER_FAMILY}-floor-specialist-quality-${DEFAULT_STEPS}.yaml"

REPO_URL="${EVACOS_REPO_URL:-https://github.com/sai-shashankN/EvacOS2.git}"
REPO_REF="${EVACOS_REPO_REF:-main}"
if [[ "${EVACOS_USE_EXISTING_SOURCE:-0}" == "1" ]]; then
  WORKDIR="${EVACOS_WORKDIR:-$(pwd)}"
else
  WORKDIR="${EVACOS_WORKDIR:-/workspace/EvacOS2}"
fi
RUN_NAME="remote-unsloth-3b-${DISASTER_FAMILY}-floor-specialist-${RUN_LABEL}-${STEPS}"
CONFIG="training/generated.${RUN_NAME}.yaml"
METRICS="outputs/training/${RUN_NAME}-metrics.csv"
JSONL_DIR="outputs/logs/${RUN_NAME}"
CHECKPOINT_DIR="outputs/training/${RUN_NAME}"
ARTIFACT_DIR="/workspace/evacos2_${DISASTER_FAMILY}_${RUN_LABEL}_${STEPS}_artifacts"
REPORT="$ARTIFACT_DIR/${DISASTER_FAMILY}_${RUN_LABEL}_${STEPS}_report.json"

echo "HF_SPECIALIST_JOB_START $(date -Is)"
echo "DISASTER_FAMILY=$DISASTER_FAMILY STEPS=$STEPS RUN_LABEL=$RUN_LABEL CONFIG=$CONFIG BASE_CONFIG=$BASE_CONFIG"
echo "REPO_URL=$REPO_URL REPO_REF=$REPO_REF"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN secret is required for artifact upload." >&2
  exit 2
fi

mkdir -p /workspace "$HF_HOME" "$ARTIFACT_DIR"

apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates build-essential

python -m pip install --upgrade pip setuptools wheel

repair_torch_cuda_if_needed() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No GPU visible to nvidia-smi; cannot repair PyTorch CUDA from inside the job." >&2
    return 1
  fi
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == "-1" || "${CUDA_VISIBLE_DEVICES:-}" == "none" ]]; then
    unset CUDA_VISIBLE_DEVICES
  fi
  python - <<'PY' && return 0
import os
import torch

print(
    "torch_preflight",
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "is_available": torch.cuda.is_available(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    flush=True,
)
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY

  echo "nvidia-smi sees GPU but torch CUDA is unavailable; reinstalling official torch 2.7.1 cu126 wheels." >&2
  python -m pip uninstall -y torch torchvision torchaudio triton >/tmp/torch_uninstall.log 2>&1 || true
  python -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
  python - <<'PY'
import ctypes
import os
import subprocess
import torch

print("cuda_env", {
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
}, flush=True)
subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    check=False,
)
try:
    ctypes.CDLL("libcuda.so.1")
    print("libcuda_load=ok", flush=True)
except Exception as exc:
    print(f"libcuda_load=ERR:{exc!r}", flush=True)
print(
    "torch_after_repair",
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "is_available": torch.cuda.is_available(),
    },
    flush=True,
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA still unavailable after torch cu126 reinstall")
print("gpu", torch.cuda.get_device_name(0), flush=True)
PY
}

repair_torch_cuda_if_needed
python -m pip install \
  "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" \
  accelerate bitsandbytes datasets "fsspec==2025.9.0" \
  "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" \
  "numpy>=1.26" pyyaml nbformat "wandb>=0.19" matplotlib pytest \
  "huggingface_hub>=0.34.0,<1.0"
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -m pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes

repair_torch_cuda_if_needed

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

python - <<PY
from pathlib import Path
import os
import yaml

base = Path("$BASE_CONFIG")
target = Path("$CONFIG")
steps = int("$STEPS")
run_name = "$RUN_NAME"
if not base.exists():
    raise SystemExit(f"Base specialist config not found: {base}")
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cfg["max_steps"] = steps
cfg.setdefault("rollout", {})["tier_schedule"] = [{"steps": steps, "mix": {"easy": steps}}]
cfg.setdefault("checkpoint", {})["root_dir"] = "$CHECKPOINT_DIR"
cfg["checkpoint"]["every_steps"] = int(os.environ.get("HF_SPECIALIST_CHECKPOINT_EVERY", "10" if steps <= 50 else "50"))
cfg.setdefault("metrics", {})["csv_path"] = "$METRICS"
cfg["metrics"]["jsonl_dir"] = "$JSONL_DIR"
cfg.setdefault("eval", {})["every_steps"] = int(os.environ.get("HF_SPECIALIST_EVAL_EVERY", "10" if steps <= 50 else "50"))
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"GENERATED_CONFIG={target} RUN_NAME={run_name} STEPS={steps}", flush=True)
PY

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
python -u -c "from pathlib import Path; from training.train import run_training; run_training(Path('$CONFIG'))" &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
  echo "TRAIN_HEARTBEAT $(date -Is) pid=$TRAIN_PID"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
  python - <<PY || true
from pathlib import Path
import csv
import json
import os
import time

metrics = Path("$METRICS")
ckpt = Path("$CHECKPOINT_DIR/latest")
payload = {
    "metrics_exists": metrics.exists(),
    "latest_checkpoint_exists": ckpt.exists(),
}
if metrics.exists():
    payload["metrics_age_seconds"] = round(time.time() - metrics.stat().st_mtime, 1)
    try:
        rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
    except Exception as exc:
        payload["metrics_error"] = repr(exc)
        rows = []
    payload["rows"] = len(rows)
    if rows:
        watch = [
            "step",
            "invalid_action_rate",
            "mean_norm_reward_floor",
            "mean_raw_reward_floor",
            "floor_agent_advantage_std",
            "floor_agent_group_raw_reward_std_mean",
        ]
        payload["last"] = {k: rows[-1].get(k) for k in watch if k in rows[-1]}
print("TRAIN_PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)
PY
done
wait "$TRAIN_PID"
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

echo "HF_SPECIALIST_JOB_END $(date -Is)"
exit "$TRAIN_EXIT"
