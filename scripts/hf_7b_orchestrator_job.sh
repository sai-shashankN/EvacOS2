#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export EVACOS_LOGPROB_MICROBATCH_SIZE="${EVACOS_LOGPROB_MICROBATCH_SIZE:-4}"

RUN_NAME="${RUN_NAME:-remote-unsloth-7b-orchestrator-frozen-specialists-main}"
STEPS="${STEPS:-120}"
BASE_CONFIG="${BASE_CONFIG:-training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml}"
WORKDIR="${EVACOS_WORKDIR:-/workspace/EvacOS2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/evacos2_7b_orchestrator_artifacts}"
REPORT="$ARTIFACT_DIR/${RUN_NAME}_report.json"
export RUN_NAME STEPS BASE_CONFIG WORKDIR ARTIFACT_DIR REPORT

echo "HF_7B_JOB_START $(date -Is)"
echo "RUN_NAME=$RUN_NAME STEPS=$STEPS BASE_CONFIG=$BASE_CONFIG"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN secret is required for source, adapter, and artifact access." >&2
  exit 2
fi

for required in HF_SOURCE_REPO HF_SOURCE_FILENAME HF_FIRE_ADAPTER_REPO HF_FIRE_ADAPTER_PATH HF_FLOOD_ADAPTER_REPO HF_FLOOD_ADAPTER_PATH HF_GAS_ADAPTER_REPO HF_GAS_ADAPTER_PATH HF_7B_ARTIFACT_REPO; do
  if [[ -z "${!required:-}" ]]; then
    echo "$required is required." >&2
    exit 2
  fi
done

if [[ -n "${HF_7B_RESUME_CHECKPOINT_REPO:-}" && -z "${HF_7B_RESUME_CHECKPOINT_PATH:-}" ]]; then
  echo "HF_7B_RESUME_CHECKPOINT_PATH is required when HF_7B_RESUME_CHECKPOINT_REPO is set." >&2
  exit 2
fi
if [[ -z "${HF_7B_RESUME_CHECKPOINT_REPO:-}" && -n "${HF_7B_RESUME_CHECKPOINT_PATH:-}" ]]; then
  echo "HF_7B_RESUME_CHECKPOINT_REPO is required when HF_7B_RESUME_CHECKPOINT_PATH is set." >&2
  exit 2
fi

mkdir -p /workspace/source "$WORKDIR" "$HF_HOME" "$ARTIFACT_DIR" /workspace/frozen_specialists

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

deadline = time.time() + int(__import__("os").environ.get("CUDA_WAIT_SECONDS", "300"))
last_error = None
while time.time() < deadline:
    try:
        import torch

        print("torch_cuda_probe", torch.__version__, torch.version.cuda, torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("cuda_ready", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
            raise SystemExit(0)
        last_error = "torch.cuda.is_available() returned False"
    except SystemExit:
        raise
    except Exception as exc:
        last_error = repr(exc)
    subprocess.run(["nvidia-smi"], check=False)
    print(f"cuda_not_ready_yet: {last_error}; sleeping 15s", flush=True)
    time.sleep(15)

raise SystemExit(f"CUDA did not become available: {last_error}")
PY

python - <<'PY'
import os
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

if os.environ.get("EVACOS_SOURCE_READY") == "1" and Path("/workspace/EvacOS2/training").exists():
    print("source_ready=/workspace/EvacOS2", flush=True)
else:
    source = hf_hub_download(
        repo_id=os.environ["HF_SOURCE_REPO"],
        filename=os.environ["HF_SOURCE_FILENAME"],
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        local_dir="/workspace/source",
    )
    with tarfile.open(source, "r:gz") as tar:
        tar.extractall("/workspace/EvacOS2")
    print("source_extracted=/workspace/EvacOS2", flush=True)

families = {
    "fire": ("HF_FIRE_ADAPTER_REPO", "HF_FIRE_ADAPTER_PATH"),
    "flood": ("HF_FLOOD_ADAPTER_REPO", "HF_FLOOD_ADAPTER_PATH"),
    "gas": ("HF_GAS_ADAPTER_REPO", "HF_GAS_ADAPTER_PATH"),
}
for family, (repo_key, path_key) in families.items():
    repo_id = os.environ[repo_key]
    adapter_path = os.environ[path_key].strip("/")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        allow_patterns=[f"{adapter_path}/*"],
        local_dir=f"/workspace/frozen_specialists/{family}_repo",
    )
    resolved = f"/workspace/frozen_specialists/{family}_repo/{adapter_path}"
    print(f"{family}_adapter={resolved}", flush=True)
PY

cd "$WORKDIR"
python -m pip install --ignore-requires-python -e .

python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth
print("unsloth", getattr(unsloth, "__version__", "no_version"))
PY

python -m pytest tests/test_config_schema.py tests/test_scope_router.py tests/test_check_grpo_contrast.py -q --basetemp .pytest_tmp_hf_7b

RUN_CONFIG="/workspace/${RUN_NAME}.yaml"
python - <<'PY'
import os
from pathlib import Path
import yaml

base_config = Path(os.environ["BASE_CONFIG"])
run_config = Path(f"/workspace/{os.environ['RUN_NAME']}.yaml")
cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
cfg["max_steps"] = int(os.environ["STEPS"])
cfg.setdefault("checkpoint", {})["root_dir"] = f"outputs/training/{os.environ['RUN_NAME']}"
cfg.setdefault("metrics", {})["csv_path"] = f"outputs/training/{os.environ['RUN_NAME']}-metrics.csv"
cfg.setdefault("metrics", {})["jsonl_dir"] = f"outputs/logs/{os.environ['RUN_NAME']}"
cfg.setdefault("roles", {})["frozen_floor_specialist_adapter_paths"] = {
    "fire": f"/workspace/frozen_specialists/fire_repo/{os.environ['HF_FIRE_ADAPTER_PATH'].strip('/')}",
    "flood": f"/workspace/frozen_specialists/flood_repo/{os.environ['HF_FLOOD_ADAPTER_PATH'].strip('/')}",
    "gas": f"/workspace/frozen_specialists/gas_repo/{os.environ['HF_GAS_ADAPTER_PATH'].strip('/')}",
}
run_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(run_config)
PY

METRICS="outputs/training/${RUN_NAME}-metrics.csv"
JSONL_DIR="outputs/logs/${RUN_NAME}"
CHECKPOINT_DIR="outputs/training/${RUN_NAME}"

if [[ -n "${HF_7B_RESUME_CHECKPOINT_REPO:-}" ]]; then
  python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_7B_RESUME_CHECKPOINT_REPO"]
checkpoint_path = os.environ["HF_7B_RESUME_CHECKPOINT_PATH"].strip("/")
target = Path(os.environ["CHECKPOINT_DIR"])
local_root = Path("/workspace/resume_7b_checkpoint_repo")
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    allow_patterns=[f"{checkpoint_path}/**"],
    local_dir=str(local_root),
)
source = local_root / checkpoint_path
if not (source / "latest" / "meta.json").exists():
    raise SystemExit(f"Resume checkpoint is missing latest/meta.json: {source}")
if target.exists():
    shutil.rmtree(target)
shutil.copytree(source, target)
print(f"RESUME_CHECKPOINT_COPIED repo={repo_id} path={checkpoint_path} target={target}", flush=True)
PY
fi

set +e
echo "TRAIN_START $(date -Is)"
python -u -c "from pathlib import Path; from training.train import run_training; run_training(Path('$RUN_CONFIG'))"
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
    "mean_norm_reward_orch",
    "mean_raw_reward_orch",
    "orchestrator_advantage_std",
    "orchestrator_group_raw_reward_std_mean",
]
summary = {
    "run_name": "$RUN_NAME",
    "train_exit": $TRAIN_EXIT,
    "contrast_exit": $CONTRAST_EXIT,
    "metrics_path": str(metrics),
    "rows": len(rows),
}
if rows:
    summary["first"] = {k: rows[0].get(k) for k in watch if k in rows[0]}
    summary["last"] = {k: rows[-1].get(k) for k in watch if k in rows[-1]}
ckpt = Path("$CHECKPOINT_DIR/latest")
summary["latest_checkpoint_exists"] = ckpt.exists()
Path("$REPORT").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

cp -f "$METRICS" "$ARTIFACT_DIR/" 2>/dev/null || true
cp -rf "$JSONL_DIR" "$ARTIFACT_DIR/logs" 2>/dev/null || true
cp -rf "$CHECKPOINT_DIR" "$ARTIFACT_DIR/checkpoints" 2>/dev/null || true
cp -f "$RUN_CONFIG" "$ARTIFACT_DIR/" 2>/dev/null || true
echo "$TRAIN_EXIT" > "$ARTIFACT_DIR/train_exit_code.txt"
echo "$CONTRAST_EXIT" > "$ARTIFACT_DIR/contrast_exit_code.txt"

python - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_7B_ARTIFACT_REPO"]
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(Path("$ARTIFACT_DIR")),
    path_in_repo=f"runs/{os.environ['RUN_NAME']}",
    commit_message=f"Upload {os.environ['RUN_NAME']} artifacts",
)
print(f"ARTIFACT_REPO={repo_id}")
print(f"ARTIFACT_PATH=runs/{os.environ['RUN_NAME']}")
PY

echo "HF_7B_JOB_END $(date -Is)"
exit "$TRAIN_EXIT"
