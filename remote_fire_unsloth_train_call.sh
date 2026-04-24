#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
export LD_LIBRARY_PATH="/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/workspace/EvacOS2/.venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

cd /workspace/EvacOS2
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
from training.train import run_training

run_training(Path("training/config.remote-unsloth-3b-fire-floor-specialist.yaml"))
PY
