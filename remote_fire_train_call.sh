set -u
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
cd /workspace/EvacOS2
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from training.train import run_training

run_training(Path("training/config.fire-hour-hf.yaml"))
PY
