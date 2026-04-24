set -u
cd /workspace/EvacOS2
source .venv/bin/activate
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl==0.24.0" "peft==0.19.1" accelerate bitsandbytes
python - <<'PY'
import torch

print("torch", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
import unsloth

print("unsloth", getattr(unsloth, "__version__", "no_version"))
from peft import LoraConfig
from trl import GRPOTrainer

print("peft_trl_ok")
PY
echo REPAIR_DONE
