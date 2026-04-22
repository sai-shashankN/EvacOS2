# Remote GPU Setup

This is the canonical runbook for bringing up a remote Linux/CUDA box for
EvacOS2 training.

Use this when training on Vast.ai or any similar remote GPU host.

## Why this file exists

The remote training stack is sensitive to install order and import order:

- `unsloth` should be installed after the baseline training deps
- `vllm==0.10.2` should be installed last
- `unsloth` should be imported before `trl`, `transformers`, and `peft`
  when probing the environment

If you skip that ordering, you can end up with:

- large unnecessary package churn
- misleading import warnings
- false-negative health checks

## Recommended machine shape

Good first remote target:

- Linux + CUDA
- 1x RTX 4090
- ~150 GB disk
- Jupyter terminal or SSH access

## Repo bootstrap

```bash
cd /workspace
git clone https://github.com/sai-shashankN/EvacOS2.git
cd EvacOS2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Preferred install order

Do not start with a broad `requirements-training.txt` install on a fresh remote
box if the goal is the Unsloth + vLLM path.

Use this order instead:

```bash
pip uninstall -y transformers trl peft accelerate bitsandbytes vllm unsloth unsloth_zoo torchcodec >/dev/null 2>&1 || true
pip install "torch" "transformers>=4.56,<5.0" "trl" "peft<0.18,>=0.13" "accelerate" "bitsandbytes" "datasets"
pip install "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" "numpy>=1.26" "pyyaml" "nbformat" "wandb>=0.19"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl" "peft<0.18,>=0.13" "accelerate" "bitsandbytes"
pip install "vllm==0.10.2"
pip install -e .
```

Notes:

- `unsloth` patches parts of the stack in-place, so it belongs after the
  baseline training deps.
- `vllm==0.10.2` is the pinned API surface expected by the current notebook and
  rollout path.
- The checked-in notebook setup cell is the source of truth for this sequence.

## Warning-aware health check

Do not probe the environment by importing `transformers`, `trl`, or `peft`
before `unsloth`.

Use this exact probe:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

try:
    import unsloth
    print("unsloth OK", getattr(unsloth, "__version__", "no_version"))
except Exception as e:
    print("unsloth FAIL", repr(e))
    raise

mods = ["transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets", "vllm"]
for name in mods:
    try:
        m = __import__(name)
        print(name, "OK", getattr(m, "__version__", "no_version"))
    except Exception as e:
        print(name, "FAIL", repr(e))
PY
```

Why:

- the actual training entrypoint imports `unsloth` before `trl` / `peft`
- a naive probe can produce a warning or failure that the real training path
  would not hit

## Training-side import rule

For any custom smoke script or notebook cell:

```python
import unsloth
```

must appear before:

```python
import trl
import transformers
import peft
```

The current `training/train.py` entrypoint already follows this rule.

## Remote-only env vars worth setting

```bash
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
```

If the model pull needs auth:

```bash
export HF_TOKEN=...
```

## First smoke-test goal

Before a long run:

1. verify the warning-aware probe passes
2. verify `python -m training.train` can import its stack
3. run a very small smoke config
4. only then start the real checkpoint-producing run

## If the environment is already partially installed

If `torch`, CUDA, and `vllm` are already installed and the only visible issue is
an import-order warning from a naive probe, do not immediately rebuild the
whole venv.

Check the warning-aware probe first.

Rebuild only if the warning-aware probe still fails.
