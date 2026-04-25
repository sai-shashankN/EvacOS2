# Remote GPU Setup

This is the canonical runbook for bringing up a remote Linux/CUDA box for
EvacOS2 training.

Use this when training on Vast.ai or any similar remote GPU host.

## Why this file exists

The remote training stack is sensitive to install order and import order:

- `unsloth` should be installed after the baseline training deps
- `vllm==0.10.2` should be installed last
- `transformers==4.56.2` and `peft==0.19.1` are the tested-compatible pair
  for the current Unsloth + vLLM setup
- `unsloth` should be imported before `trl`, `transformers`, and `peft`
  when probing the environment

If you skip that ordering, you can end up with:

- large unnecessary package churn
- misleading import warnings
- false-negative health checks

## Recommended machine shape

Good first remote target for debugging a single 3B specialist:

- Linux + CUDA
- 1x RTX 4090
- ~150 GB disk
- Jupyter terminal or SSH access

Cost-aware specialist plan for future runs:

- Use parallel 24 GB consumer/workstation GPUs for the 3B floor specialists.
- Preferred cheap lanes: 1x RTX 3090 24 GB per specialist when reliability and
  storage are acceptable.
- Also acceptable for 3B-only specialist sweeps: RTX 4090 24 GB, RTX A5000
  24 GB, or A10/A10G 24 GB.
- Reserve A100-class hardware for the finale: 7B orchestrator/generalist
  training, mixed-disaster curriculum, or larger evaluation runs.
- The 4090 was safe and fast for the fire 3B proof run, but it was likely more
  than necessary once the stack was stable. Next time, prefer three cheaper
  parallel 3090-style lanes for fire/flood/gas, then move the selected artifacts
  into the A100 finale.

Future 3B specialist curriculum:

- For real specialist training, run all tiers: `easy`, `medium`, `hard`, then
  `brutal`. Do not introduce alternate names for the hardest tier.
- Prefer staged curriculum blocks over per-step round-robin. Do not alternate
  `easy -> medium -> hard -> brutal` every single step at the start; that makes
  the reward distribution noisy before the policy has stable behavior.
- Default real specialist run:
  - 200 steps `easy`
  - 160 steps `medium` + 40 steps `easy` replay
  - 160 steps `hard` + 30 steps `medium` replay + 10 steps `easy` replay
  - 115 steps `brutal` + 25 steps `hard` replay + 10 steps `medium` replay
  - Total: 750 steps per disaster specialist
- Use the checked-in `*-750.yaml` configs for real runs. They use
  `rollout.tier_schedule`, and the config loader validates that the expanded
  schedule exactly matches `max_steps`.
- Replay samples are balanced/interleaved inside each stage so the floor policy
  does not forget easier behaviors while the main difficulty increases.
- A cheaper smoke run may still use 100-160 steps, but do not treat that as the
  final specialist quality target.
- If only running a 100-step proof, use a staged mini-curriculum such as
  30 easy / 30 medium / 25 hard / 15 brutal, but expect noisier graphs than the
  easy-only proof run.
- Always evaluate the final checkpoint on fixed held-out seeds for every tier,
  even if training used a curriculum. Training curves show learning; held-out
  tier eval proves generalization.

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
pip install "torch" "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" "accelerate" "bitsandbytes" "datasets"
pip install "pydantic>=2,<3" "fastapi>=0.115" "uvicorn>=0.30" "numpy>=1.26" "pyyaml" "nbformat" "wandb>=0.19"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl==0.24.0" "peft==0.19.1" "accelerate" "bitsandbytes"
pip install "vllm==0.10.2"
pip install -e .
```

Notes:

- `unsloth` patches parts of the stack in-place, so it belongs after the
  baseline training deps.
- `vllm==0.10.2` is the pinned API surface expected by the current notebook and
  rollout path.
- `transformers==4.56.2` and `peft==0.19.1` are the tested pair that avoids the
  `HybridCache` import failure seen with `transformers==5.5.0` and
  `peft==0.17.1`.
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

## Upload the selected adapter

After a successful run, publish the adapter folder rather than committing large
binary weights to Git:

```bash
export HF_ADAPTER_REPO=your-username/evacos2-lora-adapter
python scripts/upload_adapter.py \
  outputs/training/checkpoints/latest/lora_adapter \
  "$HF_ADAPTER_REPO"
```

## If the environment is already partially installed

If `torch`, CUDA, and `vllm` are already installed and the only visible issue is
an import-order warning from a naive probe, do not immediately rebuild the
whole venv.

Check the warning-aware probe first.

Rebuild only if the warning-aware probe still fails.

If the existing env fails with:

```text
ImportError: cannot import name 'HybridCache' from 'transformers'
```

the smallest proven repair is:

```bash
pip install --upgrade "transformers==4.56.2" "peft==0.19.1"
```

Then rerun the warning-aware probe.

## Optional SSH control path

If you want to operate the box from the local machine instead of using the
Jupyter terminal UI:

```powershell
$env:PATH += ";$env:APPDATA\Python\Python314\Scripts"
vastai create ssh-key -y
vastai attach ssh 35440032 $HOME\.ssh\id_ed25519.pub
ssh -i $HOME\.ssh\id_ed25519 -p 40989 root@96.241.192.5
```

Replace the instance id, host, and port with the values from:

```powershell
vastai show instances --raw
```
