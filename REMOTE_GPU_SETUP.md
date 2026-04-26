# Remote GPU Setup

This is the canonical runbook for bringing up a remote Linux/CUDA box for
EvacOS2 training.

Use this when training on Vast.ai or any similar remote GPU host.

## Why this file exists

The remote training stack is sensitive to install order and import order:

- `unsloth` should be installed after the baseline training deps
- `vllm==0.10.2` should be installed last
- `torch==2.10.0` and `torchvision==0.25.0` are the tested April 25
  2026 Vast stack for current Unsloth
- `transformers==4.56.2` and `peft==0.19.1` are the tested-compatible pair
  for the current Unsloth + vLLM setup
- `fsspec==2025.9.0` avoids the dataset cache conflict seen after PyTorch
  repairs
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

Current 3B specialist training rule:

- Keep the paid specialist path easy-only until the reward/grouping signal is
  proven healthy. The earlier broad curriculum hid non-learning behind noisy
  metrics, so do not launch broader runs from these configs.
- First run the `10`-step signal canary to verify same-prompt GRPO signal:
  `floor_agent_group_raw_reward_std_mean > 0`, `floor_agent_advantage_std > 0`,
  and non-zero `floor_agent_policy_loss`.
- Then run the `50`-step canary to verify parser, checkpoint, metrics CSV, and
  watchdog behavior over a slightly longer window.
- Then run the `300`-step proof config and evaluate held-out easy seeds. Continue
  only if the trained checkpoint beats baseline and the watchdog stays green.
- Do not treat "watchdog green" as enough by itself. Before a paid proof/quality
  run, confirm same-prompt grouped candidates are active: multiple completions
  for the same role/agent/prompt must share a prompt-scoped `group_id`, with
  non-zero `floor_agent_group_raw_reward_std_mean` and
  `floor_agent_advantage_std` in the metrics.
- The checked-in `*-750.yaml` configs now run `750` easy episodes for one
  disaster family. The `*-2000.yaml` quality configs run `2000` easy episodes.
- Treat watchdog triggers as stop signs, not warnings. In particular, abort on
  zero GRPO signal, high valid-but-hollow action rate, or scout-heavy floor
  collapse before spending more GPU time.
- Always save the final metrics CSV, `training_watchdog.jsonl`, held-out eval
  CSV/JSON/plots, and LoRA adapter artifact before destroying the instance.

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
pip uninstall -y torch torchvision transformers trl peft accelerate bitsandbytes vllm unsloth unsloth_zoo torchcodec >/dev/null 2>&1 || true
pip install "torch==2.10.0" "torchvision==0.25.0" "transformers==4.56.2" "trl==0.24.0" "peft==0.19.1" "accelerate" "bitsandbytes" "datasets" "fsspec==2025.9.0"
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
- Current Unsloth expects the `torch.int1` API, while `unsloth_zoo==2026.4.9`
  requires `torch<2.11`. Pinning `torch==2.10.0` is the safe middle ground.
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
print("has_torch_int1:", hasattr(torch, "int1"))
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

If the existing env fails with:

```text
AttributeError: module 'torch' has no attribute 'int1'
```

the smallest proven repair on the April 25 2026 Vast 4090 run was:

```bash
pip install --upgrade --force-reinstall "torch==2.10.0" "torchvision==0.25.0"
pip install "fsspec==2025.9.0"
```

Do not jump to `torch==2.11.*` for this stack unless Unsloth Zoo has also
relaxed its `<2.11` requirement.

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
