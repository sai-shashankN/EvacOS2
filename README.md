# EvacOS2

EvacOS2 is a hierarchical multi-agent evacuation training project built around a deterministic evacuation simulator, GRPO-based reinforcement learning, and a Colab-friendly training workflow.

## What is in this repo

- `evacos_ma/`: core environment, agents, oracle, and package code
- `training/`: training loop, rewards, checkpointing, and backend integration
- `notebooks/train_evacos_ma.ipynb`: end-to-end Colab notebook for smoke runs, training, resume, and evaluation
- `dashboard/`: local demo dashboard for live reward and rollout inspection
- `demo/`: blog/storyboard assets for the final demo package

## Training setup

The recommended path is:

1. Use Google Colab for training.
2. Enable Unsloth and vLLM in the notebook for faster rollout throughput.
3. Save checkpoints to Google Drive so training can resume across Colab restarts.
4. Push the final LoRA adapter to Hugging Face Hub.
5. Use Hugging Face Spaces for the demo surface, not for training.

## Secrets

Copy `.env.example` to `.env` and fill in only the values you need locally. The real `.env` file is gitignored.

## Status

This repository is set up for local development plus Colab-based RL training. The next recommended steps are:

- publish the repo to GitHub
- add a Hugging Face write token locally
- run the Colab smoke test
- start the first short training run
