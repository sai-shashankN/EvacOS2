# HuggingFace Mini-Blog Scaffold

## Problem
Evacuation planning is a hierarchical coordination problem. We frame it as one orchestrator supervising five floor agents across procedurally generated emergency scenarios.

## Setup
- Agents: 1 orchestrator + 5 floor agents
- Environment: Procgen building generator with fixed evaluation seeds
- Disaster families: fire, flood, gas, structural, active threat, multi-cascade

## Training
- Method: GRPO + LoRA
- Policy stack: `training.policy_adapter.hf_policy_factory(...)`
- Evaluation seeds: `42, 123, 456, 789, 1024`

## Results
- Baseline normalized reward: `TODO(from outputs/evals/baseline_vs_trained.csv)`
- Trained normalized reward: `TODO(from outputs/evals/baseline_vs_trained.csv)`
- Delta: `TODO(from outputs/evals/baseline_vs_trained.csv)`
- Invalid action rate: `TODO(from outputs/evals/baseline_vs_trained.csv)`
- Override win rate: `TODO(from outputs/evals/baseline_vs_trained.csv)`

## Reproducibility
- Fixed seed suite: `42, 123, 456, 789, 1024`
- Generator config hash: `TODO(from outputs/evals/fixed_suite_*.json)`
- Rationale mode: `TODO(from outputs/evals/rationale_sweep.json)`

## Assets
- Dashboard: `dashboard/`
- Plots: `outputs/evals/plots/`
- Renderer bridge: `renderer/unity_bridge.py`
