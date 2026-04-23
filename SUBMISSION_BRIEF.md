# EvacOS2 Submission Brief

EvacOS2 is a multi-agent evacuation RL environment where one orchestrator and multiple floor agents must coordinate people movement, exits, overrides, and hazard response inside a deterministic simulator.

## Why this is a strong submission

- Harder than single-turn tasks: decisions compound over repeated simulator rounds.
- More realistic than toy environments: the task is operational coordination under uncertainty, not just label prediction.
- More verifiable than judge-only scoring: rewards and evaluation are programmatic, with fixed-suite and baseline-vs-trained comparison support.
- More extensible than a one-model demo: the training stack supports both shared-role and split-role model configurations.
- More practical than a single monolith: the architecture can train fire/flood/gas specialists while a scope router falls back to the generalist for mixed or cascading incidents.

## What judges can verify quickly

- Live environment API:
  - `/openenv/reset`
  - `/openenv/step`
  - `/openenv/state`
- Fixed evaluation path:
  - `python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline`
  - `python -m evaluation.demo_bundle --trained-checkpoint outputs/training/checkpoints/latest/lora_adapter --config training/config.remote-unsloth-7b3b-split-bridge.yaml --output-dir outputs/demo_bundle`
- Headline artifacts:
  - `outputs/demo_bundle/submission_scorecard.md`
  - `outputs/demo_bundle/demo_bundle_summary.md`
  - `outputs/demo_bundle/baseline_vs_trained.csv`
  - If `outputs/demo_bundle` is absent, the checked-in bundle is baseline-only and the trained scorecard still needs to be generated from a selected checkpoint.

## Core competitive claims

- Real environment loop, not prompt wrapping
- Multi-agent coordination, not single-policy one-shot response
- Hierarchical model allocation: stronger orchestrator for long-horizon coordination, faster floor agents for local response
- Specialist-ready disaster routing: fire/flood/gas lanes can be trained independently and selected deterministically
- Baseline-vs-trained evidence, not anecdotal samples only
- Reward-hacking safeguards, not one loose scalar reward

## Recommended demo order

1. Open `submission_scorecard.md`.
2. Show the baseline-vs-trained delta summary.
3. Explain the verifier-style reward stack and safeguards.
4. Show the hierarchy: `7B` orchestrator, `3B` floor agents, and optional disaster-specialist routing.
5. Show one live OpenEnv interaction.

## Repo landmarks

- Project overview: [README.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/README.md)
- Hackathon runbook: [HACKATHON.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/HACKATHON.md)
- Demo bundle builder: [evaluation/demo_bundle.py](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/demo_bundle.py)
- Fixed-suite evaluation: [evaluation/fixed_suite.py](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/fixed_suite.py)
- Scope router: [training/scope_router.py](/C:/Users/LENOVO/Specializations/Competitions/Scaler/training/scope_router.py)
