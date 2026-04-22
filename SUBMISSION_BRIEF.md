# EvacOS2 Submission Brief

EvacOS2 is a multi-agent evacuation RL environment where one orchestrator and multiple floor agents must coordinate people movement, exits, overrides, and hazard response inside a deterministic simulator.

## Why this is a strong submission

- Harder than single-turn tasks: decisions compound over repeated simulator rounds.
- More realistic than toy environments: the task is operational coordination under uncertainty, not just label prediction.
- More verifiable than judge-only scoring: rewards and evaluation are programmatic, with fixed-suite and baseline-vs-trained comparison support.
- More extensible than a one-model demo: the training stack supports both shared-role and split-role model configurations.

## What judges can verify quickly

- Live environment API:
  - `/openenv/reset`
  - `/openenv/step`
  - `/openenv/state`
- Fixed evaluation path:
  - `python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline`
  - `python -m evaluation.demo_bundle --trained-checkpoint outputs/training/checkpoints/latest/lora_adapter --output-dir outputs/demo_bundle`
- Headline artifacts:
  - `outputs/demo_bundle/submission_scorecard.md`
  - `outputs/demo_bundle/demo_bundle_summary.md`
  - `outputs/demo_bundle/baseline_vs_trained.csv`

## Core competitive claims

- Real environment loop, not prompt wrapping
- Multi-agent coordination, not single-policy one-shot response
- Baseline-vs-trained evidence, not anecdotal samples only
- Reward-hacking safeguards, not one loose scalar reward

## Recommended demo order

1. Open `submission_scorecard.md`.
2. Show the baseline-vs-trained delta summary.
3. Explain the verifier-style reward stack and safeguards.
4. Show one live OpenEnv interaction.

## Repo landmarks

- Project overview: [README.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/README.md)
- Hackathon runbook: [HACKATHON.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/HACKATHON.md)
- Demo bundle builder: [evaluation/demo_bundle.py](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/demo_bundle.py)
- Fixed-suite evaluation: [evaluation/fixed_suite.py](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/fixed_suite.py)
