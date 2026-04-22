# EvacOS2

EvacOS2 is an OpenEnv-style multi-agent evacuation RL system where a central orchestrator and floor agents must coordinate under pressure inside a deterministic building evacuation simulator. The repo combines a real environment, verifier-style rewards, GRPO-style post-training, and a judge-friendly evaluation/demo path.

For the fastest high-level submission read, start with [SUBMISSION_BRIEF.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/SUBMISSION_BRIEF.md).

## Why this stands out

- Multi-agent, not single-turn: one orchestrator coordinates multiple floor agents across repeated rounds.
- Long-horizon, not toy horizon: the policy must sustain good decisions over many simulator steps rather than solve a one-shot classification task.
- Verifiable, not vibe-scored: rewards and evaluation are programmatic, with fixed-suite and baseline-vs-trained comparison support.
- Real environment loop, not prompt wrapping: `/openenv/reset`, `/openenv/step`, and `/openenv/state` hit the live simulator.
- Role-aware training path: the repo supports both shared-model training and split-role setups such as `7B orchestrator + 3B floor agents`.

## Judge-Fast Facts

- Core challenge: coordinated evacuation under uncertainty, constrained exits, and evolving hazards
- Environment shape: deterministic simulator with multi-agent rounds and role-specific actions
- RL stack: GRPO-style rollout/training with role-aware policy routing
- Default training base: `Qwen/Qwen2.5-3B-Instruct`
- Upgrade path: role-specific model overrides via `model.orchestrator_base` and `model.floor_base`
- Demo proof path: fixed-suite baseline, fixed-suite trained run, then live OpenEnv interaction

## What is in this repo

- `evacos_ma/`: core simulator, multi-agent orchestration, reward plumbing, and OpenEnv-facing API
- `training/`: rollout collection, GRPO training loop, policy adapters, checkpointing, and backend integration
- `evaluation/`: fixed-suite verification, rationale sweeps, and baseline-vs-trained comparison helpers
- `notebooks/train_evacos_ma.ipynb`: end-to-end notebook for smoke runs, training, resume, and evaluation
- `dashboard/`: local demo dashboard for rollout and reward inspection
- `demo/`: presentation/story assets that sit alongside the executable demo surfaces

## Current technical shape

- Environment first, not just fine-tuning: the simulator and multi-agent round protocol are the core product.
- Verifiable rewards: reward shaping, belief audit scoring, counterfactual/oversight signals, and fixed-suite evaluation are all programmatic.
- TRL-style RL training: the repo uses a GRPO-family loop with role-aware grouping.
- OpenEnv surface: `/openenv/reset`, `/openenv/step`, and `/openenv/state` now hit the live simulator rather than a canned stub.

## Three-minute proof

If you want the fastest end-to-end evidence path:

1. Build a baseline-only evidence bundle:

   ```bash
   python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
   ```

2. Build a trained comparison bundle:

   ```bash
   python -m evaluation.demo_bundle --trained-checkpoint outputs/training/checkpoints/latest/lora_adapter --output-dir outputs/demo_bundle
   ```

3. Open the generated summary markdown and CSV, then show one live interaction through the OpenEnv API or dashboard.

This gives a compact baseline -> trained -> live-environment story without needing to inspect the full codebase first.
The bundle now also emits `submission_scorecard.md` and `submission_scorecard.json` so the first thing a judge sees can be a one-page headline artifact instead of raw logs.

## Training setup

The recommended serious training path is:

1. Train on Linux/CUDA with the checked-in default: `backend: "unsloth"` and `rollout.use_vllm: true`.
2. Use the notebook or `training/train.py` flow for smoke runs, checkpointed training, resume, and eval.
3. If you must do Windows-only local code/test work, override locally to the `hf` backend.
4. Use fixed-suite and baseline-vs-trained outputs as the primary before/after evidence.

The checked-in default base model is currently:

- shared model: `Qwen/Qwen2.5-3B-Instruct`

The config is now role-ready:

- `model.base` is the shared default
- `model.orchestrator_base` can later override only the orchestrator
- `model.floor_base` can later override floor agents

If both roles resolve to the same base, the repo preserves the current shared-model fast path.
If the roles resolve to different bases, the training/eval stack now routes orchestrator and floor-agent work through separate role-specific policies.

## Evaluation and demo path

- Fixed verification: [`evaluation/fixed_suite.py`](evaluation/fixed_suite.py)
- Before/after comparison: [`evaluation/baseline_vs_trained.py`](evaluation/baseline_vs_trained.py)
- Bundle builder: [`evaluation/demo_bundle.py`](evaluation/demo_bundle.py)
- Local API/demo surface: [`evacos_ma/openenv/server_shell.py`](evacos_ma/openenv/server_shell.py) and `dashboard/`
- Presentation assets: `demo/`

A strong demo flow for this repo is:

1. run baseline fixed-suite
2. run trained fixed-suite
3. build the bundle with `python -m evaluation.demo_bundle`
4. show the same scenario through the dashboard or OpenEnv surface

The key submission claim is not just that the model can emit plausible actions. It is that we can measure whether coordinated evacuation behavior actually improved.

For the hackathon-facing checklist and deployment flow, see [`HACKATHON.md`](HACKATHON.md).

## Secrets

Copy `.env.example` to `.env` and fill in only the values you need locally. The real `.env` file is gitignored.

## Status

The repo is beyond the original baseline bootstrap state:

- the repair campaign is complete
- the OpenEnv shell is wired to the real environment
- evaluation surfaces are in place
- split-role training support is in place for stronger orchestrator-vs-floor experiments
- the biggest remaining hackathon work is packaging polish: public docs, demo narrative, and Space deployment clarity
