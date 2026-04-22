# Hackathon Runbook

This is the public-facing guide for turning the repo into a clean hackathon submission story.

For a one-screen judge-facing overview, see [SUBMISSION_BRIEF.md](/C:/Users/LENOVO/Specializations/Competitions/Scaler/SUBMISSION_BRIEF.md).

## What Should Impress Judges Fast

- This is a real environment, not a prompt-only wrapper.
- The task is multi-agent and long-horizon, so policy quality matters beyond formatting.
- The reward stack is programmatic and hard to game compared with a single loose scalar.
- The repo has a direct baseline-vs-trained evidence path rather than only anecdotal examples.
- The OpenEnv-facing API now runs on the live simulator, so the demo surface and the training surface are aligned.

## What The Project Already Has

- A real multi-agent evacuation environment in [`evacos_ma/env.py`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evacos_ma/env.py)
- A live OpenEnv-style API surface in [`evacos_ma/openenv/server_shell.py`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evacos_ma/openenv/server_shell.py)
- GRPO-style training and rollout collection in [`training/train.py`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/training/train.py)
- Programmatic evaluation in [`evaluation/fixed_suite.py`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/fixed_suite.py)
- Before/after comparison in [`evaluation/baseline_vs_trained.py`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/evaluation/baseline_vs_trained.py)

## Reward Stack And Safeguards

The reward channel is not a single scalar proxy. It combines multiple verifier-like signals:

- environment/base simulation reward
- team progress terms
- oversight / counterfactual deltas for orchestrator interventions
- directive quality / coordination terms
- floor invalid-action penalties
- belief audit scoring for prediction actions
- rationale bonuses that are gated by actual evidence

Safeguards against reward hacking already present in the repo:

- structured action validation by role
- invalid-action penalties instead of trusting raw generations
- belief registration and duplicate-belief handling
- counterfactual gating on orchestrator rationale bonus
- belief-score gating on floor rationale bonus
- fixed-suite evaluation separated from training-time reward flow
- debug-state gating on `/openenv/state` via `EVACOS_DEBUG_STATE`

## Training Recommendation

For serious training, prefer Linux/CUDA:

1. keep the checked-in default `backend: "unsloth"` in [`training/config.yaml`](/C:/Users/LENOVO/Specializations/Competitions/Scaler/training/config.yaml)
2. keep the checked-in default `rollout.use_vllm: true`
3. train with the notebook or `python -m training.train`

Current checked-in default:

- shared base model: `Qwen/Qwen2.5-3B-Instruct`

Role-aware config is already prepared:

- `model.base`
- `model.orchestrator_base`
- `model.floor_base`

Role-specific training is now supported. If the resolved orchestrator and floor bases differ, the training loop:

- loads separate role-specific policies
- routes rollout generation by role
- trains separate role-specific GRPO trainers
- checkpoints separate adapter directories per role

Shared-role mode still works the old way and keeps the cheaper shared fast path.

Example split config:

```yaml
model:
  base: "Qwen/Qwen2.5-3B-Instruct"
  orchestrator_base: "Qwen/Qwen2.5-7B-Instruct"
  floor_base: "Qwen/Qwen2.5-3B-Instruct"
```

## Build A Demo Bundle

Baseline-only bundle:

```bash
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
```

Baseline-vs-trained bundle:

```bash
python -m evaluation.demo_bundle --trained-checkpoint outputs/training/checkpoints/latest/lora_adapter --output-dir outputs/demo_bundle
```

This writes:

- `baseline_vs_trained.csv`
- fixed-suite JSON artifacts
- summary markdown
- submission scorecard markdown/json
- plots when `matplotlib` is installed

For split-role checkpoints, `latest/lora_adapter/` should contain:

- `orchestrator/`
- `floor_agent/`

## Local OpenEnv Run

Run the API locally:

```bash
uvicorn evacos_ma.api:app --host 0.0.0.0 --port 8000
```

Minimal client flow:

1. `POST /openenv/reset`
2. `POST /openenv/step`
3. `GET /openenv/state?episode_id=...`

If you need full internal state for debugging, set:

```bash
EVACOS_DEBUG_STATE=true
```

Keep that off for normal demos.

## Suggested Judge Demo

1. Show the environment and role split.
2. Open `submission_scorecard.md` first and anchor the audience on the headline metrics.
3. Show trained metrics and deltas from `demo_bundle_summary.md`.
4. Show one live OpenEnv interaction.
5. Explain why the reward stack is hard to game.

## What We Need To Beat Simpler Submissions

- Win on clarity, not just depth: a judge should understand the task, verifier, and proof path from one README pass.
- Win on evidence: always prefer baseline-vs-trained outputs over qualitative claims.
- Win on realism: keep emphasizing that coordinated evacuation is a multi-agent operational problem, not a single-turn classifier.
- Win on safety against reward hacking: point to verifier separation, invalid-action penalties, counterfactual gates, and fixed-suite evaluation.
- Win on final presentation: keep the demo path short, repeatable, and visibly tied to measurable improvement.
