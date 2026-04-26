# EvacOS2 Submission Brief

EvacOS2 is a multi-agent evacuation RL environment where one orchestrator and multiple floor agents must coordinate people movement, exits, overrides, and hazard response inside a deterministic simulator.

Evidence status: the repo now includes a held-out `3B` specialist comparison: Qwen2.5-3B base/no-LoRA versus trained LoRA specialists on the same unseen seeds and evaluator. On this `30`-episode controlled proof slice, trained LoRA floor specialists improved the average eval score from `15.08%` to `36.28%` and reduced invalid actions from `51.81%` to `0.00%`. The `7B` orchestrator remains smoke/training-signal validated rather than claimed as a converged held-out policy.

Benchmark scope: the submitted fixed-suite specialist comparison is a controlled proof slice. That keeps the baseline-vs-trained result deterministic, judge-runnable, and directly auditable. The simulator and training configs are built for broader curricula; higher-difficulty scorecards are the next evaluation milestone after the submitted slice.

## Why this is a strong submission

- More demanding than single-turn tasks: decisions compound over repeated simulator rounds.
- More realistic than toy environments: the task is operational coordination under uncertainty, not just label prediction.
- More verifiable than judge-only scoring: rewards and evaluation are programmatic, with fixed-suite and baseline-vs-trained comparison support.
- More extensible than a one-model demo: the training stack supports both shared-role and split-role model configurations.
- More practical than a single monolith: the architecture can train fire/flood/gas specialists, including cheap `3B` floor-only lanes, while a scope router falls back to the generalist for mixed or cascading incidents.

## What judges can verify quickly

- Live environment API:
  - `/openenv/reset`
  - `/openenv/step`
  - `/openenv/state`
- Fixed evaluation path:
  - `python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline`
  - `python -m evaluation.demo_bundle --trained-checkpoint /path/to/downloaded/lora_adapter --config training/config.remote-unsloth-7b3b-split-bridge.yaml --output-dir outputs/demo_bundle`
- Headline artifacts:
  - `demo/results/heldout_3b_base_vs_trained_summary.md`
  - `demo/results/heldout_3b_base_vs_trained_summary.csv`
  - `demo/results/plots/heldout_3b_base_vs_trained_eval_score.png`
  - `demo/results/plots/heldout_3b_base_vs_trained_invalid_action_rate.png`
  - `demo/results/specialist_canary50_report.md`
  - `demo/results/plots/3b_specialist_valid_action_score_comparison.png`
  - `demo/results/plots/h200_resume200_raw_reward_progress.png`
  - `demo/results/plots/h200_resume200_norm_reward_progress.png`
  - `demo/results/plots/h200_resume200_invalid_action_progress.png`
  - `demo/results/7b_orchestrator_behavior_card.md`
  - `demo/results/submission_scorecard_baseline.md`
  - `demo/results/baseline_fixed_suite.csv`
  - Public canary adapters: `shashankN777/evacos2-7b-orchestrator-artifacts`

## Public adapter paths

The public artifact repo is historically named `evacos2-7b-orchestrator-artifacts`, but the visible submitted checkpoints are `3B` floor-specialist H200 canaries:

- fire: `floor-specialists/fire/h200-canary3-10/checkpoints/latest`
- flood: `floor-specialists/flood/h200-canary3-10/checkpoints/latest`
- gas: `floor-specialists/gas/h200-canary-10/checkpoints/latest`
- fire stronger seed: `floor-specialists/fire/vast-canary50/checkpoints/latest`
- flood stronger seed: `floor-specialists/flood/vast-canary50/checkpoints/latest`
- gas stronger seed: `floor-specialists/gas/vast-canary50/checkpoints/latest`
- latest fire continuation: `floor-specialists/fire/h200-resume200-from-vast50/checkpoints/ckpt_169`
- latest flood continuation: `floor-specialists/flood/h200-resume200-from-vast50/checkpoints/ckpt_139`
- latest gas continuation: `floor-specialists/gas/h200-resume200-from-vast50/checkpoints/ckpt_89`
- latest continuation manifest: `floor-specialists/h200-resume200-from-vast50-MANIFEST.json`
- held-out base-vs-trained eval: `heldout/base-model-vs-h200-resume200-3b-heldout10-batched-20260429-065816`

Download example:

```bash
hf download shashankN777/evacos2-7b-orchestrator-artifacts \
  --include "floor-specialists/fire/h200-canary3-10/**" \
  --include "floor-specialists/flood/h200-canary3-10/**" \
  --include "floor-specialists/gas/h200-canary-10/**" \
  --include "floor-specialists/fire/vast-canary50/**" \
  --include "floor-specialists/flood/vast-canary50/**" \
  --include "floor-specialists/gas/vast-canary50/**" \
  --include "floor-specialists/fire/h200-resume200-from-vast50/**" \
  --include "floor-specialists/flood/h200-resume200-from-vast50/**" \
  --include "floor-specialists/gas/h200-resume200-from-vast50/**" \
  --include "floor-specialists/h200-resume200-from-vast50-MANIFEST.json" \
  --include "heldout/base-model-vs-h200-resume200-3b-heldout10-batched-20260429-065816/**" \
  --local-dir outputs/hf_public_artifacts
```

Evaluation example:

```bash
python -m evaluation.demo_bundle \
  --baseline-policy base_model \
  --trained-checkpoint outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/checkpoints/latest \
  --config outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/generated.remote-unsloth-3b-fire-floor-specialist-h200-resume200-200.yaml \
  --training-metrics-path outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/remote-unsloth-3b-fire-floor-specialist-h200-resume200-200-metrics.csv \
  --output-dir outputs/demo_bundle_fire_h200_resume200
```

## Held-out base-vs-trained headline

On a `30`-episode held-out specialist evaluation, trained LoRA floor specialists more than doubled the base/no-LoRA Qwen2.5-3B average eval score and eliminated invalid actions across fire, flood, and gas response lanes.

| Family | Episodes | Base eval score | Trained eval score | Delta | Base invalid | Trained invalid |
|---|---:|---:|---:|---:|---:|---:|
| Fire | `10` | `11.58%` | `36.34%` | `+24.76 pp` | `60.00%` | `0.00%` |
| Flood | `10` | `17.63%` | `36.34%` | `+18.70 pp` | `46.67%` | `0.00%` |
| Gas | `10` | `16.02%` | `36.16%` | `+20.14 pp` | `48.75%` | `0.00%` |
| **Average** | `30 total` | **`15.08%`** | **`36.28%`** | **`+21.20 pp`** | **`51.81%`** | **`0.00%`** |

## Core competitive claims

- Real environment loop, not prompt wrapping
- Multi-agent coordination, not single-policy one-shot response
- Hierarchical model allocation: stronger orchestrator for long-horizon coordination, faster floor agents for local response
- Specialist-ready disaster routing: fire/flood/gas lanes can be trained independently, including `3B` floor-only local-response specialists, and selected deterministically
- Baseline scorecards plus canary learning evidence, not anecdotal samples only
- Reward-hacking safeguards, not one loose scalar reward
- Public H200 canary adapters are linked explicitly; no final quality-run success is claimed without checkpoint, log, and eval artifacts
- Controlled proof-slice framing: the submitted scorecard prioritizes reproducible base-vs-trained evidence, while broader difficulty generalization is future validation work

## Recommended demo order

1. Open the Hugging Face Space and show `/openenv/health`, `/openenv/metadata`, and `/openenv/schema`.
2. Show the 3B specialist canary score table and training plots.
3. Open `demo/results/7b_orchestrator_behavior_card.md` to explain the global coordinator role and its honest smoke/training-signal status.
4. Show the hierarchy: `7B` orchestrator, `3B` floor agents, and optional disaster-specialist routing.
5. Show one live OpenEnv interaction.

## Repo landmarks

- Project overview: [README.md](README.md)
- Demo bundle builder: [evaluation/demo_bundle.py](evaluation/demo_bundle.py)
- Fixed-suite evaluation: [evaluation/fixed_suite.py](evaluation/fixed_suite.py)
- Scope router: [training/scope_router.py](training/scope_router.py)
