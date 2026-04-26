# EvacOS2 Submission Brief

EvacOS2 is a multi-agent evacuation RL environment where one orchestrator and multiple floor agents must coordinate people movement, exits, overrides, and hazard response inside a deterministic simulator.

Evidence status: the tracked fixed-suite scorecard is baseline-only; the tracked learning evidence is the fire/flood/gas `3B` canary and training-signal artifact trail. A full trained held-out comparison is supported after restoring a selected LoRA adapter checkpoint.

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
  - `demo/results/specialist_canary50_report.md`
  - `demo/results/plots/3b_specialist_valid_action_score_comparison.png`
  - `demo/results/7b_orchestrator_behavior_card.md`
  - `demo/results/submission_scorecard_baseline.md`
  - `demo/results/baseline_fixed_suite.csv`
  - Public canary adapters: `shashankN777/evacos2-7b-orchestrator-artifacts`

## Public adapter paths

The public artifact repo is historically named `evacos2-7b-orchestrator-artifacts`, but the visible submitted checkpoints are `3B` floor-specialist H200 canaries:

- fire: `floor-specialists/fire/h200-canary3-10/checkpoints/latest`
- flood: `floor-specialists/flood/h200-canary3-10/checkpoints/latest`
- gas: `floor-specialists/gas/h200-canary-10/checkpoints/latest`

Download example:

```bash
hf download shashankN777/evacos2-7b-orchestrator-artifacts \
  --include "floor-specialists/fire/h200-canary3-10/**" \
  --include "floor-specialists/flood/h200-canary3-10/**" \
  --include "floor-specialists/gas/h200-canary-10/**" \
  --local-dir outputs/hf_public_artifacts
```

Evaluation example:

```bash
python -m evaluation.demo_bundle \
  --trained-checkpoint outputs/hf_public_artifacts/floor-specialists/fire/h200-canary3-10/checkpoints/latest \
  --config outputs/hf_public_artifacts/floor-specialists/fire/h200-canary3-10/generated.remote-unsloth-3b-fire-floor-specialist-h200-canary3-10.yaml \
  --training-metrics-path outputs/hf_public_artifacts/floor-specialists/fire/h200-canary3-10/remote-unsloth-3b-fire-floor-specialist-h200-canary3-10-metrics.csv \
  --output-dir outputs/demo_bundle_fire_h200_canary
```

## Core competitive claims

- Real environment loop, not prompt wrapping
- Multi-agent coordination, not single-policy one-shot response
- Hierarchical model allocation: stronger orchestrator for long-horizon coordination, faster floor agents for local response
- Specialist-ready disaster routing: fire/flood/gas lanes can be trained independently, including `3B` floor-only local-response specialists, and selected deterministically
- Baseline scorecards plus canary learning evidence, not anecdotal samples only
- Reward-hacking safeguards, not one loose scalar reward
- Public H200 canary adapters are linked explicitly; no final quality-run success is claimed without checkpoint, log, and eval artifacts

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
