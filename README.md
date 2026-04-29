---
title: EvacOS2 OpenEnv
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# EvacOS2

EvacOS2 is a hierarchical multi-agent evacuation simulator for training agents under fire, flood, and gas emergencies.

It also serves as a benchmark for calibrated autonomy in agent teams. The evacuation domain gives the benchmark concrete stakes: partial observability, evolving hazards, bottlenecked exits, delayed consequences, and multiple agents acting at once. Beneath that domain, EvacOS2 tests when local agents should act independently, when they should escalate to an orchestrator, when the orchestrator should intervene, and whether the same model can behave differently across roles.

In other words, EvacOS2 asks not only:

> Can the agents solve the evacuation?

but also:

> Can each agent decide whether it should be the one solving the problem?

## Judge-Fast Summary

| Aspect | Detail |
|---|---|
| **Domain** | Emergency building evacuation under uncertainty, constrained exits, and evolving hazards |
| **Topology** | `1` orchestrator agent + `5` floor agents |
| **Hierarchy** | `7B` orchestrator for long-horizon coordination, `3B` floor agents for faster local decisions |
| **Benchmark focus** | Calibrated autonomy: local action, escalation, intervention, fallback, and same-model role discipline |
| **Interface** | OpenEnv-compatible live endpoints backed by the simulator, with fixed-task and procedural disaster-family resets |
| **Endpoints** | `/openenv/reset`, `/openenv/step`, `/openenv/state`, `/openenv/schema`, `/openenv/health`, `/openenv/metadata` |
| **Training/eval scope** | controlled fixed-suite fire, flood, and gas specialist response lanes |
| **Training** | Unsloth + LoRA + GRPO-style training, with shared-model and split-role support |
| **Specialization path** | Optional `fire`, `flood`, and `gas` specialist configs with a deterministic scope router driven by observed incident metadata |
| **Evaluation** | Fixed-suite verification, baseline-vs-trained comparison, scorecards, and plots |
| **Validated stronger configs** | `7B` single-model smoke, `7B + vLLM` smoke, `7B/3B` split-role smoke |
| **Metrics support** | Aggregate diagnostics, per-role diagnostics, and checkpoint-local metrics snapshots for cleaner plots |

**Bottom line:** real simulator, real training loop, real evaluation pipeline. This is not a prompt wrapper or a config-only stub.

## Submission Links

| Required field | Public URL |
|---|---|
| Hugging Face Space URL for Env | [https://huggingface.co/spaces/shashankN777/evacos2-openenv](https://huggingface.co/spaces/shashankN777/evacos2-openenv) |
| Training run notebook URL | [notebooks/train_evacos_ma.ipynb](https://huggingface.co/spaces/shashankN777/evacos2-openenv/blob/main/notebooks/train_evacos_ma.ipynb) |
| Blog post URL | [Blog.MD](https://huggingface.co/spaces/shashankN777/evacos2-openenv/blob/main/Blog.MD) |
| Public model/artifact repo | [shashankN777/evacos2-7b-orchestrator-artifacts](https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts) |
| Source code on GitHub | [sai-shashankN/EvacOS2](https://github.com/sai-shashankN/EvacOS2) |

**Evidence status:** the repo now includes a judge-clean held-out `3B` specialist comparison: Qwen2.5-3B base/no-LoRA versus the trained LoRA specialists on the same unseen seeds, same evaluator, and same family-specific lanes. On this `30`-episode controlled proof slice, trained LoRA floor specialists improved the average eval score from `15.08%` to `36.28%` and reduced invalid actions from `51.81%` to `0.00%`. The `7B` orchestrator remains smoke/training-signal validated rather than claimed as a converged held-out policy.

**Benchmark scope note:** the submitted specialist proof lane uses a controlled fixed-suite slice so the public baseline-vs-trained comparison is deterministic, cheap enough to rerun, and easy to audit. The simulator and training stack are designed for broader curricula; expanding the same fixed-suite evaluator to higher-difficulty slices is future evaluation work, not a change to the environment architecture.

## Validated Evidence

| Component | Configuration | Status | Evidence |
|---|---|---|---|
| Environment + OpenEnv shell | Deterministic evacuation simulator with `1+5` agent topology | Validated | [openenv.yaml](openenv.yaml), [evacos_ma/](evacos_ma) |
| Shared-model training | `Qwen/Qwen2.5-3B-Instruct` | Checked in | [training/config.yaml](training/config.yaml) |
| Stronger single-model lane | `Qwen/Qwen2.5-7B-Instruct` | Smoke validated on larger GPUs | [training/](training) |
| vLLM lane | `7B + vLLM` | Smoke validated on larger GPUs | [training/](training) |
| Split-role lane | `7B` orchestrator / `3B` floor agents | Smoke validated on larger GPUs | [training/config.remote-unsloth-7b3b-split-bridge.yaml](training/config.remote-unsloth-7b3b-split-bridge.yaml) |
| Split-role disaster specialist lanes | `7B/3B` configs scoped to `fire`, `flood`, or `gas` | Implemented / ready to run | [training/config.remote-unsloth-7b3b-fire-specialist.yaml](training/config.remote-unsloth-7b3b-fire-specialist.yaml), [training/config.remote-unsloth-7b3b-flood-specialist.yaml](training/config.remote-unsloth-7b3b-flood-specialist.yaml), [training/config.remote-unsloth-7b3b-gas-specialist.yaml](training/config.remote-unsloth-7b3b-gas-specialist.yaml) |
| Floor-only 3B specialist lanes | Deterministic stub orchestrator + trainable `3B` floor policy for `fire`, `flood`, or `gas` | Implemented / canary and quality-run configs ready | [training/config.remote-unsloth-3b-fire-floor-specialist-signal-canary-10.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-signal-canary-10.yaml), [training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml), [training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml), [training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml](training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml), [training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml](training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml) |
| Specialist routing | Deterministic single-disaster router with generalist fallback for mixed/cascade scenarios | Implemented | [training/scope_router.py](training/scope_router.py) |
| H200 / HF Jobs specialist artifacts | Public fire/flood/gas `3B` floor-specialist canary adapters plus logs and metrics | Public canary artifact trail is linked below; longer quality-run configs are checked in but not claimed as final results | [training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml), [training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml](training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml), [training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml](training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml) |
| Held-out `3B` specialist comparison | Base Qwen2.5-3B/no-LoRA vs trained LoRA specialists on unseen seeds | Verified on H200 | [summary](demo/results/heldout_3b_base_vs_trained_summary.md), [CSV](demo/results/heldout_3b_base_vs_trained_summary.csv), [eval plot](demo/results/plots/heldout_3b_base_vs_trained_eval_score.png), [invalid-action plot](demo/results/plots/heldout_3b_base_vs_trained_invalid_action_rate.png) |
| Split-role metrics | Aggregate + per-role CSV diagnostics, with `metrics_window.csv`, `metrics_to_date.csv`, and `metrics_summary.json` saved beside each checkpoint | Verified | [training/metrics.py](training/metrics.py), [training/train.py](training/train.py) |
| Checkpoint + resume | LoRA adapters, optimizer state, RNG state | Implemented | [training/checkpoint.py](training/checkpoint.py) |
| Evaluation bundle | Fixed suite, comparison, scorecards, plots | Implemented | [evaluation/demo_bundle.py](evaluation/demo_bundle.py), [evaluation/plots.py](evaluation/plots.py) |

Here, **smoke validated** means a capped end-to-end run completed model load, rollout, training, and checkpointing on larger GPUs.

## Why It Matters

Most multi-agent demos stop at showing that agents can talk to each other. The more demanding question is whether post-training measurably improves coordination. Evacuation under uncertainty is a concrete testbed for that question: hazards evolve, exits bottleneck, and no single agent sees the full building.

EvacOS2 is built around three contributions:

1. **A reproducible benchmark.** Deterministic simulator and fixed evaluation suites make runs comparable instead of anecdotal.
2. **Role-aware training.** Shared-model and split-role paths are both supported, including a validated `7B` orchestrator / `3B` floor-agent lane with per-role diagnostics so you can see which role improved and which did not.
3. **Verifiable evaluation.** Baseline-vs-trained comparison, scorecards, and plots come from real rollouts and checkpoints, not hand-curated examples.
4. **A practical specialization path.** The same environment can train a mixed-disaster generalist or separate `fire`, `flood`, and `gas` specialists, then route scenarios through a deterministic scope layer using the incident type that would realistically come from building alarms, sensors, or dispatch reports.

## Benchmark Modes

EvacOS2 can be evaluated in three role configurations:

1. **Orchestrator benchmark:** keep floor agents fixed and vary the orchestrator. This tests when the orchestrator should intervene, when it should trust local responders, how it handles outliers, and whether its overrides improve the whole evacuation rather than merely adding control.
2. **Floor-agent benchmark:** keep the orchestrator fixed and vary the floor agents. This tests instruction following, local decision quality, escalation timing, behavior when the orchestrator is unreachable, and whether a floor agent understands when local autonomy helps or harms the team.
3. **Same-model role benchmark:** use the same model for both orchestrator and floor-agent roles. This tests whether a model can preserve role discipline despite equal intelligence: acting locally when appropriate, escalating when the team benefits, and avoiding both selfish autonomy and blind deference.

The deeper target is not hierarchy for its own sake. It is controlled interdependence: agents that can work independently, but know when the safer and more useful decision is to route context upward for the whole team.

## Architecture

```mermaid
graph LR
    subgraph "Deterministic Simulator"
        O["Orchestrator"]
        F1["Floor Agent 1"]
        F2["Floor Agent 2"]
        F3["Floor Agent 3"]
        F4["Floor Agent 4"]
        F5["Floor Agent 5"]
        O --> F1
        O --> F2
        O --> F3
        O --> F4
        O --> F5
    end

    subgraph "OpenEnv API"
        R["/reset"]
        P["/step"]
        S["/state"]
        C["/schema"]
        H["/health"]
        M["/metadata"]
    end

    subgraph "Training Stack"
        U["Unsloth Backend"]
        G["GRPO Trainer"]
        L["LoRA Adapters"]
        U --> G --> L
    end

    subgraph "Evaluation"
        FS["Fixed Suite"]
        BVT["Baseline vs Trained"]
        SC["Scorecards + Plots"]
        FS --> BVT --> SC
    end

    O --> P
    F1 --> P
    F2 --> P
    F3 --> P
    F4 --> P
    F5 --> P
    P --> G
    L --> BVT
```

## Hierarchical Specialist Strategy

EvacOS2 supports two complementary training stories:

- **Generalist bridge:** the existing `7B/3B` split-role config samples `fire`, `flood`, and `gas` episodes in one run. This is the most ambitious setting because the same policy stack must learn cross-disaster behavior.
- **Split-role specialist lanes:** the `7B/3B` fire/flood/gas configs keep the same hierarchy but restrict rollouts to one disaster family. These are useful when we want the orchestrator and floor agents to adapt together inside one disaster mode.
- **Floor-only specialist lanes:** the `3B` fire/flood/gas configs use a deterministic stub orchestrator and train only the floor-agent policy. These are cheaper parallel lanes for quickly learning local fire, flood, and gas evacuation behavior before a later shared orchestrator is trained against the frozen specialists.

[training/scope_router.py](training/scope_router.py) is a deterministic routing helper for placing in front of trained specialist checkpoints: it maps single-family scenarios to the matching specialist lane and falls back to the generalist for unknown, mixed, structural, active-threat, or cascade cases. The fixed-suite evaluator records the selected route for each episode and can pass it to a scope-aware policy factory. That gives the demo a clean story without overclaiming specialist training: fast local floor agents handle immediate routing, the stronger orchestrator handles slower global coordination, and the scope layer decides which disaster-specific policy lane to use once those checkpoints are trained.

This routing is meant to model realistic incident classification, not hidden omniscience. In a real deployment, fire mode can be raised by smoke/heat/fire-alarm signals, flood mode by water or moisture sensors plus facility reports, and gas mode by gas/CO/air-quality detectors or manual dispatch metadata. EvacOS2 represents that already-known incident context as scenario metadata, then uses it to select the matching frozen floor specialist while the `7B` orchestrator focuses on building-level coordination.

The `7B` orchestrator is included as a proof of concept for a practical multi-agent workflow: it does not need to slow down every local floor decision. Fast `3B` specialists handle routine floor-level routing, while the larger coordinator can be reserved for global priorities, cross-floor conflicts, stalled evacuations, sensor disagreement, and outlier scenarios that do not fit a single specialist lane. This creates a natural future path toward self-healing coordination: specialist fallback, anomaly escalation, policy repair suggestions, and human-in-the-loop override review can sit above the fast local responders without replacing them.

## What Smoke Testing Showed

The validated stronger configurations each completed:

- model load and multi-agent initialization across the `1+5` topology
- multi-step rollouts over the live environment
- GRPO training steps with LoRA checkpoint persistence
- aggregate and per-role metrics emission to CSV

The split-role lane (`7B` orchestrator / `3B` floor agents) emits separate `orchestrator_loss` and `floor_agent_loss` diagnostics, confirming that the training stack tracks each role independently rather than only globally.

These smoke runs prove computational fit and end-to-end training integrity. The public submission artifact trail includes H200 canary adapters, logs, and metrics for the floor-specialist lanes; held-out trained-vs-baseline claims should only be made from selected checkpoints, logs, and eval artifacts.

## Submission Artifacts

The repo now includes lightweight, Git-tracked artifacts for reviewers. Large LoRA adapters and raw logs remain outside Git by design.

| Artifact | Current status | What will land here |
|---|---|---|
| **H200 / HF Jobs specialist artifacts** | Public canary trail | Fire/flood/gas H200 canary adapters, logs, and metrics are hosted on Hugging Face; longer quality-run configs are checked in but not claimed as final results |
| **Held-out `3B` specialist comparison** | Tracked + public artifact repo | [held-out summary](demo/results/heldout_3b_base_vs_trained_summary.md), [CSV](demo/results/heldout_3b_base_vs_trained_summary.csv), and plots show base/no-LoRA vs trained LoRA on the same unseen seeds |
| **Fixed-suite baseline evidence** | Tracked | [baseline CSV](demo/results/baseline_fixed_suite.csv), [scorecard](demo/results/submission_scorecard_baseline.md), [plots](demo/results/plots) |
| **`3B` specialist canaries** | Tracked | [canary report](demo/results/specialist_canary50_report.md), [score CSV](demo/results/3b_specialist_canary50_scores.csv), and checkpoint plots covering fire/flood/gas route validity, invalid-action reduction, checkpoints, and runtime |
| **`7B` orchestrator behavior card** | Tracked | [behavior card](demo/results/7b_orchestrator_behavior_card.md) explains the orchestrator role, split-role smoke evidence, one trace, and the remaining held-out eval gap |
| **Hugging Face Space / live demo surface** | Deployed | [evacos2-openenv](https://huggingface.co/spaces/shashankN777/evacos2-openenv) exposes the canonical `/openenv/*` API surface |
| **Walkthrough video** | External link slot | Link the final public video in this row and in the submission form; draft flow lives in [demo/storyboard.md](demo/storyboard.md) |
| **Hugging Face blog / write-up** | Drafted | Root write-up lives in [Blog.MD](Blog.MD) and is mirrored into the Hugging Face Space |

## Public Adapter Artifacts

The public adapter repository is [shashankN777/evacos2-7b-orchestrator-artifacts](https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts). Despite the historical repository name, the visible public artifacts here are primarily `3B` floor-specialist checkpoints and evidence. The `7B` orchestrator path is documented separately as smoke/training-signal validated, not as a final converged orchestrator policy.

| Specialist | Public checkpoint path | Run type |
|---|---|---|
| Fire floor specialist | `floor-specialists/fire/h200-canary3-10/checkpoints/latest` | `10`-step H200 canary |
| Flood floor specialist | `floor-specialists/flood/h200-canary3-10/checkpoints/latest` | `10`-step H200 canary |
| Gas floor specialist | `floor-specialists/gas/h200-canary-10/checkpoints/latest` | `10`-step H200 canary |
| Fire floor specialist | `floor-specialists/fire/vast-canary50/checkpoints/latest` | `50`-step validity-stabilized Vast canary |
| Flood floor specialist | `floor-specialists/flood/vast-canary50/checkpoints/latest` | `50`-step validity-stabilized Vast canary |
| Gas floor specialist | `floor-specialists/gas/vast-canary50/checkpoints/latest` | `50`-step validity-stabilized Vast canary |
| Fire floor specialist | `floor-specialists/fire/h200-resume200-from-vast50/checkpoints/ckpt_169` | H200 continuation resumed from Vast `ckpt_49` |
| Flood floor specialist | `floor-specialists/flood/h200-resume200-from-vast50/checkpoints/ckpt_139` | H200 continuation resumed from Vast `ckpt_49` |
| Gas floor specialist | `floor-specialists/gas/h200-resume200-from-vast50/checkpoints/ckpt_89` | H200 continuation resumed from Vast `ckpt_49` |

Download the public specialist artifacts with:

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
  --local-dir outputs/hf_public_artifacts
```

Example trained-checkpoint evaluation command after download:

```bash
CHECKPOINT_DIR=outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/checkpoints/latest
CONFIG_PATH=outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/generated.remote-unsloth-3b-fire-floor-specialist-h200-resume200-200.yaml
METRICS_PATH=outputs/hf_public_artifacts/floor-specialists/fire/h200-resume200-from-vast50/remote-unsloth-3b-fire-floor-specialist-h200-resume200-200-metrics.csv

python -m evaluation.demo_bundle \
  --baseline-policy base_model \
  --trained-checkpoint "$CHECKPOINT_DIR" \
  --config "$CONFIG_PATH" \
  --training-metrics-path "$METRICS_PATH" \
  --output-dir outputs/demo_bundle_fire_h200_resume200
```

For the checked-in base-vs-trained held-out specialist comparison, use [demo/results/heldout_3b_base_vs_trained_summary.md](demo/results/heldout_3b_base_vs_trained_summary.md). For the tracked learning-signal summary, use [demo/results/specialist_canary50_report.md](demo/results/specialist_canary50_report.md) and [demo/results/3b_specialist_canary50_scores.csv](demo/results/3b_specialist_canary50_scores.csv).

## Held-Out Base-vs-Trained Result

This is the clearest reward-improvement evidence: same held-out seeds, same fixed-suite evaluator, same family-specific lane. The baseline is `Qwen/Qwen2.5-3B-Instruct` with no LoRA adapter and the same stub orchestrator used by the specialist training lane. The trained policy is the latest mirrored `h200-resume200-from-vast50` LoRA checkpoint.

For submission clarity, this table is reported on the controlled specialist proof slice rather than the full simulator difficulty ceiling. That choice keeps the public result reproducible and directly comparable across base, checkpoint, and trained policies; harder benchmark slices are the next validation milestone using the same evaluator shape.

Headline: on a `30`-episode held-out specialist evaluation, trained LoRA floor specialists more than doubled the base/no-LoRA Qwen2.5-3B average eval score, from `15.08%` to `36.28%`, while eliminating invalid actions across all fire, flood, and gas response lanes.

| Family | Held-out episodes | Base eval score | Trained eval score | Delta | Base invalid rate | Trained invalid rate |
|---|---:|---:|---:|---:|---:|---:|
| Fire | `10` | `11.58%` | `36.34%` | `+24.76 pp` | `60.00%` | `0.00%` |
| Flood | `10` | `17.63%` | `36.34%` | `+18.70 pp` | `46.67%` | `0.00%` |
| Gas | `10` | `16.02%` | `36.16%` | `+20.14 pp` | `48.75%` | `0.00%` |
| **Average** | `30 total` | **`15.08%`** | **`36.28%`** | **`+21.20 pp`** | **`51.81%`** | **`0.00%`** |

Public artifact path: `heldout/base-model-vs-h200-resume200-3b-heldout10-batched-20260429-065816` in [shashankN777/evacos2-7b-orchestrator-artifacts](https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts).

![Held-out 3B base-vs-trained eval score](demo/results/plots/heldout_3b_base_vs_trained_eval_score.png)

![Held-out 3B base-vs-trained invalid action rate](demo/results/plots/heldout_3b_base_vs_trained_invalid_action_rate.png)

## 3B Specialist Training Evidence

The cleanest completed specialist evidence is the `50`-step fire/flood/gas floor-agent canary suite. These are not final convergence claims; they are checkpointed proof that the `3B` local responders receive valid observations, produce valid route actions, preserve target IDs, and receive non-zero GRPO contrast.

| Specialist | Start valid-action score | Trained checkpoint score | Delta | Last-10 invalid rate | Last-10 GRPO reward std |
|---|---:|---:|---:|---:|---:|
| Fire `3B` | `83.65%` | `96.54%` | `+12.89 pp` | `3.46%` | `0.7168` |
| Flood `3B` | `88.46%` | `97.54%` | `+9.08 pp` | `2.46%` | `1.3848` |
| Gas `3B` | `89.62%` | `97.13%` | `+7.51 pp` | `2.87%` | `0.9769` |

`valid-action score = 100 * (1 - invalid_action_rate)`. The start score is step `0`; the trained checkpoint score is the last-10 average ending at checkpoint `ckpt_49`.

The continuation path resumes from those `ckpt_49` checkpoints and saves/upload-checkpoints every `10` steps. The snapshot below comes from uploaded H200 continuation metrics, so it is a reward-signal trend, not a final held-out scorecard.

| Specialist | Uploaded continuation steps | First-10 invalid rate | Last-10 invalid rate | First-10 raw reward | Last-10 raw reward | Reading |
|---|---:|---:|---:|---:|---:|---|
| Fire `3B` | `50 -> 169` | `1.44%` | `0.00%` | `10.56` | `11.59` | reward up, invalid actions eliminated in the latest window |
| Flood `3B` | `50 -> 139` | `3.69%` | `0.00%` | `7.60` | `8.68` | reward up, invalid actions eliminated in the latest window |
| Gas `3B` | `50 -> 89` | `0.19%` | `0.15%` | `29.40` | `30.44` | reward up, invalid actions remain near-zero |

Training reward is intentionally diagnostic and contrastive for GRPO. Public evaluation is separate: it reports bounded metrics such as valid-action score, invalid-action rate, saved/lost outcomes, and baseline-vs-trained deltas.

![3B specialist valid-action score comparison](demo/results/plots/3b_specialist_valid_action_score_comparison.png)

![3B specialist invalid action rate across checkpoints](demo/results/plots/3b_specialist_invalid_action_checkpoints.png)

![3B specialist raw reward across checkpoints](demo/results/plots/3b_specialist_raw_reward_checkpoints.png)

![H200 continuation raw reward progress](demo/results/plots/h200_resume200_raw_reward_progress.png)

![H200 continuation normalized reward progress](demo/results/plots/h200_resume200_norm_reward_progress.png)

![H200 continuation invalid-action progress](demo/results/plots/h200_resume200_invalid_action_progress.png)

## Agent Behavior

Each episode places civilians, hazards, stairwells, elevators, and constrained exits inside a deterministic multi-floor building.

| Element | Detail |
|---|---|
| **Orchestrator observation** | Floor summaries, inter-floor bottlenecks, belief rollups, recent floor actions, directive outcomes, unresolved escalations |
| **Floor-agent observation** | Visible rooms/corridors, local hazards, civilian groups, exits on floor, stairwell entries, active directives, action mask |
| **Floor-agent actions** | `route_within_floor`, `prioritize_room`, `open_exit`, `lockdown_room`, `scout`, `predict_state`, `handoff_to_orchestrator`, `wait` |
| **Orchestrator actions** | `route_between_floors`, `call_elevator`, `evacuate_floor_priority`, `broadcast_directive`, `override_floor_agent`, `request_explanation`, `wait` |
| **Reward signal** | Base simulation reward plus team progress, floor saved/lost terms, invalid-action penalties, coordination/directive quality, rationale bonuses |
| **Success shape** | Move civilians to safety while minimizing loss, bottlenecks, and invalid behavior across repeated rounds |

The runtime schema is exposed through `/openenv/schema`, and the canonical types live in [evacos_ma/schemas/multi_agent.py](evacos_ma/schemas/multi_agent.py) and [evacos_ma/schemas/rewards.py](evacos_ma/schemas/rewards.py).

## Fastest Proof Path

```bash
# 1. Generate a baseline evidence bundle. No GPU or checkpoint required.
#    Produces: baseline_vs_trained.csv, submission_scorecard.md, plots/
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline

# 2. Inspect the OpenEnv contract and live environment package.
cat openenv.yaml
ls evacos_ma/openenv

# 3. Confirm the split-role bridge config is checked in.
cat training/config.remote-unsloth-7b3b-split-bridge.yaml
```

The tracked proof artifacts are available without regenerating:

```bash
cat demo/results/submission_scorecard_baseline.md
```

For a full trained comparison, download or restore the selected LoRA adapter artifact first, then pass that adapter path explicitly:

```bash
CHECKPOINT_DIR=/path/to/downloaded/lora_adapter
python -m evaluation.demo_bundle \
  --trained-checkpoint "$CHECKPOINT_DIR" \
  --config training/config.remote-unsloth-7b3b-split-bridge.yaml \
  --output-dir outputs/demo_bundle
```

To publish the selected adapter artifact to Hugging Face Hub:

```bash
python scripts/upload_adapter.py \
  outputs/training/checkpoints/latest/lora_adapter \
  "$HF_ADAPTER_REPO"
```

Start with [SUBMISSION_BRIEF.md](SUBMISSION_BRIEF.md) if you want the narrative overview first.

## Repository Map

| Path | Purpose |
|---|---|
| [evacos_ma/](evacos_ma) | Simulator, schemas, reward interfaces, OpenEnv server code |
| [training/](training) | Rollout collection, policy adapters, GRPO trainer, checkpoints, backend integration |
| [evaluation/](evaluation) | Fixed-suite verification, comparison helpers, scorecards, plots, demo bundle |
| [dashboard/](dashboard) | Local inspection and demo UI |
| [demo/](demo) | Presentation and story assets |
| [notebooks/train_evacos_ma.ipynb](notebooks/train_evacos_ma.ipynb) | End-to-end notebook flow |

## Training Configs

The checked-in shared-model default is [training/config.yaml](training/config.yaml) with `Qwen/Qwen2.5-3B-Instruct`.

The strongest checked-in split bridge is [training/config.remote-unsloth-7b3b-split-bridge.yaml](training/config.remote-unsloth-7b3b-split-bridge.yaml):

- orchestrator: `Qwen/Qwen2.5-7B-Instruct`
- floor agents: `Qwen/Qwen2.5-3B-Instruct`
- disaster mix: `fire`, `flood`, `gas`

Split-role specialist variants keep the `7B/3B` topology but use one disaster family per run:

- fire: [training/config.remote-unsloth-7b3b-fire-specialist.yaml](training/config.remote-unsloth-7b3b-fire-specialist.yaml)
- flood: [training/config.remote-unsloth-7b3b-flood-specialist.yaml](training/config.remote-unsloth-7b3b-flood-specialist.yaml)
- gas: [training/config.remote-unsloth-7b3b-gas-specialist.yaml](training/config.remote-unsloth-7b3b-gas-specialist.yaml)

Floor-only `3B` specialist variants use a deterministic stub orchestrator and train only the floor-agent adapter:

- fire signal canary: [training/config.remote-unsloth-3b-fire-floor-specialist-signal-canary-10.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-signal-canary-10.yaml)
- fire canary: [training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml)
- fire throughput smoke: [training/config.remote-unsloth-3b-fire-floor-specialist-throughput-smoke-100.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-throughput-smoke-100.yaml)
- flood throughput smoke: [training/config.remote-unsloth-3b-flood-floor-specialist-throughput-smoke-100.yaml](training/config.remote-unsloth-3b-flood-floor-specialist-throughput-smoke-100.yaml)
- gas throughput smoke: [training/config.remote-unsloth-3b-gas-floor-specialist-throughput-smoke-100.yaml](training/config.remote-unsloth-3b-gas-floor-specialist-throughput-smoke-100.yaml)
- quality runs: [training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml), [training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml](training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml), [training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml](training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml)
The deeper experimental configs are intentionally not part of the judge-facing path; the active story is the focused specialist quality lane plus routed orchestration.

The specialist canaries are intentionally short proof runs. Use the `10`-step signal canary to prove same-prompt GRPO plumbing, then the `50`-step canary to prove parser/checkpoint/CSV health. The successful fire/flood/gas canary summary is tracked in [demo/results/specialist_canary50_report.md](demo/results/specialist_canary50_report.md).

Modern GRPO/OpenEnv practice requires candidate contrast inside each prompt group. Before any further paid specialist run, verify that the rollout path samples multiple completions for the same role/agent/prompt, groups them under the same prompt-scoped `group_id`, and logs non-zero `floor_agent_group_raw_reward_std_mean` plus non-zero `floor_agent_advantage_std`. A long run with flat groups is expected to waste compute even if mean normalized reward looks positive.

Use [scripts/check_grpo_contrast.py](scripts/check_grpo_contrast.py) as the quick pre-rental/post-run guard for those CSV columns:

```bash
python scripts/check_grpo_contrast.py outputs/path/to/metrics.csv
```

Longer specialist runs should use the focused quality configs on H200-class HF Jobs after a throughput-tuned `100`-step smoke. The canaries proved the reward/grouping signal is healthy; the next risk is wall-clock cost, especially for gas. Do not treat a quality lane as proven until its checkpoint, logs, and held-out eval artifact are captured.

For the active submission path, use the quality configs:

- fire: [training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml](training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml)
- flood: [training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml](training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml)
- gas: [training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml](training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml)

These are the default judge-facing specialist runs. They keep the public scope focused on the capability we can explain and verify: reliable fire, flood, and gas response behavior with checkpointed metrics.

After each real specialist run, convert the saved JSONL traces into a judge-facing eval artifact: one CSV and one plot showing `0-100%` held-out eval score at each 50-step eval checkpoint. Training rewards stay normalized/noisy for GRPO; the public result should show the cleaner eval score curve plus saved/lost civilian outcomes.

Phase-2 orchestrator training supports two frozen-floor modes:

- Single frozen floor policy: set `roles.trainable: ["orchestrator"]` and point `roles.frozen_adapter_paths.floor_agent` at one trained floor adapter.
- Routed frozen specialists: set `roles.frozen_floor_specialist_adapter_paths.fire/flood/gas` so the floor policy deterministically switches among trained `3B` specialists by observed disaster family while the `7B` orchestrator trains.

Start from [training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml](training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml) for the routed-specialist run. Checkpoints copy both trainable role adapters and frozen specialist adapter paths into the checkpoint metadata so final eval/upload does not depend on the original remote artifact folders.

Validated stronger lanes include:

- `7B` single-model
- `7B + vLLM`
- `7B/3B` split-role

## Evaluation And Evidence

Key files:

- fixed-suite evaluation: [evaluation/fixed_suite.py](evaluation/fixed_suite.py)
- baseline-vs-trained comparison: [evaluation/baseline_vs_trained.py](evaluation/baseline_vs_trained.py)
- bundle builder: [evaluation/demo_bundle.py](evaluation/demo_bundle.py)
- plot generation: [evaluation/plots.py](evaluation/plots.py)
- tracked proof bundle: [demo/results/](demo/results)

The bundle path emits:

- `baseline_vs_trained.csv`
- `demo_bundle_summary.md`
- `submission_scorecard.md`
- `submission_scorecard.json`
- plots generated from the run's actual metrics CSV

The scorecard includes a `0-100` headline eval score:

```text
eval_score_pct = 100 * save_rate * (1 - 0.5 * invalid_action_rate)
```

This is intentionally separate from training reward. It gives judges a clean success percentage while preserving raw and normalized rewards for debugging GRPO behavior.

Dedicated eval entrypoints wrap the same bundle builder with safer defaults:

```bash
# 3B floor specialists: one disaster family per report.
python scripts/eval_3b_fire.py --trained-checkpoint /path/to/fire/checkpoint-or-lora_adapter
python scripts/eval_3b_flood.py --trained-checkpoint /path/to/flood/checkpoint-or-lora_adapter
python scripts/eval_3b_gas.py --trained-checkpoint /path/to/gas/checkpoint-or-lora_adapter

# Shared 7B orchestrator over routed frozen fire/flood/gas floor specialists.
python scripts/eval_7b_orchestrator.py --trained-checkpoint /path/to/orchestrator/checkpoint-or-lora_adapter
```

By default these scripts evaluate the standard response lane on held-out seeds `42,123,456,789,1024`. The `3B` scripts restrict `disaster_families` to their specialist lane, while the `7B` script evaluates the routed `fire,flood,gas` orchestration stack.

---

Full narrative overview: [SUBMISSION_BRIEF.md](SUBMISSION_BRIEF.md)

Environment secrets: copy [.env.example](.env.example) to `.env` and fill only the local values you need. The real `.env` is gitignored.
