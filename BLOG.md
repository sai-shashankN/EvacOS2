# EvacOS2: An OpenEnv Environment For Multi-Agent Evacuation Training

EvacOS2 is an OpenEnv-compatible reinforcement-learning environment for hierarchical evacuation coordination. One orchestrator supervises five floor agents inside a deterministic building simulator while hazards evolve, exits bottleneck, and local agents only see part of the world.

GitHub: [https://github.com/sai-shashankN/EvacOS2](https://github.com/sai-shashankN/EvacOS2)

Hugging Face Space: [https://huggingface.co/spaces/shashankN777/evacos2-openenv](https://huggingface.co/spaces/shashankN777/evacos2-openenv)

## Why This Is Interesting

Most agent demos can look good even if the model is only producing plausible text. EvacOS2 asks a harder question: can an LLM policy act inside a simulator, receive programmatic feedback, and become more reliable over repeated rollouts?

The environment targets three capability gaps:

- **Multi-agent interaction:** a building-level orchestrator and floor-level responders must coordinate under partial observability.
- **Long-horizon planning:** early routing choices affect later casualties, congestion, and invalid actions.
- **World modeling:** the policy must react to fire, flood, gas, and cascade-style dynamics rather than static prompts.

## Environment Design

Each episode creates a deterministic multi-floor building with civilians, rooms, corridors, stairs, elevators, exits, hazards, and bottlenecks. The orchestrator sees floor summaries and coordination state. Each floor agent sees local rooms, local hazards, local civilians, exits, active directives, and an action mask.

The OpenEnv-facing API exposes:

- `/openenv/reset`
- `/openenv/step`
- `/openenv/state`
- `/openenv/schema`
- `/openenv/health`
- `/openenv/metadata`

This means judges can pull the Space and interact with the same simulator surface used by the training and evaluation pipeline.

## Training Stack

EvacOS2 uses:

- **TRL / GRPO-style post-training** for verifier-driven policy improvement.
- **Unsloth + LoRA** for cheaper and faster adaptation.
- **Fixed-suite evaluation** for reproducible baseline-vs-trained comparisons.
- **Checkpoint-local metrics** so each checkpoint can carry its own `metrics_window.csv`, `metrics_to_date.csv`, and `metrics_summary.json`.

The training system supports a shared policy, split-role policies, and disaster-specialized floor responders.

## 3B Specialist Evidence

The cleanest completed evidence so far is the fire/flood/gas `3B` floor-specialist canary suite. These are short `50`-step proof runs, not final convergence claims. They show that the training lane is structurally alive: valid observations reach the model, route actions preserve target IDs, checkpoints write successfully, and GRPO groups have non-zero contrast.

| Specialist | Start valid-action score | Trained checkpoint score | Delta | Last-10 invalid rate | Last-10 GRPO reward std |
|---|---:|---:|---:|---:|---:|
| Fire `3B` | `83.65%` | `96.54%` | `+12.89 pp` | `3.46%` | `0.7168` |
| Flood `3B` | `88.46%` | `97.54%` | `+9.08 pp` | `2.46%` | `1.3848` |
| Gas `3B` | `89.62%` | `97.13%` | `+7.51 pp` | `2.87%` | `0.9769` |

`valid-action score = 100 * (1 - invalid_action_rate)`. The start score is step `0`; the trained checkpoint score is the last-10 average ending at checkpoint `ckpt_49`.

![3B specialist valid-action score comparison](demo/results/plots/3b_specialist_valid_action_score_comparison.png)

![3B specialist invalid action rate across checkpoints](demo/results/plots/3b_specialist_invalid_action_checkpoints.png)

![3B specialist raw reward across checkpoints](demo/results/plots/3b_specialist_raw_reward_checkpoints.png)

## Why The 7B Orchestrator Exists

The `7B` orchestrator is not meant to replace fast local responders. It is a proof of concept for a multi-agent workflow where most floor-level decisions remain cheap and fast, while a larger coordinator handles the unusual cases.

In the intended hierarchy:

- `3B` floor specialists handle routine local routing.
- A deterministic scope router chooses the relevant fire/flood/gas lane from incident metadata.
- The `7B` orchestrator handles global priorities, cross-floor conflicts, stalled evacuations, sensor disagreement, and other outlier scenarios.

That architecture leaves room for future self-healing behavior: specialist fallback, anomaly escalation, policy repair suggestions, and human-in-the-loop override review can sit above the fast responders without slowing every routine decision.

## Reward And Anti-Hacking Design

The reward is not one loose scalar. It combines:

- civilians saved or lost
- team progress through the building
- invalid-action penalties
- directive and override quality
- counterfactual gates for orchestrator rationale bonuses
- belief-audit gates for floor-agent prediction behavior

The fixed-suite evaluator is separate from the training-time reward. Training rewards stay useful for GRPO optimization, while judge-facing evaluation reports a bounded `0-100%` score, save rate, invalid-action rate, and scenario outcomes.

## What Is Proven And What Comes Next

Proven now:

- The OpenEnv environment is real and runnable.
- The multi-agent simulator is deterministic and evaluable.
- The training stack runs end to end with LoRA checkpoints.
- The `3B` specialist canaries reduce invalid actions and keep route targets valid.
- The `7B/3B` split-role path has completed smoke training and checkpointing.

Still ongoing:

- Longer specialist training for stronger held-out performance.
- Final selected-checkpoint baseline-vs-trained scorecards.
- A stronger routed `7B` orchestrator run over frozen specialists.

The core story is already useful for OpenEnv: EvacOS2 is not just a prompt wrapper. It is a trainable, measurable, multi-agent environment for emergency coordination.
