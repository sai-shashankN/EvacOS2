# EvacOS2: Training LLM Agents To Coordinate Real Evacuations

EvacOS2 is an OpenEnv-compatible reinforcement-learning environment for one of the places where agentic AI should be tested hardest: emergency coordination. Instead of asking a model to write a plausible evacuation plan in one message, EvacOS2 places a hierarchy of agents inside a deterministic building simulator. Hazards evolve. Exits bottleneck. Floor agents see only local state. A building-level orchestrator has to coordinate priorities across floors. Every action is checked by the environment.

That makes the project different from a normal chatbot demo. The model cannot win by sounding confident. It has to act, observe consequences, and improve against programmatic rewards.

GitHub: [https://github.com/sai-shashankN/EvacOS2](https://github.com/sai-shashankN/EvacOS2)

Hugging Face Space: [https://huggingface.co/spaces/shashankN777/evacos2-openenv](https://huggingface.co/spaces/shashankN777/evacos2-openenv)

Training notebook: [https://huggingface.co/spaces/shashankN777/evacos2-openenv/blob/main/notebooks/train_evacos_ma.ipynb](https://huggingface.co/spaces/shashankN777/evacos2-openenv/blob/main/notebooks/train_evacos_ma.ipynb)

Public model artifacts: [https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts](https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts)

## Why I Built This

This project started from grief.

During the war in Lebanon, a Lebanese friend of mine stopped replying. Later, I learned he was gone. That changed how I thought about emergency response. Evacuation was no longer an abstract optimization problem; it was about whether people get the right help in the small window where help still matters.

Around the same time, while I was thinking about this hackathon, I felt an earthquake. The coincidence stayed with me. Disasters are chaotic, local, and time-sensitive. People do not fail only because there is no plan; they fail because information is partial, coordination breaks down, and decisions arrive too late.

EvacOS2 is my attempt to turn that feeling into a technical environment: one where LLM agents are not rewarded for sounding calm, but for coordinating actions that move people toward safety.

## Judge TL;DR

EvacOS2 is built to score well on the things that matter in this hackathon:

- **Environment innovation:** this is a hierarchical emergency-response simulator, not another toy grid world or static prompt set. It tests partial observability, bottlenecks, cross-floor coordination, and disaster-specific response.
- **Benchmark angle:** beyond evacuation, it tests calibrated autonomy: when subagents should act locally, escalate, recover if the orchestrator is unreachable, or accept a smaller local win for better team coordination.
- **Storytelling:** the demo has a clear mental model: fast `3B` floor responders handle local movement, while a slower `7B` orchestrator handles building-level priorities and outliers.
- **Training evidence:** the public fire/flood/gas `3B` canaries show valid-action improvement, checkpointed LoRA adapters, logs, metrics, and plots from real runs.
- **Evaluation scope:** the submitted held-out comparison uses a controlled specialist proof slice so base-vs-trained results are deterministic, rerunnable, and easy to audit; harder slices are a natural next benchmark milestone.
- **OpenEnv compliance:** the live Hugging Face Space exposes `/openenv/health`, `/openenv/metadata`, `/openenv/schema`, `/openenv/reset`, `/openenv/step`, and `/openenv/state`.
- **Honest frontier:** the `7B` orchestrator is included as a smoke/training-signal validated coordination layer, with a dedicated behavior card explaining what is proven and what the next held-out eval should measure.

The project’s pitch is simple: if LLM agents are going to operate in the world, they need environments where actions have consequences. EvacOS2 gives them one.

## The Core Idea

Most LLM evaluation still happens in prompt-response form. That is useful, but it misses a large part of what agents need to do in the world: maintain state, delegate work, react to partial observations, and make decisions whose consequences unfold over time.

EvacOS2 turns that into a trainable environment. The simulator creates a multi-floor building with civilians, rooms, corridors, stairwells, elevators, exits, local hazards, and global coordination pressure. The system then asks a hierarchy of LLM policies to evacuate civilians while minimizing invalid actions, congestion, and casualties.

While I was on a flight thinking about what we had actually built, I realized the evacuation setting was only the surface. EvacOS2 is also a benchmark for calibrated autonomy: when a local agent should solve a problem itself, when it should escalate, when the orchestrator should interfere, and when a locally capable agent should still pass context upward because the whole team may benefit from that shared knowledge later.

That matters because most subagents are trained or prompted to finish only their own task. EvacOS2 creates a safer place to test a different behavior: a floor agent that may accept a smaller local win because routing an outlier to the orchestrator improves the global workflow, helps other floors, or gives the team a fallback plan if the coordinator is slow. The benchmark is not only independence versus obedience. It is independence versus positively controlled interdependence.

The novelty is the combination:

- A deterministic emergency simulator with multi-agent roles.
- An OpenEnv API that judges can pull and run directly.
- A role-aware training stack for shared policies, split-role policies, and specialist floor responders.
- Verifiable evaluation with scorecards and plots instead of hand-picked examples.
- A realistic routing layer where incident metadata selects the right specialist lane, while a larger orchestrator remains available for cross-floor conflicts and outliers.

In practical terms, EvacOS2 is a benchmark for whether LLM post-training can make agents better at coordinated action, not just better at explanation. It also asks a harder question: can each agent decide whether it should be the one solving the problem?

## Why Evacuation Is A Strong Testbed

Emergency response is naturally multi-agent. A floor responder may know which rooms are blocked locally, but not whether another floor has already saturated the stairwell. A building coordinator may know that one exit is overloaded, but not every local hazard detail. Good behavior requires both local speed and global judgment.

That is exactly the kind of structure hierarchical agents should learn:

- **Partial observability:** floor agents see local rooms, civilians, hazards, exits, and active directives.
- **Long-horizon consequences:** an early route choice can create a later bottleneck.
- **Coordination pressure:** the orchestrator must allocate scarce exits, override bad local plans, and broadcast directives.
- **Verifiable outcomes:** the simulator can score saved civilians, invalid actions, losses, and progress.

This matters because many agent benchmarks reward verbal correctness. EvacOS2 rewards operational correctness.

## Architecture

The environment uses a `1 + 5` hierarchy: one orchestrator supervises five floor agents.

```mermaid
graph LR
    I["Incident metadata"] --> R["Scope router"]
    R --> F["Fire floor specialist"]
    R --> W["Flood floor specialist"]
    R --> G["Gas floor specialist"]
    R --> X["Generalist fallback"]

    O["7B orchestrator"] --> A1["Floor agent 1"]
    O --> A2["Floor agent 2"]
    O --> A3["Floor agent 3"]
    O --> A4["Floor agent 4"]
    O --> A5["Floor agent 5"]

    F --> A1
    W --> A2
    G --> A3
    X --> A4

    A1 --> S["Deterministic simulator"]
    A2 --> S
    A3 --> S
    A4 --> S
    A5 --> S
    S --> E["OpenEnv rewards + observations"]
```

The design is intentionally pragmatic. The `3B` floor specialists are the fast responders. They handle local routing decisions quickly. The `7B` orchestrator is not meant to replace them; it is the slower global coordinator that can reason about building-wide priorities, cross-floor conflicts, stalled evacuations, and weird edge cases.

That mirrors how real emergency systems work. Fire alarms, moisture or water sensors, gas detectors, facility reports, and dispatch metadata can identify the broad incident family before an AI policy acts. EvacOS2 uses that already-known incident context to route the local floor policy deterministically. The larger orchestrator then focuses on coordination rather than wasting capacity guessing the disaster type.

## OpenEnv Surface

EvacOS2 exposes the standard OpenEnv-style endpoints:

- `/openenv/health`
- `/openenv/metadata`
- `/openenv/schema`
- `/openenv/reset`
- `/openenv/step`
- `/openenv/state`

The Hugging Face Space is the canonical live environment surface. The same API shape is used by the training and evaluation code, so the demo is not a separate toy wrapper.

## Training Stack

The training stack is built for hackathon practicality:

- **OpenEnv** standardizes the environment boundary.
- **TRL / GRPO-style optimization** handles RL post-training.
- **Unsloth + LoRA** keeps fine-tuning affordable.
- **Checkpoint-local metrics** make it easier to inspect progress over time.
- **Fixed-suite evaluation** reports clean baseline-vs-trained comparisons.

GRPO is a good fit here because the environment provides verifiable reward signals and we need a lighter path than PPO-style setups. Unsloth helps keep the memory and runtime budget realistic. LoRA adapters let us preserve checkpoints without uploading full model copies.

## Reward Design

The reward is deliberately not one vague scalar. EvacOS2 combines several signals:

- civilians saved or lost
- movement progress through the building
- invalid-action penalties
- floor-level routing quality
- orchestrator directive quality
- override quality
- belief and prediction checks

The key design choice is that reward is layered. Route validity and target preservation keep the model from learning malformed actions. Evacuation progress and saved/lost terms keep it attached to the real objective. Orchestrator and override terms measure whether global coordination is helping rather than merely adding control. This reduces the risk that one easy-to-game reward dominates the policy.

Training reward and judge-facing evaluation are kept separate. Training rewards can remain normalized and contrastive for GRPO, while evaluation reports bounded, readable metrics such as valid-action score, invalid-action rate, saved civilians, and baseline-vs-trained deltas.

That separation matters. A messy reward curve may still be useful for optimization, but a judge should see a finite scorecard and a plot they can understand in seconds.

## Evidence So Far

The cleanest reward-improvement evidence is now a held-out `3B` specialist comparison: the same Qwen2.5-3B base model with no LoRA adapter versus the trained LoRA specialist, on the same unseen seeds and the same fixed-suite evaluator.

This is intentionally framed as a controlled proof lane, not the full ceiling of the simulator. For judges, the goal is a clean scientific slice: same seeds, same evaluator, same task families, base model versus trained adapter. The simulator and curriculum can be extended to harder slices, but the submitted public table focuses on the benchmark slice we can make reproducible and inspectable end to end.

The headline is the part that matters: on a `30`-episode held-out specialist evaluation, trained LoRA floor specialists improved the base/no-LoRA Qwen2.5-3B bounded eval score from `58.65%` to `80.45%` (`+21.80 pp`), while cutting invalid actions from `39.18%` to `0.87%`.

One audit detail is worth making explicit because this is a simulator, not a static worksheet: the scorecard records raw save-rate overflow counts for fire/gas traces where the simulator-accounting layer can double-count saved civilians. The public eval score is still bounded to `0-100%`, but the cleanest learning signal is the invalid-action reduction plus the bounded score movement, not a claim that this is a final converged policy.

| Family | Held-out episodes | Base bounded score | Trained bounded score | Delta | Base invalid rate | Trained invalid rate | Raw save overflows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fire | `10` | `84.77%` | `96.64%` | `+11.87 pp` | `27.52%` | `2.60%` | `5/7` |
| Flood | `10` | `22.95%` | `45.42%` | `+22.47 pp` | `47.67%` | `0.00%` | `0/0` |
| Gas | `10` | `68.22%` | `99.28%` | `+31.06 pp` | `42.34%` | `0.00%` | `2/8` |
| **Average** | `30 total` | **`58.65%`** | **`80.45%`** | **`+21.80 pp`** | **`39.18%`** | **`0.87%`** | **audit logged** |

The public artifact path is `heldout/base-model-vs-h200-resume200-ckpt199-3b-heldout10-batched-20260429-094214` in [shashankN777/evacos2-7b-orchestrator-artifacts](https://huggingface.co/shashankN777/evacos2-7b-orchestrator-artifacts). This run is intentionally labeled as a fast `10`-seed held-out comparison; the same evaluator path supports larger seed counts.

![Held-out 3B base-vs-trained eval score](demo/results/plots/heldout_3b_base_vs_trained_eval_score.png)

![Held-out 3B base-vs-trained invalid action rate](demo/results/plots/heldout_3b_base_vs_trained_invalid_action_rate.png)

The `3B` floor-specialist canary suite across fire, flood, and gas response lanes adds the training-signal story. These short runs are not presented as final convergence. They are proof that the environment, training, checkpointing, and metrics pipeline are live end to end.

| Specialist | Start valid-action score | Trained checkpoint score | Delta | Last-10 invalid rate | Last-10 GRPO reward std |
|---|---:|---:|---:|---:|---:|
| Fire `3B` | `83.65%` | `96.54%` | `+12.89 pp` | `3.46%` | `0.7168` |
| Flood `3B` | `88.46%` | `97.54%` | `+9.08 pp` | `2.46%` | `1.3848` |
| Gas `3B` | `89.62%` | `97.13%` | `+7.51 pp` | `2.87%` | `0.9769` |

`valid-action score = 100 * (1 - invalid_action_rate)`. The start score is step `0`; the trained checkpoint score is the last-10 average ending at checkpoint `ckpt_49`.

The continuation run resumes from those `ckpt_49` specialists and uploads checkpoints every `10` steps. This is not presented as a final held-out scorecard; it is the live reward-signal snapshot showing that the next phase is moving in the right direction.

| Specialist | Uploaded continuation steps | First-10 invalid rate | Last-10 invalid rate | First-10 raw reward | Last-10 raw reward | Reading |
|---|---:|---:|---:|---:|---:|---|
| Fire `3B` | `50 -> 199` | `1.44%` | `0.00%` | `10.56` | `12.06` | reward up, invalid actions eliminated in the latest window |
| Flood `3B` | `50 -> 199` | `3.69%` | `0.00%` | `7.60` | `8.83` | reward up, invalid actions eliminated in the latest window |
| Gas `3B` | `50 -> 199` | `0.19%` | `5.89%` | `29.40` | `31.89` | reward up, with latest-window invalid actions needing continued monitoring |

![3B specialist valid-action score comparison](demo/results/plots/3b_specialist_valid_action_score_comparison.png)

![3B specialist invalid action rate across checkpoints](demo/results/plots/3b_specialist_invalid_action_checkpoints.png)

![3B specialist raw reward across checkpoints](demo/results/plots/3b_specialist_raw_reward_checkpoints.png)

![H200 continuation raw reward progress](demo/results/plots/h200_resume200_raw_reward_progress.png)

![H200 continuation normalized reward progress](demo/results/plots/h200_resume200_norm_reward_progress.png)

![H200 continuation invalid-action progress](demo/results/plots/h200_resume200_invalid_action_progress.png)

The important result is not only that the score improves. It is that the system produces the full artifact chain: rollouts, rewards, checkpoints, metrics, plots, and reproducible evaluation scripts.

## Why The 7B Orchestrator Matters

The `7B` orchestrator is the future-proofing layer. If every scenario were routine, specialist floor responders would be enough. Real incidents are not always routine.

The orchestrator exists for:

- global evacuation priorities
- cross-floor bottlenecks
- conflicting local recommendations
- stalled evacuations
- uncertainty in incident reports
- fallback when a specialist lane is not appropriate
- human-review summaries and escalation notes

This creates a path toward self-healing multi-agent coordination. A local specialist can act quickly; a larger coordinator can notice anomalies, redirect policy choices, request explanations, and escalate cases where the normal route fails. That is a more realistic architecture than a single monolithic model trying to do every decision at every speed.

The public behavior card for this layer is [demo/results/7b_orchestrator_behavior_card.md](demo/results/7b_orchestrator_behavior_card.md). Its status is intentionally precise: the `7B/3B` split-role path is smoke/training-signal validated and checkpoints role-specific adapters, but final held-out `7B` orchestrator convergence is still the next evaluation target.

## What Makes This Submission Novel

EvacOS2 is not trying to be another grid-world clone. The point is to test a capability gap that matters: can LLM agents coordinate under constrained, evolving, partially observable conditions and get measurably better from RL?

The strongest parts of the submission are:

- **Operational realism:** alarms and sensor metadata route the response lane before model action.
- **Multi-agent hierarchy:** local agents and global coordinator have different jobs.
- **Trainability:** the environment is wired into an actual GRPO/LoRA pipeline.
- **Verifiability:** the evaluator produces scorecards and plots.
- **Extensibility:** the same framework can support specialist responders, a generalist fallback, and larger orchestrator training.
- **OpenEnv compliance:** the environment is exposed through a standard runnable API instead of private scripts.

The result is a serious environment for agent post-training: ambitious enough to be interesting, but concrete enough to run, score, and debug.

## How To Reproduce The Evidence

Baseline evidence can be regenerated without a GPU:

```bash
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
```

The checked-in notebook and training configs show the GPU training path with TRL, Unsloth, and LoRA. For trained comparisons, restore a selected adapter checkpoint and run:

```bash
CHECKPOINT_DIR=/path/to/downloaded/lora_adapter
python -m evaluation.demo_bundle \
  --trained-checkpoint "$CHECKPOINT_DIR" \
  --config training/config.remote-unsloth-7b3b-split-bridge.yaml \
  --output-dir outputs/demo_bundle
```

The public Space exposes the OpenEnv endpoints so reviewers can inspect the environment contract directly.

## Closing

EvacOS2 is built around a simple belief: LLM agents should be evaluated where their actions matter. Emergency evacuation is a domain where coordination, uncertainty, and delayed consequences all show up naturally. That makes it a strong setting for reinforcement learning with verifiable environments.

The current submission demonstrates a runnable OpenEnv environment, an end-to-end training pipeline, specialist checkpoint evidence, and a scalable path toward `7B` orchestration over fast local responders. It is a foundation for training agents that do not merely describe plans, but improve at acting inside a changing world.
