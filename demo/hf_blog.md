# EvacOS2: Training LLM Agents To Coordinate Evacuations

EvacOS2 is an OpenEnv-compatible benchmark for hierarchical multi-agent evacuation. One orchestrator supervises five floor agents inside a deterministic building simulator while hazards evolve, exits bottleneck, and local agents only see part of the world.

## Why This Environment

Most agent demos can look good even when the model is only producing plausible text. EvacOS2 instead asks whether an LLM policy can act inside a simulator, receive programmatic feedback, and improve coordinated behavior over repeated rollouts.

The environment targets three hackathon themes at once:

- multi-agent interaction: floor agents and an orchestrator must coordinate under partial observability
- long-horizon planning: early routing choices affect later casualties, bottlenecks, and invalid actions
- world modeling: the model has to react to changing fire, flood, gas, and cascade conditions

## Agent Setup

- Topology: `1` orchestrator plus `5` floor agents
- Fast local policy: `Qwen/Qwen2.5-3B-Instruct` floor agents
- Slower global policy: `Qwen/Qwen2.5-7B-Instruct` orchestrator
- Training: Unsloth + LoRA + GRPO-style updates
- Evaluation: held-out seeds, response-family scorecards, CSVs, and plots

This split is intentional: floor agents need low-latency local routing, while the orchestrator can spend more capacity on cross-floor planning, overrides, and disaster-level coordination.

## Reward Design

The reward is not one loose scalar. It combines:

- civilians saved or lost
- team progress through the building
- invalid-action penalties
- directive and override quality
- counterfactual gates for orchestrator rationale bonuses
- belief-audit gates for floor-agent prediction behavior

This makes the reward more robust than a judge-only score or a single regex-like success check.

## Current Evidence

The tracked proof bundle lives in `demo/results/`.

- Fire/flood/gas `3B` specialist canaries completed end-to-end with LoRA checkpoints.
- Valid-action score improved from `83.65% -> 96.54%` for fire, `88.46% -> 97.54%` for flood, and `89.62% -> 97.13%` for gas.
- The split-role `7B/3B` path has a tracked `100`-step historical smoke run with final checkpoint `ckpt_99`.
- Fixed-suite baseline scorecards and plots are tracked for reproducible inspection.

This is training-signal evidence from the end-to-end pipeline: the environment runs, the trainer emits grouped GRPO diagnostics, checkpoints persist, and the public scorecard path can evaluate selected LoRA adapters.

## Reproduce The Proof Path

```bash
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
cat demo/results/specialist_canary50_report.md
```

For selected checkpoint comparison:

```bash
python -m evaluation.demo_bundle \
  --trained-checkpoint /path/to/downloaded/lora_adapter \
  --config training/config.remote-unsloth-7b3b-split-bridge.yaml \
  --output-dir outputs/demo_bundle
```

## Links To Include When Published

- GitHub: `https://github.com/sai-shashankN/EvacOS2`
- Space: `https://huggingface.co/spaces/shashankN777/evacos2-openenv`
- Training signal plot: `demo/results/plots/3b_specialist_valid_action_score_comparison.png`
- Baseline scorecard: `demo/results/submission_scorecard_baseline.md`
