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
- Evaluation: fixed seeds, fixed tiers, scorecards, CSVs, and plots

This split is intentional: floor agents need low-latency local routing, while the orchestrator can spend more capacity on cross-floor planning, overrides, and disaster-level coordination.

## Reward Design

The reward is not one loose scalar. It combines:

- civilians saved or lost
- team progress through the building
- invalid-action penalties
- directive and override quality
- counterfactual gates for orchestrator rationale bonuses
- belief-audit gates for floor-agent prediction behavior

This makes the reward harder to game than a judge-only score or a single regex-like success check.

## Current Evidence

The tracked proof bundle lives in `demo/results/`.

- A100 split-role run: `100` steps, final checkpoint `ckpt_99`
- Model stack: `7B` orchestrator / `3B` floor agents
- Wall clock: `44.8` minutes
- Fire-medium curriculum reward moved from a first-window mean of `-1.5750` to a last-window mean of `0.8348`
- Final fire-medium reward reached `1.5382`; best observed fire-medium reward was `3.1896`
- Fixed-suite baseline scorecard is tracked separately for reproducible baseline inspection

This is training-signal evidence from the end-to-end pipeline. A final held-out trained-vs-baseline scorecard should be regenerated from the selected LoRA checkpoint artifact before the live pitch.

## Reproduce The Proof Path

```bash
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
cat demo/results/a100_7b3b_run_summary.md
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
- Training signal plot: `demo/results/plots/a100_7b3b_training_signal.png`
- Baseline scorecard: `demo/results/submission_scorecard_baseline.md`
