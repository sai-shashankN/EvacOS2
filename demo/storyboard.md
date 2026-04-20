# Phase 8 Demo Storyboard

## 0:00-0:10 Intro
- Artifact: `outputs/evals/plots/baseline_vs_trained_bar.png`
- Beat: establish the evacuation benchmark, fixed-suite evaluation, and the baseline-versus-trained framing.

## 0:10-0:30 Hierarchy Shot
- Artifact: dashboard capture from `dashboard/static/index.html`
- Beat: show the orchestrator plus five floor agents, then point to the live per-floor grid and reward ticker.

## 0:30-1:00 Orchestrator Override Clip
- Artifact: `outputs/evals/oversight_examples.md`
- Beat: narrate one override decision, why it was taken, and the counterfactual delta.

## 1:00-1:20 Reward Plot Reveal
- Artifact: `outputs/evals/plots/reward_curve.png`
- Beat: show the training curve, mention the fixed holdout suite, and connect the curve to the final checkpoint.

## 1:20-1:35 Rationale Sweep Callout
- Artifact: `outputs/evals/plots/rationale_mode_comparison.png`
- Beat: compare `off`, `linear_capped`, and `log_uncapped` on the same seed suite and name the selected mode.

## 1:35-1:50 Trajectory Render
- Artifact: renderer GIF from `renderer.unity_bridge.build_headless_render(...)`
- Beat: cut to the offline trajectory playback and call out that the Unity bridge consumes saved trajectory JSON only.

## 1:50-2:00 Outro
- Artifact: `demo/hf_blog.md`
- Beat: close on reproducibility, fixed seeds, and the HF mini-blog follow-up for reviewers.
