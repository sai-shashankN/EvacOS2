# Held-Out 3B Specialist Eval: Base Model vs Trained LoRA

Same held-out seeds, same evaluator, same family-specific lane. Baseline is Qwen2.5-3B-Instruct with no LoRA adapter and the same stub orchestrator used by the specialist training lane. This is a fast 10-seed held-out run; the 50-seed path uses the same command shape.

This is the controlled fixed-suite proof slice used for the public submission. It is meant to make the base-vs-trained comparison deterministic and auditable; broader higher-difficulty slices use the same evaluation shape as future validation work.

Headline: on this 30-episode held-out specialist evaluation, trained LoRA floor specialists more than doubled the base/no-LoRA Qwen2.5-3B average eval score, from 15.08% to 36.28%, while reducing invalid actions from 51.81% to 0.00%.

| Family | Episodes | Base eval score | Trained eval score | Delta | Base invalid | Trained invalid | Save-rate delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| fire | 10 | 11.58% | 36.34% | 24.76 | 0.6000 | 0.0000 | +0.2099 |
| flood | 10 | 17.63% | 36.34% | 18.70 | 0.4667 | 0.0000 | +0.1498 |
| gas | 10 | 16.02% | 36.16% | 20.14 | 0.4875 | 0.0000 | +0.1680 |
| **average** | **30 total** | **15.08%** | **36.28%** | **21.20** | **0.5181** | **0.0000** | **+0.1759** |
