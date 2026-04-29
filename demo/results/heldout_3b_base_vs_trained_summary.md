# Held-Out 3B Specialist Eval: ckpt_199 Base Model vs Trained LoRA

Same held-out seeds `9101-9110`, same evaluator, same family-specific controlled proof lane. Baseline is Qwen2.5-3B-Instruct with no LoRA adapter; trained policies are fire/flood/gas H200 resume200 `ckpt_199` LoRA specialists.

Headline: average eval score 58.65% -> 80.45% (+21.80 pp); invalid actions 39.18% -> 0.87%.

| Family | Episodes | Base eval score | Trained eval score | Delta | Base invalid | Trained invalid | Save-rate delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| fire | 10 | 84.77% | 96.64% | +11.87 pp | 27.52% | 2.60% | +0.6400 |
| flood | 10 | 22.95% | 45.42% | +22.47 pp | 47.67% | 0.00% | +0.1876 |
| gas | 10 | 68.22% | 99.28% | +31.06 pp | 42.34% | 0.00% | +0.6740 |
| **Average** | **30 total** | **58.65%** | **80.45%** | **+21.80 pp** | **39.18%** | **0.87%** | **+0.5006** |
