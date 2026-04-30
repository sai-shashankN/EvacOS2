# Held-Out 3B Specialist Eval: ckpt_199 Base Model vs Trained LoRA

Same held-out seeds `9101-9110`, same evaluator, same family-specific controlled proof lane. Baseline is `Qwen/Qwen2.5-3B-Instruct` with no LoRA adapter; trained policies are fire/flood/gas H200 resume200 `ckpt_199` LoRA specialists. The gas row was refreshed on Vast after the route-target audit to use the public `gas/h200-resume200-ckpt199` adapter and uploaded to `heldout/vast-gas-ckpt199-heldout10-f3ad625-20260430T202253Z`.

The base/no-LoRA policy is intentionally non-trivial because EvacOS2 exposes clear emergency observations and action contracts. The judge-facing claim is therefore not that the base model is helpless; it is that LoRA training improves bounded score and contract reliability under the same unseen seeds, same evaluator, and same route/action parser.

Audit note: the fixed-suite score is bounded to `0-100%`, but this refresh also records raw save-rate overflow counts because some fire traces double-count saved civilians at the simulator-accounting layer. For judging, the most robust headline is therefore the base/no-LoRA to trained LoRA invalid-action reduction, plus the bounded score as an audited proof-slice signal rather than a claim of final convergence.

Headline: average bounded eval score `62.38% -> 80.05%` (`+17.67 pp`); invalid actions `34.47% -> 1.10%` (`-33.37 pp`).

| Family | Episodes | Base bounded score | Trained bounded score | Delta | Base invalid | Trained invalid | Capped save-rate delta | Raw save overflows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fire | 10 | 84.77% | 96.64% | +11.87 pp | 27.52% | 2.60% | -0.0030 | 5/7 |
| flood | 10 | 22.95% | 45.42% | +22.47 pp | 47.67% | 0.00% | +0.1876 | 0/0 |
| gas | 10 | 79.43% | 98.09% | +18.67 pp | 28.23% | 0.69% | +0.0668 | 0/0 |
| **Average** | **30 total** | **62.38%** | **80.05%** | **+17.67 pp** | **34.47%** | **1.10%** | **+0.0838** | **audit logged** |

Route-target discipline is tracked separately in the canary scorecard: fire/flood/gas `3B` specialists all record `1.0000` route-action averages and `0.0000` missing-target averages in [3b_specialist_canary50_scores.csv](3b_specialist_canary50_scores.csv). The refreshed gas held-out audit records a low `0.73%` route-missing-target rate on the stricter 10-seed held-out slice.

