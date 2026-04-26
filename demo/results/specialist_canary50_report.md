# 3B Specialist Canary-50 Report

This report summarizes the first clean `50`-step floor-specialist canaries after the route/action and GRPO grouping fixes.

All three runs used `Qwen/Qwen2.5-3B-Instruct`, Unsloth LoRA, a deterministic stub orchestrator, `candidates_per_floor_prompt: 4`, and `include_oracle_floor_candidate: true`.

## Outcome

| Disaster | Status | Rows | Final step | Checkpoint | Local artifact |
|---|---:|---:|---:|---:|---|
| Fire | `TRAIN_EXIT=0` | `50` | `49` | yes | `outputs/vast_fire_canary50_oraclefix_35603298/fire_canary50_artifacts.tgz` |
| Flood | `TRAIN_EXIT=0` | `50` | `49` | yes | `outputs/vast_flood_canary50_35606519/flood_canary50_artifacts.tgz` |
| Gas | `TRAIN_EXIT=0` | `50` | `49` | yes | `outputs/vast_gas_canary50_35606521/gas_canary50_artifacts.tgz` |

## Training Signal

| Disaster | First invalid | Last invalid | Last-10 invalid avg | Last-10 raw reward avg | Route action avg | Missing target avg |
|---|---:|---:|---:|---:|---:|---:|
| Fire | `0.1635` | `0.0096` | `0.0346` | `8.01` | `1.0000` | `0.0000` |
| Flood | `0.1154` | `0.0000` | `0.0246` | `7.01` | `1.0000` | `0.0000` |
| Gas | `0.1038` | `0.0038` | `0.0287` | `28.41` | `1.0000` | `0.0000` |

Interpretation: the specialist training lane is structurally alive for all three disasters. The fixed prompt/action path now produces valid `route_within_floor` actions with exact target IDs, the GRPO groups have non-zero reward contrast, and checkpoints are written.

This is not a final strength claim. It is the proof gate that says longer training is worth running after throughput tuning.

## Runtime

| Disaster | Max rounds | Total train wall time | Avg seconds / step | Last-10 seconds / step |
|---|---:|---:|---:|---:|
| Fire | `4` | `35.46 min` | `42.6s` | `42.6s` |
| Flood | `10` | `54.03 min` | `64.8s` | `64.0s` |
| Gas | `10` | `105.27 min` | `126.3s` | `122.3s` |

The flood/gas canaries were intentionally conservative with `max_rounds_per_episode: 10`. Post-run traces show fire episodes always used `4` rounds, flood always used `5`, and gas usually used `8-10`.

## Scaling Decision

Before a `400+` step specialist run:

- Use disaster-specific horizons: fire `4`, flood `5`, gas `10` unless a held-out gas eval proves `8` is safe.
- Use less frequent checkpoint/eval cadence for long runs, such as every `25` or `50` steps instead of every `10`.
- Keep `candidates_per_floor_prompt: 4` during the bootstrap window, then consider dropping to `2` after invalid rates are consistently below `0.05`.
- Keep `include_oracle_floor_candidate: true` for early training; remove only after a trained-policy eval shows the model no longer needs oracle contrast.

Recommended next run is a throughput-tuned `100`-step smoke, not a direct jump to `750`.
