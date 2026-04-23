# Demo Bundle Summary

## Artifacts
- comparison CSV: `demo/results/baseline_fixed_suite.csv`
- baseline fixed suite: `demo/results/fixed_suite_baseline_linear_capped.json`
- trained fixed suite: skipped

## Rationale Mode
- `linear_capped`

## Baseline Metrics
- orchestrator mean normalized reward: `-3.0526`
- floor-agent mean normalized reward: `-1.1667`
- save rate: `0.5853`
- invalid action rate: `0.0000`
- override win rate: `0.0000`

## Suggested Demo Flow
- show one baseline fixed-suite summary
- show the comparison CSV and generated plots
- show one live `/openenv/reset -> /openenv/step -> /openenv/state` interaction
- explain which safeguards prevent reward hacking
