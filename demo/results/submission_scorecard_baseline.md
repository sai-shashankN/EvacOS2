# EvacOS2 Submission Scorecard

## Judge-Fast Take
- Environment: multi-agent evacuation simulator with a live OpenEnv-facing API
- Evidence: fixed-suite baseline artifacts plus a separate tracked A100 training-signal summary
- Goal: show measurable improvement in coordinated evacuation behavior, not just plausible text output

## Headline
- rationale mode: `linear_capped`
- improved metrics: `0`
- regressed metrics: `0`
- flat metrics: `0`
- no-trained-data metrics: `5`

## Metrics
| metric | goal | baseline | trained | delta | status |
| --- | --- | ---: | ---: | ---: | --- |
| orchestrator normalized reward | higher | -3.0526 | n/a | n/a | no_trained_data |
| floor-agent normalized reward | higher | -1.1667 | n/a | n/a | no_trained_data |
| save rate | higher | 0.5853 | n/a | n/a | no_trained_data |
| invalid action rate | lower | 0.0000 | n/a | n/a | no_trained_data |
| override win rate | higher | 0.0000 | n/a | n/a | no_trained_data |

## Bundle Artifacts
- comparison CSV: `demo/results/baseline_fixed_suite.csv`
- baseline fixed suite: `demo/results/fixed_suite_baseline_linear_capped.json`
- trained fixed suite: skipped

## Suggested Submission Flow
- open this scorecard first
- open `demo/results/demo_bundle_summary_baseline.md` for the slightly longer explanation
- show the CSV/plots only after the headline metrics are clear
- finish with one live OpenEnv interaction
