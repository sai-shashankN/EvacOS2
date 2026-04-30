# 7B Orchestrator Behavior Card

This card explains what the `7B` orchestrator is responsible for, what has been validated publicly, and what is still pending. It exists because the floor-specialist evidence is easier to read than the orchestrator evidence.

## Role

The `7B` orchestrator is the slower, stronger building-level coordinator. It sees floor summaries, global bottlenecks, directive history, recent floor-agent actions, and escalation state.

Its job is not to replace the fast `3B` floor responders. Its job is to decide:

- which floor should be prioritized when multiple floors are under pressure
- when a local floor plan should be overridden
- when exits, elevators, or stairwells need global coordination
- when a stalled evacuation should be escalated
- when an unusual or mixed incident should fall back from a specialist lane to the generalist path

## Intended Before / After Behavior

| Situation | Baseline-style behavior | Desired `7B` orchestrator behavior |
|---|---|---|
| One floor has rising hazard severity while another floor is blocking an exit path | Waits or emits a local-looking action that does not change global priority | Broadcasts a directive or priority decision that routes low-risk floors away from the bottleneck and prioritizes the endangered floor |
| A floor agent repeatedly chooses an invalid or low-value local action | Lets the local loop continue | Overrides the floor agent or requests an explanation before committing more rounds |
| Incident metadata is single-family and sensor-supported | Uses the default policy path | Lets the deterministic scope router use the matching frozen floor specialist, while the `7B` focuses on building-level conflicts |
| Incident metadata is mixed, unknown, or cascading | Over-specializes to one local response mode | Falls back to the generalist path and handles the scenario as an outlier |

## Public Evidence

The public evidence currently shows that the `7B/3B` split-role training path is wired, trainable, resumable, and observable:

- `100` split-role training steps completed.
- Final checkpoint reached `ckpt_99`.
- Both `orchestrator` and `floor_agent` LoRA adapters were checkpointed.
- Per-role metrics were emitted, including `orchestrator_loss`, `floor_agent_loss`, KL terms, mask coverage, and reward diagnostics.
- The run produced non-zero reward movement in the tracked fire curriculum lanes.
- A later `7B` continuation canary over frozen fire/flood/gas `3B` specialists resumed from public checkpoint `ckpt_349` and ran steps `350-359`.
- The continuation canary completed with `TRAIN_EXIT=0` and uploaded adapters/metrics to `runs/vast-unsloth-7b-orchestrator-frozen-specialists-continue360-1c40744-48gb` in `shashankN777/evacos2-7b-orchestrator-artifacts`.
- That canary recorded `0.00%` aggregate invalid actions, `0.00%` orchestrator parse errors, `86.45%` average priority-rank fraction, `78.33%` average priority coverage, and `100.00%` priority-effect bonus rate.

Source artifacts:

- [a100_7b3b_run_summary.md](a100_7b3b_run_summary.md)
- [a100_7b3b_training_signal.csv](a100_7b3b_training_signal.csv)
- [a100_7b3b_training_signal_summary.csv](a100_7b3b_training_signal_summary.csv)
- Public continuation canary: `shashankN777/evacos2-7b-orchestrator-artifacts/runs/vast-unsloth-7b-orchestrator-frozen-specialists-continue360-1c40744-48gb`

## One Metric

The split-role smoke run produced visible reward movement, especially in the harder fire curriculum lane recorded in the summary:

| Lane | Samples | First reward | Last reward | First window mean | Last window mean | Best reward |
|---|---:|---:|---:|---:|---:|---:|
| Fire lane A | `50` | `0.5453` | `2.1024` | `0.7123` | `0.7115` | `2.1584` |
| Fire lane B | `36` | `-2.3947` | `1.5382` | `-1.5750` | `0.8348` | `3.1896` |

This is training-signal evidence, not a final held-out trained-vs-baseline claim.

The later continuation canary is the cleaner parser/role-metric check:

| Window | Steps | Invalid actions | Orchestrator parse errors | Avg priority rank | Avg priority coverage | Priority effect bonus |
|---|---:|---:|---:|---:|---:|---:|
| Post-fix continuation canary | `350-359` | `0.00%` | `0.00%` | `86.45%` | `78.33%` | `100.00%` |

## One Trace From The Split-Role Run

The trace below is useful because it shows both the intended control surface and the remaining gap.

Observation excerpt from a fire episode:

```json
{
  "round_id": 0,
  "disaster_family": "fire",
  "per_floor_civilians": {
    "floor_0": 10,
    "floor_1": 8,
    "floor_2": 8,
    "floor_3": 8,
    "floor_4": 7
  },
  "per_floor_hazard_severity": {
    "floor_0": 0.0,
    "floor_1": 0.05,
    "floor_2": 0.1333,
    "floor_3": 0.0,
    "floor_4": 0.0
  }
}
```

Baseline-style orchestrator action captured in the trace:

```json
{
  "agent_id": "orchestrator",
  "action_type": "wait",
  "valid": false,
  "parse_status": "arguments_invalid",
  "completion_text": "{\"action_type\":\"evacuate_floor_priority\",\"arguments\":{\"priority_floor\":\"floor_1\"}}"
}
```

Simulator result excerpt:

```json
{
  "orchestrator_action_type": "wait",
  "floor_action_types": {
    "floor_0_agent": "scout",
    "floor_1_agent": "scout",
    "floor_2_agent": "scout",
    "floor_3_agent": "scout",
    "floor_4_agent": "scout"
  },
  "reward_ticker": {
    "orchestrator": -0.1974
  }
}
```

Interpretation: the model reached the correct action family conceptually, but the schema/argument contract was not yet clean enough to turn that into a valid orchestrator directive. This is exactly why the submission separates the mature `3B` specialist evidence from the `7B` orchestration evidence.

## Honest Status

The `7B` orchestrator is smoke/training-signal validated, and the latest continuation canary shows the fixed parser/resume path can run without the previous sticky invalid-action fractions. It is architecturally important because it provides the global coordination layer above fast floor specialists, and the training stack can checkpoint role-specific adapters for it.

The public evidence does **not** yet prove final held-out `7B` orchestrator convergence. The next artifact needed is a trained-vs-baseline orchestrator eval focused on directive success, override usefulness, bottleneck reduction, and invalid orchestrator action rate.

## Next Evaluation Target

The clean next scorecard should report:

- orchestrator invalid action rate
- directive success rate
- override usefulness / override win rate
- bottleneck reduction
- evacuation save-rate delta on held-out seeds
- comparison against baseline and against `3B` specialist-only routing

That would turn the current architecture proof into a legible learned-policy proof.
