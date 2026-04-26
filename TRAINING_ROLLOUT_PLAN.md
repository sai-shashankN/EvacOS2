# EvacOS2 Training Rollout Plan

This is the short, practical plan for the next training wave after the fixed
fire/flood/gas 50-step canaries.

## Current Known-Good Checkpoint

The fire, flood, and gas 50-step canaries completed successfully.

- Fire: `TRAIN_EXIT=0`, final step `49/50`, last-10 invalid `0.03462`.
- Flood: `TRAIN_EXIT=0`, final step `49/50`, last-10 invalid `0.02463`.
- Gas: `TRAIN_EXIT=0`, final step `49/50`, last-10 invalid `0.02871`.
- All three: route action rate `1.0`, missing target rate `0.0`, latest checkpoint present.
- Tracked summary: `demo/results/specialist_canary50_report.md`.
- Local artifacts:
  - `outputs/vast_fire_canary50_oraclefix_35603298/fire_canary50_artifacts.tgz`
  - `outputs/vast_flood_canary50_35606519/flood_canary50_artifacts.tgz`
  - `outputs/vast_gas_canary50_35606521/gas_canary50_artifacts.tgz`

This proves the RL/training plumbing is now alive: the floor model receives
usable observations, the candidate groups have valid contrast, route arguments
are valid, and the trainer no longer crashes on long prompt windows.

It does **not** prove the final model is strong yet. It proves we can now train
for strength.

## Throughput Finding

The canaries also exposed the next bottleneck:

```text
fire:  35.46 min total,  42.6 sec/step, max_rounds_per_episode=4
flood: 54.03 min total,  64.8 sec/step, max_rounds_per_episode=10
gas:  105.27 min total, 126.3 sec/step, max_rounds_per_episode=10
```

Trace summaries show flood episodes actually used `5` rounds, while gas usually
used `8-10`. Before any quality run, use disaster-specific horizons:

- fire: `max_rounds_per_episode: 4`
- flood: `max_rounds_per_episode: 5`
- gas: `max_rounds_per_episode: 10`, or `8` only if a held-out gas smoke proves it is safe

Also reduce long-run checkpoint/eval overhead:

- canaries: checkpoint/eval every `10` steps
- long runs: checkpoint/eval every `25` or `50` steps
- keep enough checkpoints for rollback, but do not write six 470MB artifacts every few minutes unless debugging

## Immediate Commit Gate

Before renting more GPUs:

1. Review the working tree and stage only intentional source/docs changes.
2. Run focused tests for the current fixes.
3. Run `openenv validate . --json`.
4. Commit and push.

Suggested local verification:

```bash
python -m pytest tests/test_visibility.py tests/test_prompts.py tests/test_config_schema.py tests/test_group_for_grpo.py tests/test_rollout_collector.py tests/test_train_build_grpo_trainer.py tests/test_train_tokenize_batch.py tests/test_round_protocol.py -q
openenv validate . --json
```

Do not launch long runs from an unpushed local-only fix unless explicitly doing
a disposable smoke test.

## Phase 1: Flood And Gas Canary Runs

Run cheap 50-step canaries for flood and gas before any long training.

The fire fixes should transfer because all three specialists share:

- `evacos_ma/observability.py`
- `training/prompts.py`
- `training/rollout.py`
- `training/train.py`
- route action parsing and reward metrics

But flood/gas can still reveal disaster-specific reward or action issues, so
they need their own proof runs.

### Canary Pass Criteria

A 50-step specialist canary is acceptable only if:

- `TRAIN_EXIT=0`
- metrics CSV has `50` rows
- latest checkpoint exists
- `floor_route_action_rate` is consistently high, ideally `~1.0`
- `floor_route_missing_target_rate` is `0.0` or near-zero
- `active_empty_args_rate` is `0.0` or near-zero
- last-10 invalid action average is below `0.10`
- no watchdog crash, no sequence-length crash, no pure-wait collapse

Status: complete. Both flood and gas passed. If future prompt/reward changes land,
rerun the 50-step canaries before launching long runs.

## Phase 2: Throughput-Tuned 100-Step Smoke

Before a full specialist run, run the three throughput-tuned `100`-step smokes.

Recommended smoke set:

- fire: `training/config.remote-unsloth-3b-fire-floor-specialist-throughput-smoke-100.yaml`
- flood: `training/config.remote-unsloth-3b-flood-floor-specialist-throughput-smoke-100.yaml`
- gas: `training/config.remote-unsloth-3b-gas-floor-specialist-throughput-smoke-100.yaml`
- checkpoint/eval every `25` steps
- `candidates_per_floor_prompt: 4`
- keep oracle candidate enabled for this smoke
- disaster-specific horizon from the throughput finding above

Pass criteria:

- `TRAIN_EXIT=0`
- all `100` rows written
- average step time is materially lower than the canary
- route/missing-target metrics stay healthy
- invalid rate remains below `0.10`, ideally below `0.05`
- `python scripts/check_grpo_contrast.py <metrics.csv>` passes
- latest checkpoint exists and reloads

## Phase 3: Longer 3B Floor Specialist Runs

After fire/flood/gas all pass 50-step canaries, run longer training.

Active submission plan:

```text
fire:  400 steps
flood: 500 steps
gas:   700 steps
```

Use these configs:

- fire: `training/config.remote-unsloth-3b-fire-floor-specialist-quality-400.yaml`
- flood: `training/config.remote-unsloth-3b-flood-floor-specialist-quality-500.yaml`
- gas: `training/config.remote-unsloth-3b-gas-floor-specialist-quality-700.yaml`

Do **not** expand the active submission scope into extra scenario tiers before
the deadline. The faster win is to make the default fire/flood/gas specialist
suite look strong, clean, and well-evaluated.

### Long-Run Pass Criteria

For each specialist:

- train artifact downloaded locally
- metrics CSV downloaded locally
- checkpoint exists
- final and last-10 invalid action rate remain low
- route/missing-target metrics stay healthy
- held-out eval score improves vs baseline

## Phase 4: 7B Orchestrator Smoke

Do not wait for perfect floor specialists before testing the 7B path.

Run the 7B orchestrator with frozen floor specialists progressively:

1. Smoke with the 50-step fire specialist and matching flood/gas canary adapters.
2. Repeat with 100-step or 150-step floor adapters if available.
3. Repeat with final quality-run floor adapters.

The point is to catch integration bugs early. The 7B does not need final floor
models to prove that routing, frozen adapters, and orchestration training work.

### 7B Job Definition

The 7B orchestrator should:

- reason across floors
- set priorities
- resolve cross-floor conflicts
- coordinate evacuation flow
- use deterministic disaster routing rather than guessing fire/flood/gas
- train against frozen 3B floor responders

The 3Bs are fast floor responders. The 7B is the slower global coordinator.

## Phase 5: Final Training Story

The submission story should separate three claims:

1. **Environment claim:** EvacOS2 is a multi-agent emergency-response environment
   with partial observability, floor specialists, and global orchestration.
2. **Training claim:** GRPO training improves floor-agent action quality, shown by
   route validity, invalid-action reduction, reward curves, and held-out eval.
3. **Architecture claim:** Small 3B models handle fast local floor response while
   a larger 7B model handles slower cross-floor planning.

Avoid claiming theoretical convergence. Aim for:

- clear reward improvement
- 0-100 eval improvement
- low invalid actions
- readable plots
- before/after examples

## GPU Plan

For 3B specialists:

- preferred: cheap 3090/4090/A5000-class GPUs
- minimum practical VRAM: 24 GB
- 4090 is faster but not strictly required if cost matters
- run fire first as the canary domain
- then run flood and gas in parallel only after canaries pass

For 7B orchestrator:

- preferred: A100 80GB or H100/H200 if Hugging Face credits are available
- H100 is probably the better default than H200 for cost unless sequence length
  or batch size requires the extra H200 memory
- keep 3B specialists frozen during 7B training

## Monitoring Rules

Monitor every 5 minutes for active paid GPU runs.

Notify or intervene on:

- no metrics growth for more than 10 minutes after training starts
- process death
- GPU idle while process is alive for more than 10 minutes after metrics start
- disk usage above 80%
- invalid action regression above the canary threshold
- route/missing-target metrics collapse
- checkpoint movement stalls
- final artifact ready

After successful artifact download, destroy the Vast instance immediately.

## Decision Gates

Use this exact flow:

```text
commit/push fixes
-> fire/flood/gas 50-step canaries passed
-> create canary report
-> run throughput-tuned 100-step smoke
-> if smoke passes, run longer 3B specialist training
-> run 7B orchestrator smoke with current frozen specialists
-> replace frozen specialists with stronger checkpoints as they arrive
-> final held-out eval and plots
```

If a longer run regresses badly, keep the canary artifact as proof that the
system works and debug the curriculum or reward before spending more compute.
