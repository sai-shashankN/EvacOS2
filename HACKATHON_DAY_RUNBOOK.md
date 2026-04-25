# EvacOS2 Hackathon-Day Execution Runbook

This is the operator plan for producing a strong EvacOS2 submission with real trained artifacts.

When the user says `execute HACKATHON_DAY_RUNBOOK.md`, follow this file as the source of truth unless the live user instruction explicitly overrides it.

## Mission

Produce judge-facing proof that EvacOS2 is not just trainable, but can produce meaningfully better behavior:

- Train strong frozen `3B` floor specialists for `fire`, `flood`, and `gas`.
- Train one shared `7B` orchestrator against the frozen specialists.
- Evaluate on held-out seeds and all tiers.
- Publish/collect adapters, metrics CSVs, plots, README links, HF Space, and demo proof.

## Non-Negotiable Architecture Choice

Freeze the floor specialists during orchestrator training.

Do not jointly update the `7B` orchestrator and `3B` floor specialists in the hackathon run.

Reason:

- Frozen specialists make the 7B training target stable.
- The story is cleaner: fast local specialists plus a slower global coordinator.
- It avoids floor-agent forgetting/drift.
- It reduces cost and debugging risk.

## Hermes MLOps Rules

Use the `hermes-mlops` skill before touching any of:

- TRL / GRPO
- Unsloth
- PEFT / LoRA
- vLLM
- Hugging Face Hub / Spaces / adapter upload
- checkpointing
- training metrics
- GPU setup
- evaluation artifacts

Hermes-aligned operating principles:

- Use the smallest model that can learn for specialist lanes.
- Start with a stable canary before parallelizing.
- Monitor reward mean and reward variance, not just loss.
- Treat rising reward without behavior/eval improvement as suspicious.
- Keep LoRA adapters out of Git; upload or download them as artifacts.
- Pin the known-good stack before long runs.

Known-good remote stack:

```text
transformers==4.56.2
trl==0.24.0
peft==0.19.1
vllm==0.10.2 when vLLM is used
Unsloth installed after the baseline deps
```

## GPU Plan

### 3B Specialist GPUs

Use cheap reliable 24 GB GPUs.

Acceptable choices:

- RTX 3090 24 GB
- RTX 4090 24 GB
- RTX A5000 24 GB
- A10 / A10G 24 GB

Minimum instance requirements:

- Verified host preferred.
- Reliability at least `96%`, ideally `99%+`.
- Disk at least `150 GB`.
- CUDA compatible with current image.
- Max duration at least `8 hours`.
- SSH/Jupyter access.
- No hidden giant bandwidth fees.

Preferred price targets:

- RTX 3090: ideally under `$0.25/hr`.
- RTX 4090: ideally under `$0.35/hr`.
- A5000/A10/A10G: only if price is competitive and reliability/disk are good.

Expected cost:

- Three 4090s for 5 hours: about `$4.85`, budget `$5-6`.
- Three 3090s for 6 hours may be similar or cheaper depending on offers.

### 7B Orchestrator GPU

Use HF A100 large credits for the orchestrator/finale, not for 3B specialists.

Target shape:

- 1x A100 80 GB
- at least `100 GB` CPU RAM
- at least `150 GB` disk

Expected time from our previous A100 evidence:

- `100` steps: about `45 min`
- `300` steps: about `2.5-3 hr`
- `500` steps: about `4-5 hr`
- `750` steps: about `6-7 hr`
- `1000` steps: about `8-10 hr`

With `$30` HF credits, target `500` orchestrator steps first. Continue toward `750-1000` only if reward/eval curves are still improving.

## Critical Preflight Gate

Before spending A100 credits, confirm the code can train a 7B orchestrator against the actual frozen specialist stack.

Current capability after the routed-specialist layer:

- `roles.trainable: ["orchestrator"]`
- `roles.frozen_floor_specialist_adapter_paths.fire: <fire adapter path>`
- `roles.frozen_floor_specialist_adapter_paths.flood: <flood adapter path>`
- `roles.frozen_floor_specialist_adapter_paths.gas: <gas adapter path>`
- optional fallback: `roles.frozen_adapter_paths.floor_agent: <generalist adapter path>`

Required final behavior:

- The 7B orchestrator interacts with a frozen floor policy that routes each floor-agent call to the trained `fire`, `flood`, or `gas` specialist based on the prompt/scenario disaster family.

Plain English:

- `roles.frozen_adapter_paths.floor_agent: <path>` alone means "load one frozen floor-agent LoRA adapter."
- `roles.frozen_floor_specialist_adapter_paths` means "load three frozen floor-agent LoRA adapters and switch deterministically by disaster family."
- Before A100 training, verify the run uses the routed-specialist config or a real combined generalist fallback, not just one single-disaster floor adapter.
- If the 7B only sees the fire floor adapter, then the run is a fire-orchestrator run, not a fire/flood/gas orchestrator run.

If routed specialists fail under verification, use one of these before A100 training:

1. Preferred: fix the scope-routed frozen floor policy that loads `fire`, `flood`, and `gas` floor adapters and chooses by disaster family.
2. Acceptable: one combined frozen floor adapter produced by continuing training across fire/flood/gas, but label it as a combined floor generalist rather than three separate specialists.
3. Fallback: train/evaluate the 7B against the best single frozen floor specialist only, but label it honestly as a fallback.
4. Do not pretend one `floor_agent` adapter path means all three specialists are active.

This gate matters. Do not burn A100 credits until the frozen specialist behavior matches the story.

## Local Preflight

Run these before renting anything:

```powershell
python -m pytest tests/test_config_schema.py tests/test_train_build_grpo_trainer.py tests/test_baseline_vs_trained_yaml_optional.py tests/test_specialist_configs.py tests/test_reward_normalization.py tests/test_openenv_shell.py -q
openenv validate . --json
python -m compileall training scripts evacos_ma evaluation
git status --short
```

Expected:

- targeted tests pass
- OpenEnv validation passes
- compileall passes
- no accidental large artifacts staged

If this fails, fix locally before renting.

## Training Order

### Phase 1: Fire Specialist Canary

Train fire alone first.

Reason:

- Fire is our debugging canary.
- If parser/reward/checkpoint/CSV issues show up, fix them once before duplicating the mistake into gas/flood.
- Fire already showed positive early signal, so it is the lowest-risk place to validate the long run.

Target:

```text
200 easy + 200 medium + 200 hard + 150 brutal = 750 steps
```

This is a staged curriculum budget, not naive "forget the old tier forever" training.
Replay tiers are interleaved inside each stage rather than clumped at the beginning.

Default exact-750 specialist schedule:

```text
Stage 1: 200 steps
  200 easy

Stage 2: 200 steps
  160 medium
   40 easy replay

Stage 3: 200 steps
  160 hard
   30 medium replay
   10 easy replay

Stage 4: 150 steps
  115 brutal
   25 hard replay
   10 medium replay
```

Why:

- Early easy steps stabilize action format, parser behavior, and basic rescue behavior.
- Later replay prevents catastrophic forgetting when difficulty increases.
- The hardest tier should dominate late training, but not erase easier-tier competence.
- This follows the Hermes/GRPO bias toward stable curricula, reward diversity, and avoiding reward collapse.

If the implementation cannot do replay-aware scheduled sampling on a remote clone, use sequential resume blocks as the fallback:

```text
200 easy -> resume
200 medium -> resume
200 hard -> resume
150 brutal -> finish
```

But that fallback is weaker. Prefer replay-aware scheduling if we can implement it before the long run.

Expected time:

- RTX 4090: `4-5 hr`
- RTX 3090: `5-6 hr`

Important implementation note:

- Real 750-step specialist configs now use `rollout.tier_schedule`.
- `eval.tiers` still controls evaluation only; training difficulty comes from `rollout.tier_schedule`.
- The config loader rejects mismatches where `tier_schedule` does not expand to `max_steps`, so a remote run should fail fast instead of silently training the wrong curriculum.

Minimum viable fire execution:

1. Create or verify `training/config.remote-unsloth-3b-fire-floor-specialist-750.yaml`.
2. Confirm it trains only `floor_agent`.
3. Confirm `orchestrator_policy: "stub"`.
4. Confirm `rollout.disaster_families: ["fire"]`.
5. Confirm checkpoint root and metrics CSV are fire-specific.
6. Confirm tier schedule is real, not just documentation.

Remote command pattern:

```bash
cd /workspace/EvacOS2
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
python -c "from pathlib import Path; from training.train import run_training; run_training(Path('training/config.remote-unsloth-3b-fire-floor-specialist-750.yaml'))"
```

### Phase 1 Monitoring

Monitor every `5 minutes` during the first `30 minutes`, then every `10-15 minutes` if stable.

Notify/intervene if:

- process dies
- no metrics CSV growth for more than `10 minutes`
- GPU idle for more than `10 minutes` while process is alive
- disk above `80%`
- invalid action rate exceeds `0.35`
- raw/norm reward std collapses near zero for multiple windows
- reward rises but saved civilians / raw reward / held-out eval does not improve
- checkpoint does not move
- CSV header mismatch appears
- any OOM, CUDA, PEFT, TRL, Unsloth, vLLM import error appears

Track at minimum:

- `step`
- `wall_seconds`
- `tier_mix`
- `mean_raw_reward_floor`
- `mean_norm_reward_floor`
- `raw_reward_std_floor`
- `norm_reward_std_floor`
- `invalid_action_rate`
- `wait_rate`
- `active_empty_args_rate`
- latest checkpoint
- GPU VRAM/utilization

Fire acceptance gate:

- reaches at least `300` steps cleanly
- invalid action rate stays low
- CSV/checkpoint/export are healthy
- no obvious reward hacking in sampled actions
- reward/eval signal is not flatlining badly

If fire reveals a bug:

- stop fire if the bug affects reward validity, parsing, checkpointing, or CSV integrity
- patch locally
- run targeted tests
- redeploy patch before gas/flood
- restart from scratch if the bug polluted rewards/checkpoints
- resume only if the fix is clearly non-behavioral

### Phase 2: Gas And Flood Parallel

After fire is stable or complete, run gas and flood in parallel on two separate 24 GB GPUs.

Reason:

- Fire catches bugs first.
- Gas/flood benefit from the fixes.
- Parallel execution saves wall-clock time.

Target per specialist:

```text
200 easy + 200 medium + 200 hard + 150 brutal = 750 steps
```

Use the same replay-aware schedule as fire:

```text
200 easy
160 medium + 40 easy replay
160 hard + 30 medium replay + 10 easy replay
115 brutal + 25 hard replay + 10 medium replay
```

Expected time:

- RTX 4090: `4-5 hr`
- RTX 3090: `5-6 hr`

Required configs:

- `training/config.remote-unsloth-3b-gas-floor-specialist-750.yaml`
- `training/config.remote-unsloth-3b-flood-floor-specialist-750.yaml`

Each must have:

- one disaster family only
- trainable `floor_agent` only
- stub orchestrator
- unique checkpoint root
- unique metrics CSV
- real tier schedule

Remote commands:

```bash
python -c "from pathlib import Path; from training.train import run_training; run_training(Path('training/config.remote-unsloth-3b-gas-floor-specialist-750.yaml'))"
```

```bash
python -c "from pathlib import Path; from training.train import run_training; run_training(Path('training/config.remote-unsloth-3b-flood-floor-specialist-750.yaml'))"
```

Monitor both every `5-10 minutes`.

Use the same intervention rules as fire.

### Phase 3: Specialist Artifact Collection

For each specialist, collect:

- `latest/lora_adapter`
- latest `meta.json`
- metrics CSV
- JSONL logs if reasonably sized
- generated report JSON
- plots
- exact config used
- commit SHA

Local destination:

```text
outputs/vast_specialists_<date_or_instance>/
```

Suggested layout:

```text
outputs/vast_specialists_<run_id>/
  fire/
    lora_adapter/
    metrics.csv
    report.json
    config.yaml
  flood/
    lora_adapter/
    metrics.csv
    report.json
    config.yaml
  gas/
    lora_adapter/
    metrics.csv
    report.json
    config.yaml
```

After local verification, destroy the Vast instances immediately.

Do not leave stopped storage running unless artifact recovery is still unresolved.

### Phase 4: Specialist Evaluation

Run held-out fixed-suite eval for each specialist.

Evaluate all tiers:

```text
easy, medium, hard, brutal
```

Use held-out seeds:

```text
42, 123, 456, 789, 1024
```

Judge-facing metrics:

- save rate
- casualties / lost civilians
- raw reward
- normalized reward
- invalid action rate
- wait rate
- wall-clock / rollout rounds

Generate readable plots:

- training reward vs step
- invalid action rate vs step
- raw reward std vs step
- before/after bars
- outcome breakdown

Commit lightweight plots/CSVs to `demo/results/`, not giant adapters.

## Phase 5: 7B Orchestrator Over Frozen Specialists

Do this only after the Critical Preflight Gate passes.

Goal:

```text
train one shared 7B orchestrator while frozen floor specialists handle local actions
```

Target training:

- start with `300` steps if debugging
- target `500` steps for strong final
- continue to `750-1000` only if curves improve

Training mix:

- The orchestrator should train on mixed/interleaved `fire`, `flood`, and `gas`, not one giant fire block followed by one giant flood block.
- Reason: the 7B is learning global coordination and when/how to rely on each frozen floor specialist. That skill is inherently cross-disaster.
- The frozen floor specialists are already stable, so the orchestrator does not need disaster-by-disaster local bootstrapping in the same way the 3B specialists do.
- Large single-disaster blocks raise forgetting risk and make the 7B overfit to one specialist's behavior before seeing the others.

Default 500-step orchestrator schedule:

```text
Stage 1: 90 steps
  easy mixed fire/flood/gas, roughly round-robin

Stage 2: 140 steps
  112 medium mixed fire/flood/gas
   28 easy replay mixed fire/flood/gas

Stage 3: 150 steps
  120 hard mixed fire/flood/gas
   20 medium replay mixed fire/flood/gas
   10 easy replay mixed fire/flood/gas

Stage 4: 120 steps
   90 brutal mixed fire/flood/gas
   20 hard replay mixed fire/flood/gas
   10 medium replay mixed fire/flood/gas
```

For a 750-step extension, continue with:

```text
175 brutal mixed fire/flood/gas
 50 hard replay mixed fire/flood/gas
 25 medium replay mixed fire/flood/gas
```

Do not let the family mix collapse. Each tier block should stay close to one-third `fire`, one-third `flood`, one-third `gas`, unless a live bug forces narrowing.

HF A100 command pattern:

```bash
cd /workspace/EvacOS2
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
python -c "from pathlib import Path; from training.train import run_training; run_training(Path('training/config.remote-unsloth-7b-orchestrator-frozen-specialists.yaml'))"
```

Required config behavior:

- `model.orchestrator_base: "Qwen/Qwen2.5-7B-Instruct"`
- floor specialists remain frozen
- `roles.trainable: ["orchestrator"]`
- `roles.frozen_floor_specialist_adapter_paths.fire/flood/gas` point at real downloaded specialist adapters
- mixed disaster curriculum includes `fire`, `flood`, `gas`
- checkpoint includes enough adapter metadata to evaluate elsewhere
- metrics distinguish orchestrator and floor roles

Start from `training/config.remote-unsloth-7b-orchestrator-frozen-specialists.example.yaml`, replace adapter paths, then save the runnable copy without `.example`.

Monitor:

- `mean_raw_reward_orch`
- `mean_norm_reward_orch`
- `raw_reward_std_orch`
- `norm_reward_std_orch`
- `orchestrator_loss`
- `orchestrator_kl_loss`
- `orchestrator_ratio_mean`
- `override_win_rate`
- `invalid_action_rate`
- save rate / casualties in eval

Stop/repair if:

- orchestrator reward collapses
- KL spikes repeatedly
- ratio stats show unstable updates
- override behavior becomes degenerate
- floor specialists are accidentally trainable
- checkpoint lacks frozen specialist adapter metadata

## Phase 6: Publish Adapters

Use adapter upload, not Git.

Example:

```bash
export HF_ADAPTER_REPO=your-username/evacos2-fire-3b-floor
python scripts/upload_adapter.py \
  outputs/vast_specialists_<run_id>/fire/lora_adapter \
  "$HF_ADAPTER_REPO"
```

Repeat for:

- fire 3B floor specialist
- flood 3B floor specialist
- gas 3B floor specialist
- final 7B orchestrator adapter

If using one repo, upload under separate paths:

```bash
python scripts/upload_adapter.py outputs/.../fire/lora_adapter your-username/evacos2-adapters --path-in-repo fire-3b-floor
```

## Phase 7: Final Demo Bundle

After selected adapters exist locally or on HF:

```powershell
python -m evaluation.demo_bundle --skip-trained --output-dir outputs/demo_bundle_baseline
```

Then run trained comparison with the selected final checkpoint:

```powershell
python -m evaluation.demo_bundle `
  --trained-checkpoint "path/to/final/lora_adapter" `
  --config training/config.remote-unsloth-7b-orchestrator-frozen-specialists.yaml `
  --output-dir outputs/demo_bundle_final
```

Copy lightweight final outputs into `demo/results/`.

Required public proof:

- training curve
- baseline vs trained bar chart
- scenario outcome breakdown
- scorecard markdown
- exact config used
- adapter location
- HF Space URL
- YouTube/HF blog links

## Phase 8: Submission Lock

Final checks:

```powershell
python -m pytest tests/test_config_schema.py tests/test_train_build_grpo_trainer.py tests/test_baseline_vs_trained_yaml_optional.py tests/test_specialist_configs.py tests/test_reward_normalization.py tests/test_openenv_shell.py -q
openenv validate . --json
python -m compileall training scripts evacos_ma evaluation
```

Then:

- verify live HF Space matches `openenv.yaml`
- verify README links work
- verify video/blog URLs are real, not placeholders
- verify no secrets are committed
- verify no large adapters are committed
- verify `demo/results/` has the final plots
- push GitHub
- redeploy/update HF Space

## GPU Buying Safety Rule

Do not rent an instance unless one of these is true:

- the user explicitly gives an offer id and says to rent it
- the user says to execute this runbook and has already approved renting by budget

Before buying, report:

- offer id
- GPU type
- VRAM
- disk size
- price per hour
- expected total cost
- max duration
- reliability
- whether it satisfies the runbook criteria

If disk is under `150 GB`, reject it.

If the offer is unverified or low reliability, reject it unless the user explicitly accepts the risk.

## Expected Total Wall Clock

If using one fire canary GPU first, then two parallel GPUs:

```text
fire: 4-6 hours
gas + flood parallel: 4-6 hours
specialist total wall clock: 8-12 hours
orchestrator 500 steps: 4-5 hours
final eval/submission: 1-2 hours
```

Total practical hackathon-day runtime:

```text
13-19 hours, depending on bugs and GPU speed
```

If all three specialists run in parallel immediately:

```text
specialist total wall clock: 4-6 hours
```

But the recommended path is fire canary first because it reduces the risk of duplicating a bug across three paid runs.

## Expected Cost

Specialists on Vast:

```text
3 x RTX 4090 x 5 hr ~= $4.85
Budget: $5-7
```

Orchestrator on HF A100 large:

```text
500 steps ~= $10-13
750 steps ~= $15-18
1000 steps ~= $20-25
```

With `$30` HF credits, spend HF on the A100 orchestrator/finale only.

## Stop Conditions

Stop and fix before continuing if:

- OpenEnv validation fails
- parser errors return
- invalid action rate remains above `0.35`
- checkpoints are missing or corrupt
- metrics CSV stops growing
- reward variance collapses
- raw reward worsens while normalized reward improves suspiciously
- artifacts cannot be downloaded
- HF Space contract differs from GitHub
- final trained comparison cannot be reproduced

## Final Pitch Shape

The final story should be:

```text
EvacOS2 trains LLM agents to coordinate evacuations in multi-floor disaster simulations.
We first train fast 3B floor specialists for fire, flood, and gas.
Then we freeze them and train a stronger 7B orchestrator to coordinate global decisions.
The environment exposes OpenEnv reset/step/state/schema endpoints, rewards saved civilians and coordination, and penalizes invalid or harmful actions.
Here is baseline behavior.
Here is trained behavior.
Here are reward curves and held-out before/after metrics.
```

Do not oversell what was not trained.

If the final 7B only trained against one frozen floor specialist, say so.

If all three frozen specialists are actually active through a routed floor policy, say that clearly and show it.
