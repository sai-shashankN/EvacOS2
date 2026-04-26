# EvacOS2 Hackathon-Day Execution Runbook

This is the operator plan for producing a strong EvacOS2 submission with real trained artifacts.

When the user says `execute HACKATHON_DAY_RUNBOOK.md`, follow this file as the source of truth unless the live user instruction explicitly overrides it.

## Mission

Produce judge-facing proof that EvacOS2 is not just trainable, but can produce meaningfully better behavior:

- Train strong frozen `3B` floor specialists for `fire`, `flood`, and `gas`.
- Train one shared `7B` orchestrator against the frozen specialists.
- Evaluate on held-out seeds across the trained easy/medium/hard operating tiers.
- Publish/collect adapters, metrics CSVs, plots, README links, HF Space, and demo proof.

Current evidence checkpoint:

- `fire`, `flood`, and `gas` 50-step canaries have all completed successfully with `TRAIN_EXIT=0`.
- Invalid-action rates dropped into a healthy low range for all three specialists.
- The remaining bottleneck is runtime/throughput, especially longer flood/gas episode horizons and frequent checkpoint/eval writes.
- Do a throughput-tuned 100-step smoke before launching any longer specialist run.

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

## Codex Harness For ml-intern

`ml-intern` can be used as a top-level Hugging Face / ML specialist lane,
preferably on GPT-5.5 when available. Codex stays the operational harness:
use `ml-intern` for serious ML reasoning, review, HF docs/examples, and patch
recommendations, but do not hand it unapproved direct control over Vast, HF
Jobs, repo mutations, or adapter uploads.

Hermes-agent reference pattern:

- Hermes uses an OpenAI Codex OAuth / ChatGPT-account path rather than a normal OpenAI API key for subscription-backed Codex calls.
- The local `.tmp_external_review/ml-intern` checkout has been patched with the same practical bridge: `openai-codex/...` model ids read the Codex CLI OAuth access token and call the ChatGPT Codex Responses backend.
- Use direct `--runner ml-intern --model openai-codex/gpt-5.5` when we want ml-intern itself to run on the local Codex/ChatGPT subscription session.
- Use `scripts/ml_intern_harness.py --runner codex` only when we want to bypass ml-intern and run a Codex CLI executor in the same scratch-harness style.

Recommended Pro-backed ml-intern runner:

```powershell
python scripts/ml_intern_harness.py "Review the EvacOS2 GRPO reward/eval plan for holes." --context README.md --context HACKATHON_DAY_RUNBOOK.md --runner ml-intern --model openai-codex/gpt-5.5
```

That command is a dry-run: it writes a guarded task file and the exact
`ml-intern` command under `logs/ml_intern/`. To actually launch the
subscription-backed specialist lane, add `--execute`:

```powershell
python scripts/ml_intern_harness.py "Find current HF/TRL/OpenEnv guidance relevant to our training script." --context REMOTE_GPU_SETUP.md --runner ml-intern --model openai-codex/gpt-5.5 --execute
```

Codex CLI runner fallback:

```powershell
python scripts/ml_intern_harness.py "Review the EvacOS2 GRPO reward/eval plan for holes." --context README.md --context HACKATHON_DAY_RUNBOOK.md --runner codex --model gpt-5.5
```

Other direct `ml-intern` providers are still supported when their credentials are configured:

```powershell
python scripts/ml_intern_harness.py "Find current HF/TRL/OpenEnv guidance relevant to our training script." --context REMOTE_GPU_SETUP.md --runner ml-intern --model anthropic/claude-opus-4-6 --max-iterations 8 --execute
```

Default harness rules:

- Run from a scratch workspace, not the repo root.
- Strip `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and `GITHUB_TOKEN` unless explicitly allowed.
- In `--runner codex` mode, strip `OPENAI_*` API env vars unless `--allow-api-key-env` is passed, so the default path uses the signed-in Codex/ChatGPT session instead of accidental API billing.
- Default to the `top-level` profile: serious ML execution/review authority with Codex as final committer.
- Use `--profile read-only` for stricter public-docs-only research.
- Ask for recommendations and patches unless the user explicitly approves autonomous edits.
- Let `ml-intern` ask for YOLO-class actions in `approval_requests.md`; Codex reviews those requests and either executes them separately, asks the user, or re-runs `ml-intern` with a narrow approval scope.
- Keep paid compute, repo uploads, deletes, merges, and HF Job launches under Codex approval.
- Store stdout/stderr in `logs/ml_intern/<run-id>/` for later review.

Subscription/model note:

- Use direct `ml-intern --model openai-codex/gpt-5.5` for ChatGPT Pro-backed GPT-5.5 work when Codex is signed in through ChatGPT.
- Use Codex/`codex exec -m gpt-5.5` as a fallback runner when we want Codex itself, not the ml-intern agent loop.
- Do not use `ml-intern --model openai/gpt-5.5` unless a real OpenAI API key/billing path is intentionally configured.

Known-good remote stack:

```text
torch==2.10.0
torchvision==0.25.0
transformers==4.56.2
trl==0.24.0
peft==0.19.1
fsspec==2025.9.0
vllm==0.10.2 when vLLM is used
Unsloth installed after the baseline deps
```

Why: current Unsloth needs `torch.int1`, but `unsloth_zoo==2026.4.9` still
requires `torch<2.11`; `torch==2.10.0` is the known-good remote midpoint.

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

Use HF GPU credits for the orchestrator/finale, not for 3B specialists.

Target shape:

- 1x A100 80 GB, H100 80 GB, or H200 141 GB
- at least `100 GB` CPU RAM
- at least `150 GB` disk

Selection rule:

- Prefer H100 over H200 when the run fits comfortably in 80 GB VRAM and H100 is cheaper.
- Prefer H200 when memory headroom is the risk, for example longer context, larger batch, or loading the 7B orchestrator plus frozen specialist stack in one process.
- A100 80 GB remains a good budget fallback if H100/H200 availability is poor.

Expected time from our previous A100 evidence:

- `100` steps: about `45 min`
- `300` steps: about `2.5-3 hr`
- `500` steps: about `4-5 hr`
- `750` steps: about `6-7 hr`, only as a stretch
- `1000` steps: about `8-10 hr`, only if curves clearly justify it

With `$30+` HF credits, target `300-500` orchestrator steps first. Continue beyond that only if reward/eval curves are still improving.

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

### Phase 1: Specialist Canary Triad

Status: complete.

Fire was the debugging canary, then flood and gas were run in parallel after the shared reward/parser/oracle fixes landed.

Completed canary configs:

```text
training/config.remote-unsloth-3b-fire-floor-specialist-canary-50.yaml
training/config.remote-unsloth-3b-flood-floor-specialist-canary-50.yaml
training/config.remote-unsloth-3b-gas-floor-specialist-canary-50.yaml
```

Acceptance gate met:

- metrics CSV grows through the run
- checkpoints/artifacts are produced
- invalid action rate trends under `0.05` in the final window
- sampled traces show real routing/rescue behavior, not wait spam or parser exploits
- reward variance stays non-zero enough for GRPO to keep learning

Canary artifact summary:

- Fire: `outputs/vast_fire_canary50_oraclefix_35603298/`
- Flood: `outputs/vast_flood_canary50_35606519/`
- Gas: `outputs/vast_gas_canary50_35606521/`
- Tracked report: `demo/results/specialist_canary50_report.md`

### Phase 2: Throughput-Tuned 100-Step Smoke

Do not jump from green 50-step canaries straight to a full 400-550 step run.

Run a 100-step smoke with the same corrected reward/parser/oracle path, but with cheaper runtime settings.

Disaster-specific rollout horizons:

```text
fire:  max_rounds_per_episode = 4
flood: max_rounds_per_episode = 5
gas:   max_rounds_per_episode = 10
```

Why:

- Fire actually terminated at 4 rounds in the canary.
- Flood actually terminated at 5 rounds in the canary, despite being configured for 10.
- Gas used nearly the full 10-round horizon, so reduce only after a held-out smoke proves 8 rounds is safe.

Use cheaper checkpoint/eval cadence:

```text
save_every_steps: 25 or 50
eval_every_steps: 25 or 50
```

Candidate count:

- Keep `candidates_per_decision = 4` during bootstrap unless a tested config proves `2` keeps reward signal healthy.
- After invalid-action rates stay below `0.05`, a separate resume/smoke may test `2` candidates for speed.
- Do not silently change candidate count in the middle of a paid long run unless the code/config explicitly supports that schedule.

100-step smoke acceptance gate:

- metrics reach final step with `TRAIN_EXIT=0`
- invalid action final-window average stays under `0.08`
- no missing-target or route-action collapse
- raw reward and saved-civilian behavior do not regress against the 50-step canary
- `python scripts/check_grpo_contrast.py <metrics.csv>` passes
- wall-clock estimate for the next run is affordable

### Phase 3: Longer 3B Specialist Curriculum

After the 100-step smoke passes, scale specialists with easy/medium/hard only.

Default recommended specialist schedule:

```text
Stage 1: 150 steps
  150 easy

Stage 2: 150 steps
  120 medium
   30 easy replay

Stage 3: 100 steps
   80 hard
   15 medium replay
    5 easy replay
```

Total: `400` steps per specialist.

Stretch schedule if time, cost, and eval curves are still good:

```text
Stage 1: 200 steps
  200 easy

Stage 2: 200 steps
  160 medium
   40 easy replay

Stage 3: 150 steps
  120 hard
   20 medium replay
   10 easy replay
```

Total: `550` steps per specialist.

Do not include a brutal tier in the default hackathon-day run. It is a stretch research tier only if the trained easy/medium/hard path is already strong and time remains.

Why:

- Early easy steps stabilize action format, parser behavior, and basic rescue behavior.
- Later replay prevents catastrophic forgetting when difficulty increases.
- Hard tasks should matter late, but not erase easier-tier competence.
- This follows the Hermes/GRPO bias toward stable curricula, reward diversity, and avoiding reward collapse.

If the implementation cannot do replay-aware scheduled sampling on a remote clone, use sequential resume blocks as the fallback:

```text
150 easy -> resume
150 medium -> resume
100 hard -> finish
```

But that fallback is weaker. Prefer replay-aware scheduling if we can implement it before the long run.

Expected time must be recalculated from the 100-step smoke. The raw 50-step canaries showed:

```text
fire:  ~35.5 min for 50 steps
flood: ~54.0 min for 50 steps
gas:   ~105.3 min for 50 steps
```

Those numbers are too slow to extrapolate blindly. Use the throughput smoke before buying multi-hour runs.

Important implementation note:

- Real longer-run specialist configs should use `rollout.tier_schedule`.
- `eval.tiers` still controls evaluation only; training difficulty comes from `rollout.tier_schedule`.
- The config loader rejects mismatches where `tier_schedule` does not expand to `max_steps`, so a remote run should fail fast instead of silently training the wrong curriculum.

Minimum viable specialist execution:

1. Create or verify the disaster-specific 100-step smoke config first.
2. Confirm it trains only `floor_agent`.
3. Confirm `orchestrator_policy: "stub"`.
4. Confirm `rollout.disaster_families` contains exactly one family.
5. Confirm checkpoint root and metrics CSV are disaster-specific.
6. Confirm tier schedule is real for longer configs, not just documentation.

Remote command pattern:

```bash
cd /workspace/EvacOS2
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export WANDB_DISABLED=true
python -c "from pathlib import Path; from training.train import run_training; run_training(Path('training/config.remote-unsloth-3b-fire-floor-specialist-throughput-smoke-100.yaml'))"
```

### Phase 3 Monitoring

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

Specialist smoke acceptance gate:

- reaches the planned step count cleanly
- invalid action rate stays low
- CSV/checkpoint/export are healthy
- no obvious reward hacking in sampled actions
- reward/eval signal is not flatlining badly

If any specialist reveals a bug:

- stop that run if the bug affects reward validity, parsing, checkpointing, or CSV integrity
- patch locally
- run targeted tests
- redeploy patch before scaling the other specialists
- restart from scratch if the bug polluted rewards/checkpoints
- resume only if the fix is clearly non-behavioral

### Phase 4: Specialist Artifact Collection

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

### Phase 5: Specialist Evaluation

Run held-out fixed-suite eval for each specialist.

Evaluate trained tiers by default:

```text
easy, medium, hard
```

Only evaluate `brutal` as an optional stretch/diagnostic if the model was actually trained or explicitly tested there. Do not make brutal-tier claims in judge-facing material without evidence.

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

## Phase 6: 7B Orchestrator Over Frozen Specialists

Do this only after the Critical Preflight Gate passes.

Goal:

```text
train one shared 7B orchestrator while frozen floor specialists handle local actions
```

Target training:

- start with `150-300` steps if debugging
- target `300-500` steps for a strong final
- continue beyond `500` only if reward/eval curves are still improving and HF credits remain

Training mix:

- The orchestrator should train on mixed/interleaved `fire`, `flood`, and `gas`, not one giant fire block followed by one giant flood block.
- Reason: the 7B is learning global coordination and when/how to rely on each frozen floor specialist. That skill is inherently cross-disaster.
- The frozen floor specialists are already stable, so the orchestrator does not need disaster-by-disaster local bootstrapping in the same way the 3B specialists do.
- Large single-disaster blocks raise forgetting risk and make the 7B overfit to one specialist's behavior before seeing the others.

Default 400-step orchestrator schedule:

```text
Stage 1: 100 steps
  easy mixed fire/flood/gas, roughly round-robin

Stage 2: 150 steps
  120 medium mixed fire/flood/gas
   30 easy replay mixed fire/flood/gas

Stage 3: 150 steps
  120 hard mixed fire/flood/gas
   20 medium replay mixed fire/flood/gas
   10 easy replay mixed fire/flood/gas
```

For a 500-step extension, continue with:

```text
 80 hard mixed fire/flood/gas
 15 medium replay mixed fire/flood/gas
  5 easy replay mixed fire/flood/gas
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

## Phase 7: Publish Adapters

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

## Phase 8: Final Demo Bundle

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

## Phase 9: Submission Lock

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

Current reality after canaries:

```text
fire canary-50:  ~35.5 min
flood canary-50: ~54.0 min
gas canary-50:   ~105.3 min
```

Next estimate should come from the 100-step throughput smoke, not a naive 750-step extrapolation.

Planning placeholder:

```text
100-step smoke, all three specialists in parallel: roughly 1.5-3 hours, gas likely limiting
400-step specialist curriculum after tuning: recalculate from smoke before renting
orchestrator 300-500 steps: roughly 2.5-5 hours on A100/H100/H200 class GPU
final eval/submission: 1-2 hours
```

Total practical hackathon-day runtime after canaries:

```text
unknown until throughput smoke; do not promise 4-6 hour specialist completion yet
```

The recommended path is now throughput smoke first because correctness is proven and speed is the risk.

## Expected Cost

Specialists on Vast:

```text
100-step smoke on three RTX 4090s: budget about $2-4 depending gas runtime and offer price
400-step specialist curriculum: recalculate from smoke; do not pre-commit to a fixed dollar number
```

Orchestrator on HF A100/H100/H200 class GPU:

```text
300 steps ~= $7-10 on A100-class pricing, more on H100/H200
500 steps ~= $10-13
750+ steps only if curves justify it
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
