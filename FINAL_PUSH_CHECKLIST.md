# Final Push Checklist

This is the active Sunday checklist for taking EvacOS2 from "strong system" to "strong submission."

The goal is no longer to prove the stack exists.
The goal is to show **clean, undeniable improvement** with minimal inconsistency.

## Must Fix

- [ ] Fix the OpenEnv manifest/schema contract mismatch
- [x] Unify difficulty naming around `brutal` for the hardest tier
- [ ] Make evaluation config-safe for split-role checkpoints
- [ ] Generate one clean baseline-vs-trained evidence bundle
- [ ] Lock one deterministic demo scenario
- [ ] Update README / brief / demo story after the run

## Do Not Waste Time On

- [ ] Adding new benchmark modes
- [ ] Large refactors
- [ ] More architecture polish
- [ ] More training features
- [ ] Overselling orchestrator convergence

## 1. Contract Consistency

Goal: remove easy judge attacks caused by naming or interface drift.

- [ ] `evacos_ma.openenv.manifest` should reference the schema the runtime actually accepts
- [ ] `/openenv/*` should be clearly identified as the canonical benchmark surface
- [ ] Root-level `/reset`, `/step`, `/state`, `/health`, `/schema`, `/metadata` should be described as legacy/local helpers if retained
- [ ] Public difficulty labels should consistently use one vocabulary
- [ ] README, `openenv.yaml`, task registry, procgen paths, and rendered docs should agree

Success bar:
- [ ] A reviewer comparing manifest, schema, README, and runtime does not see contradictions

## 2. Evaluation Safety

Goal: ensure the evaluation run is scoring the checkpoint you actually trained, not silently assuming `training/config.yaml`.

- [ ] `evaluation/baseline_vs_trained.py` must not rely on the wrong default config for split-role runs
- [ ] Evaluation should accept explicit config/model routing, or resolve it from checkpoint metadata
- [ ] Split-role checkpoints should evaluate without hidden assumptions about shared-model layout
- [ ] Baseline-vs-trained output should remain reproducible across reruns

Success bar:
- [ ] You can point at a checkpoint and trust that the reported comparison matches that checkpoint's actual model layout

## 3. Proof Bundle

Goal: produce the evidence that actually wins.

### Required artifacts

- [ ] `baseline_vs_trained.csv`
- [ ] `submission_scorecard.md`
- [ ] `submission_scorecard.json`
- [ ] `demo_bundle_summary.md`
- [ ] plots for the final selected checkpoint

### Required story

- [ ] Graph 1: performance vs training progress
- [ ] Graph 2: baseline vs trained comparison
- [ ] Graph 3: scenario outcome breakdown
- [ ] At least one visible behavior change in the chosen demo scenario

### Required metric framing

- [ ] One main headline metric judges can understand immediately
- [ ] One operational metric such as casualties / safe evacuations / evacuation time
- [ ] One role-aware diagnostic if useful, especially `floor_agent` vs `orchestrator`

Success bar:
- [ ] A judge can understand "same environment, trained agent performs better" in under 30 seconds

## 4. Demo Lock

Goal: remove randomness from the live pitch.

- [ ] Pick one disaster family
- [ ] Pick one seed
- [ ] Pick one layout / reset path
- [ ] Use the same scenario for before/after comparison
- [ ] Rehearse the exact flow without improvising

Demo rule:
- [ ] Do not rely on live procgen randomness during the actual pitch

Success bar:
- [ ] The demo is reproducible and confidence-building, not surprising

## 5. Messaging

Goal: sound credible, not defensive.

- [ ] Say floor agents are currently the clearest improvement signal
- [ ] Present orchestrator as the longer-horizon coordination challenge
- [ ] Do not claim full convergence if you do not have it
- [ ] Keep the story outcome-first, not complexity-first

Recommended framing:

- [ ] "Baseline struggles with coordination and hazard response"
- [ ] "After training, floor agents learn localized evacuation strategies"
- [ ] "EvacOS measures meaningful improvement, not just simulated scenarios"

Avoid:

- [ ] "We're still training" vibes
- [ ] log-dump demos
- [ ] architecture-overload explanations
- [ ] claiming more than the run proves

## 6. Repo Polish

Goal: make the public repo feel finished.

- [ ] README should match the actual repo state exactly
- [ ] Pending sections should be updated as soon as real artifacts exist
- [ ] One canonical demo path should be obvious
- [ ] HF Space / demo surface should be linked clearly if live
- [ ] Screenshots or committed lightweight result artifacts should be added if they improve first impression

Success bar:
- [ ] The repo reads as "proven and intentional," not "almost there"

## 7. Final Review Pass

- [ ] Re-read `README.md`
- [ ] Re-read `SUBMISSION_BRIEF.md`
- [ ] Re-read `HACKATHON.md`
- [ ] Re-read generated scorecard + summary bundle
- [ ] Re-check that the README claims match the final outputs
- [ ] Re-check that demo commands still work exactly as written

## Done Means

You are ready when:

- [ ] the contract is consistent
- [ ] evaluation is config-safe
- [ ] one clean evidence bundle exists
- [ ] one deterministic demo is locked
- [ ] the README and pitch both point to the same proof
