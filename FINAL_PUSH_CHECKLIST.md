# Final Push Checklist

This is the active checklist for taking EvacOS2 from "strong repo" to "strongest possible submission."

## Priority Order

- [ ] Run a real split-role training checkpoint on remote GPU hardware
- [ ] Generate trained submission artifacts immediately after the run
- [ ] Tighten demo / deployment clarity
- [ ] Do one final judge-eye consistency pass across the repo

## 1. Split-Role Training Run

Target:
- [ ] Start with split-role training directly
- [ ] Orchestrator model: `Qwen/Qwen2.5-7B-Instruct`
- [ ] Floor-agent model: `Qwen/Qwen2.5-3B-Instruct`
- [ ] Primary target runtime: Vast.ai
- [ ] First budget target: about `2` hours for the initial run

Preflight:
- [ ] Vast.ai account created
- [ ] Suitable Linux CUDA machine selected
- [ ] Repo available on the remote box
- [ ] Python environment created
- [ ] Follow `REMOTE_GPU_SETUP.md` instead of ad-hoc package install order
- [ ] CUDA-visible PyTorch / training stack verified
- [ ] Hugging Face login/token available if model pulls require it
- [ ] Output/checkpoint directory chosen on persistent storage

Training config:
- [ ] Confirm `training/config.yaml` or a dedicated remote config points to split-role bases
- [ ] Prefer Linux/CUDA path with `backend: "unsloth"`
- [ ] Enable `rollout.use_vllm: true` if supported on the chosen machine
- [ ] Confirm checkpoint cadence is frequent enough that a short run still leaves usable artifacts
- [ ] Confirm outputs write to a persistent path, not disposable container scratch space

During run:
- [ ] Capture the exact command used
- [ ] Capture start time, stop time, and actual wall-clock runtime
- [ ] Capture any OOM / throughput / dependency issues
- [ ] Preserve latest checkpoint and any adapter directories
- [ ] Preserve training logs/metrics

Success bar for this step:
- [ ] At least one real split-role checkpoint saved successfully
- [ ] Enough evidence collected to run baseline-vs-trained comparison

## 2. Trained Submission Artifacts

Immediately after the first successful training run:
- [ ] Build the trained demo bundle
- [ ] Generate a real `submission_scorecard.md`
- [ ] Generate a real `submission_scorecard.json`
- [ ] Generate `demo_bundle_summary.md`
- [ ] Generate `baseline_vs_trained.csv`
- [ ] Check whether the trained result actually improves the headline metrics

Key outputs:
- [ ] `outputs/demo_bundle/submission_scorecard.md`
- [ ] `outputs/demo_bundle/submission_scorecard.json`
- [ ] `outputs/demo_bundle/demo_bundle_summary.md`
- [ ] `outputs/demo_bundle/baseline_vs_trained.csv`

## 3. Demo / Deployment Tightening

Do this after trained artifacts exist:
- [ ] Make the quickest judge path unmistakable
- [ ] Tighten hosted-vs-local demo language so there is one canonical story
- [ ] Add or refine a "demo in 60 seconds" section if still needed
- [ ] Ensure README, HACKATHON, and SUBMISSION_BRIEF point to the same final demo flow

## 4. Final Judge-Eye Pass

Do this last:
- [ ] Re-read `README.md`
- [ ] Re-read `HACKATHON.md`
- [ ] Re-read `SUBMISSION_BRIEF.md`
- [ ] Re-read the generated trained scorecard and bundle summary
- [ ] Remove wording drift, contradictions, or stale claims
- [ ] Make sure the repo tells one clean story from top to bottom

## Vast.ai Migration Notes

When the Vast.ai account is ready, the next practical work is:
- [ ] choose the machine spec
- [ ] clone or sync the repo onto the box
- [ ] install dependencies
- [ ] set secrets/tokens
- [ ] point training outputs at persistent storage
- [ ] run the split-role training command
- [ ] verify checkpoints exist before ending the session

## Done Means

We can call this truly submission-done when:
- [ ] a real split-role checkpoint exists
- [ ] trained scorecard artifacts exist
- [ ] the demo/deployment path is tight
- [ ] the final repo-wide judge pass is complete
