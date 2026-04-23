# A100 7B/3B Split-Role Training Run

This is a lightweight, Git-tracked summary of the April 23-24 A100 run. The full LoRA adapters are intentionally not committed because they are large binary artifacts.

## Run Metadata

- steps completed: `100`
- final checkpoint step: `ckpt_99`
- wall clock: `44.8 minutes`
- model stack: `orchestrator=Qwen/Qwen2.5-7B-Instruct;floor_agent=Qwen/Qwen2.5-3B-Instruct`
- config hash: `sha256:5cebf445544f`
- W&B run id: `e7ljdmh1`

## Reward Signal

| disaster | tier | samples | first mean | last mean | first reward | last reward | best reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| fire | easy | 50 | 0.7123 | 0.7115 | 0.5453 | 2.1024 | 2.1584 |
| fire | medium | 36 | -1.5750 | 0.8348 | -2.3947 | 1.5382 | 3.1896 |

## Interpretation

The run demonstrates that the end-to-end split-role training pipeline reaches checkpoints and produces non-zero curriculum reward signal. The strongest visible improvement is in the fire:medium lane, whose early rewards were mostly negative and whose later window contains repeated positive outcomes. This is training evidence, not a final held-out trained-vs-baseline claim.

For judge-facing proof, pair this with the fixed-suite baseline bundle in this directory and run a fresh `evaluation.demo_bundle` comparison against the selected LoRA checkpoint artifact.
