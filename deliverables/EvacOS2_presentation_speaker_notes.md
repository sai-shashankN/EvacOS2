## Title

Open by saying this is not a chatbot demo. It is an environment where agents have to act, and those actions are checked by a simulator.

## The gap

Frame the problem as evaluation, not just evacuation. We need environments that can catch bad actions, not just bad wording.

## What EvacOS2 is

Keep this slide concrete. It is a simulator, an OpenEnv API, a training stack, and a public evidence trail.

## Architecture

This is the simple story: do not make every local move wait for a large model. Use the large model where global context matters.

## Evidence stack

This slide helps avoid overclaiming. We have a clean 3B held-out result and a 7B training-integrity result, not final 7B convergence.

## Held-out result

Say this is the main quantitative claim. Same model family, same held-out seeds, same evaluator, LoRA versus no LoRA.

## Family breakdown

Use this slide to explain why flood matters. It starts much lower and gets a large improvement, while gas and fire show strong reliability gains.

## Route discipline

Explain that this is the slide that makes the reliability claim concrete. Route targets are a simple, strict contract.

## Training trail

This answers the question: did we actually train? The answer is yes, with checkpoints, metrics, plots, and public artifacts.

## 7B

Be precise. The 7B evidence is real, but it is not the same type of claim as the 3B held-out result.

## Artifacts

This slide is for credibility. The judges can inspect the live Space, code, notebook, paper, plots, and adapter repo.

## Close

End with the evaluator story. The system makes hidden coordination failures visible, which is the point of the benchmark.

## Tech stack

Keep this practical. OpenEnv gives the public environment contract, Qwen2.5 gives the base instruction models, Unsloth and LoRA make training affordable, and Hugging Face makes the runtime and artifacts inspectable.

## Why Unsloth?

Say this was about iteration speed. We were running canaries, parser repairs, resume tests, specialist runs, and orchestrator checks. The lower-memory path made that realistic on rented GPUs.

## LoRA vs QLoRA

Use precise wording: we trained LoRA adapters while loading the base models in 4-bit mode. So it is a QLoRA-style setup, but the public artifacts are LoRA adapter checkpoints, not full model copies.

## Training loop

Each step goes through the simulator, not just a text dataset. The model sees structured state, emits schema-bound actions, gets checked by the environment, receives reward, and then the adapter is updated. Held-out evaluation stays separate so the headline numbers remain clean.
