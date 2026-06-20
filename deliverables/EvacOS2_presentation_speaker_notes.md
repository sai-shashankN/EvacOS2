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
