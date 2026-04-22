from __future__ import annotations

from types import SimpleNamespace

from training.train import _group_for_grpo


def _sample(
    *,
    group_id: str,
    prompt_text: str,
    completion_text: str,
    raw_reward: float,
    normalized_reward: float,
):
    return SimpleNamespace(
        group_id=group_id,
        prompt=[{"role": "user", "content": prompt_text}],
        completion_text=completion_text,
        completion_token_ids=None,
        raw_reward=raw_reward,
        normalized_reward=normalized_reward,
    )


def test_group_for_grpo_uses_deterministic_key_order_without_exposing_group_names():
    grouped = _group_for_grpo(
        [
            SimpleNamespace(
                samples=[
                    _sample(
                        group_id="ep_demo_r_2_floor",
                        prompt_text="prompt-2a",
                        completion_text="completion-2a",
                        raw_reward=2.0,
                        normalized_reward=0.2,
                    ),
                    _sample(
                        group_id="ep_demo_r_10_floor",
                        prompt_text="prompt-10a",
                        completion_text="completion-10a",
                        raw_reward=10.0,
                        normalized_reward=1.0,
                    ),
                ]
            ),
            SimpleNamespace(
                samples=[
                    _sample(
                        group_id="ep_demo_r_2_floor",
                        prompt_text="prompt-2b",
                        completion_text="completion-2b",
                        raw_reward=3.0,
                        normalized_reward=0.3,
                    ),
                    _sample(
                        group_id="rollout_orchestrator",
                        prompt_text="prompt-orch",
                        completion_text="completion-orch",
                        raw_reward=1.0,
                        normalized_reward=0.1,
                    ),
                ]
            ),
        ]
    )

    # Lexicographic ordering is intentional: "r_10" sorts before "r_2".
    assert grouped["completions"] == [
        ["completion-10a"],
        ["completion-2a", "completion-2b"],
        ["completion-orch"],
    ]
    assert grouped["raw_rewards"] == [[10.0], [2.0, 3.0], [1.0]]
    assert "group_ids" not in grouped
