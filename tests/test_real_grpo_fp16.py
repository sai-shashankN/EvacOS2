from __future__ import annotations

import importlib.util
from contextlib import contextmanager

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="real torch required",
)


def _build_trainer():
    import torch

    from training.train import MultiAgentGRPOTrainer

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16))

        @contextmanager
        def disable_adapter(self):
            yield self

    class TinyTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 1
        padding_side = "right"

    return MultiAgentGRPOTrainer(
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        learning_rate=1e-4,
        kl_coef=0.04,
        clip_range=0.2,
        num_train_epochs_per_step=1,
    )


def _grouped_inputs():
    return {
        "prompts": [[{"role": "user", "content": "prompt-0"}, {"role": "user", "content": "prompt-1"}]],
        "completions": [["completion-0", "completion-1"]],
        "completion_token_ids": [[[7, 8], [9, 10]]],
        "raw_rewards": [[1.0, -1.0]],
        "normalized_rewards": [[1.0, -1.0]],
        "samples": [[]],
    }


def test_non_finite_kl_loss_raises_runtime_error():
    import torch

    trainer = _build_trainer()
    encoded_full = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
    }
    shifted_labels = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)

    old_lp = torch.zeros((2, 2), dtype=torch.float16)
    ref_lp = torch.full((2, 2), 50.0, dtype=torch.float16)
    new_lp = torch.zeros((2, 2), dtype=torch.float16, requires_grad=True)

    trainer._tokenize_batch = lambda *args, **kwargs: (encoded_full, shifted_labels)
    call_count = {"value": 0}

    def fake_logprobs(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return old_lp
        if call_count["value"] == 2:
            return ref_lp
        return new_lp

    trainer._masked_token_logprobs = fake_logprobs

    with pytest.raises(RuntimeError, match="Non-finite kl_loss"):
        trainer.step(_grouped_inputs())


def test_small_fp16_deltas_complete_without_raising():
    import torch

    trainer = _build_trainer()
    encoded_full = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
    }
    shifted_labels = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)

    old_lp = torch.zeros((2, 2), dtype=torch.float16)
    ref_lp = torch.full((2, 2), 0.5, dtype=torch.float16)
    new_lp = torch.zeros((2, 2), dtype=torch.float16, requires_grad=True)

    trainer._tokenize_batch = lambda *args, **kwargs: (encoded_full, shifted_labels)
    call_count = {"value": 0}

    def fake_logprobs(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return old_lp
        if call_count["value"] == 2:
            return ref_lp
        return new_lp

    trainer._masked_token_logprobs = fake_logprobs

    metrics = trainer.step(_grouped_inputs())

    assert "loss" in metrics
    assert call_count["value"] >= 3
