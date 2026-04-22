from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None,
    reason="transformers + torch required",
)


def _load_cached_chat_tokenizer():
    from transformers import AutoTokenizer

    candidates = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
    ]
    for name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "ping"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer
        except Exception:
            continue
    pytest.skip("No cached chat-template tokenizer available locally")


def _build_trainer(tokenizer):
    import torch

    from training.train import MultiAgentGRPOTrainer

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("forward should not run in _tokenize_batch test")

    return MultiAgentGRPOTrainer(
        model=TinyModel(),
        tokenizer=tokenizer,
        learning_rate=1e-4,
        kl_coef=0.04,
        clip_range=0.2,
        num_train_epochs_per_step=1,
    )


def test_real_tokenizer_path_restores_left_padding_and_marks_exact_completion_span():
    tokenizer = _load_cached_chat_tokenizer()
    tokenizer.padding_side = "left"
    trainer = _build_trainer(tokenizer)

    prompts = [
        [{"role": "user", "content": "Floor 1 status?"}],
        [{"role": "user", "content": "Summarize floor 3 observations with extra operational context."}],
        [{"role": "user", "content": "Need a concise evacuation recommendation for the east stairwell."}],
    ]
    completions = [
        " Route civilians to exit A.",
        " Lock room 3B and redirect traffic to the north exit immediately.",
        " Wait.",
    ]

    encoded_full, shifted_labels = trainer._tokenize_batch(prompts, completions)
    completion_mask = (shifted_labels != -100).float()

    rendered_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    full_texts = [
        rendered_prompt + completion
        for rendered_prompt, completion in zip(rendered_prompts, completions, strict=False)
    ]
    max_len = getattr(tokenizer, "model_max_length", 4096)

    batch_width = int(encoded_full["input_ids"].shape[1])
    for row_idx, (rendered_prompt, full_text) in enumerate(zip(rendered_prompts, full_texts, strict=False)):
        prompt_len = int(
            tokenizer(rendered_prompt, return_tensors="pt", truncation=True, max_length=max_len)[
                "attention_mask"
            ].sum().item()
        )
        full_len = int(
            tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)[
                "attention_mask"
            ].sum().item()
        )
        observed = completion_mask[row_idx].tolist()
        expected = [0.0] * (batch_width - 1)
        for idx in range(max(prompt_len - 1, 0), max(full_len - 1, 0)):
            expected[idx] = 1.0
        assert observed == expected
        assert completion_mask[row_idx].sum().item() == full_len - prompt_len

    assert tokenizer.padding_side == "left"
