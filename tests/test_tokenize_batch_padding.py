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


def _expected_mask_for_row(batch_width: int, prompt_len: int, full_len: int) -> list[float]:
    expected = [0.0] * (batch_width - 1)
    for idx in range(max(prompt_len - 1, 0), max(full_len - 1, 0)):
        expected[idx] = 1.0
    return expected


def test_tokenize_batch_masks_only_completion_tokens_with_left_padding_tokenizer():
    tokenizer = _load_cached_chat_tokenizer()
    tokenizer.padding_side = "left"
    trainer = _build_trainer(tokenizer)

    prompts = [
        [{"role": "user", "content": "Short prompt."}],
        [{"role": "user", "content": "A much longer prompt with extra words for left padding coverage."}],
    ]
    completions = [
        " Short completion.",
        " Longer completion with more tokens.",
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
    prompt_lengths = [
        int(tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)["attention_mask"].sum().item())
        for text in rendered_prompts
    ]
    full_lengths = [
        int(tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)["attention_mask"].sum().item())
        for text in full_texts
    ]

    batch_width = int(encoded_full["input_ids"].shape[1])
    for row_idx, (prompt_len, full_len) in enumerate(zip(prompt_lengths, full_lengths, strict=False)):
        expected = _expected_mask_for_row(batch_width, prompt_len, full_len)
        observed = completion_mask[row_idx].tolist()
        assert observed == expected
        assert sum(observed) == full_len - prompt_len

    assert tokenizer.padding_side == "left"
