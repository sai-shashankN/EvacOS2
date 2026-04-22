from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


try:
    import torch as _real_torch_check  # noqa: F401

    _USE_FAKE = False
except ImportError:
    _USE_FAKE = True


def _build_fake_torch() -> types.ModuleType:
    ft = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data: Any, dtype: Any = None, device: Any = None):
            del dtype
            if isinstance(data, _Tensor):
                self._data = data._data.copy()
            else:
                self._data = np.array(data)
            self.shape = self._data.shape
            self.device = device or "cpu"

        def clone(self):
            return _Tensor(self._data.copy(), device=self.device)

        def to(self, device):
            self.device = device
            return self

        def tolist(self):
            return self._data.tolist()

        def __getitem__(self, idx):
            return _Tensor(self._data[idx], device=self.device)

        def __setitem__(self, idx, value):
            if isinstance(idx, _Tensor):
                idx = idx._data.astype(bool)
            if isinstance(value, _Tensor):
                value = value._data
            self._data[idx] = value

        def __eq__(self, other):  # type: ignore[override]
            if isinstance(other, _Tensor):
                other = other._data
            return _Tensor(self._data == other, device=self.device)

    ft.Tensor = _Tensor  # type: ignore[attr-defined]
    ft.tensor = lambda data: _Tensor(data)  # type: ignore[attr-defined]
    ft.optim = types.SimpleNamespace(AdamW=lambda *args, **kwargs: MagicMock())
    ft.cuda = types.SimpleNamespace(is_available=lambda: False)
    return ft


if _USE_FAKE:
    _fake_torch = _build_fake_torch()
    _torch_modules = {
        "torch": _fake_torch,
        "torch.optim": _fake_torch.optim,
    }


def _inject_fake_torch():
    saved = {}
    for name, module in _torch_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module
    return saved


def _restore_fake_torch(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestTokenizeBatchCompletionIds:
    @pytest.fixture(autouse=True)
    def _setup_torch(self):
        if _USE_FAKE:
            self._saved = _inject_fake_torch()
        yield
        if _USE_FAKE:
            _restore_fake_torch(self._saved)

    def test_tokenize_batch_uses_completion_token_ids_and_reencode_fallback(self):
        from training.train import MultiAgentGRPOTrainer

        param_mock = MagicMock()
        param_mock.device = "cpu"
        param_mock.requires_grad = True

        model = MagicMock()
        model.parameters.side_effect = lambda *a, **kw: iter([param_mock])
        model.named_parameters.side_effect = lambda *a, **kw: iter([("lora_a.weight", param_mock)])

        class FakeTokenizer:
            padding_side = "left"
            model_max_length = 16
            pad_token = "<pad>"
            pad_token_id = 0
            eos_token = "<eos>"
            eos_token_id = 99

            def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
                del add_generation_prompt
                text = prompt[0]["content"]
                if tokenize:
                    if text == "prompt-1":
                        return [11, 12]
                    if text == "prompt-2":
                        return [21, 22]
                return text

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                if text == "fallback":
                    return {"input_ids": [41, 42]}
                raise AssertionError(f"Unexpected tokenizer fallback text: {text!r}")

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=FakeTokenizer(),
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )

        prompts = [
            [{"role": "user", "content": "prompt-1"}],
            [{"role": "user", "content": "prompt-2"}],
        ]
        completions = ["unused", "fallback"]
        encoded_full, shifted_labels = trainer._tokenize_batch(
            prompts,
            completions,
            completion_token_ids=[[7, 8, 9], None],
        )

        assert encoded_full["input_ids"][0].tolist()[:5] == [11, 12, 7, 8, 9]
        assert encoded_full["input_ids"][1].tolist()[:4] == [21, 22, 41, 42]
        assert shifted_labels[0].tolist() == [-100, 7, 8, 9]
        assert shifted_labels[1].tolist()[:4] == [-100, 41, 42, -100]
