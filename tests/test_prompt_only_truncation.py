"""Regression test for H34: prompt-only truncation must not silently produce
zero-gradient rows.

When a prompt tokenizes to >= L_full tokens (the full prompt+completion
sequence length after truncation), the completion mask is empty and the row
contributes zero gradient.  The trainer must fail-closed via RuntimeError
rather than silently accepting such rows.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reuse the fake-torch machinery from test_real_grpo when real torch is absent
# ---------------------------------------------------------------------------

try:
    import torch as _real_torch_check  # noqa: F401

    _USE_FAKE = False
except ImportError:
    _USE_FAKE = True


def _build_fake_torch() -> types.ModuleType:
    """Minimal fake torch (subset of test_real_grpo's version)."""
    import numpy as np_  # type: ignore

    ft = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, data: Any, dtype: Any = None, device: Any = None):
            if isinstance(data, _Tensor):
                self._data = data._data.copy()
            elif isinstance(data, (list, tuple)):
                self._data = np_.array(data, dtype=np_.float64)
            else:
                self._data = np_.array(data, dtype=np_.float64)
            self.shape = self._data.shape
            self.device = device or "cpu"

        def item(self):
            return float(self._data)

        def detach(self):
            return _Tensor(self._data.copy())

        def clone(self):
            return _Tensor(self._data.copy())

        def numel(self):
            return self._data.size

        def to(self, *a, **kw):
            return self

        def tolist(self):
            d = self._data
            if d.ndim == 0:
                return float(d)
            return d.tolist()

        def sum(self, dim=None, **kw):
            if dim is not None:
                return _Tensor(np_.sum(self._data, axis=dim))
            return _Tensor(np_.sum(self._data))

        def __getitem__(self, idx):
            return _Tensor(self._data[idx])

        def __setitem__(self, idx, val):
            if isinstance(val, _Tensor):
                val = val._data
            if isinstance(idx, _Tensor):
                idx = idx._data.astype(bool)
            self._data[idx] = val

        def __eq__(self, other):  # type: ignore[override]
            if isinstance(other, _Tensor):
                return _Tensor((self._data == other._data).astype(np_.float64))
            return _Tensor((self._data == other).astype(np_.float64))

        def __ne__(self, other):  # type: ignore[override]
            if isinstance(other, _Tensor):
                return _Tensor((self._data != other._data).astype(np_.float64))
            return _Tensor((self._data != other).astype(np_.float64))

        def __add__(self, other):
            if isinstance(other, _Tensor):
                return _Tensor(self._data + other._data)
            return _Tensor(self._data + other)

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            if isinstance(other, _Tensor):
                return _Tensor(self._data - other._data)
            return _Tensor(self._data - other)

        def __mul__(self, other):
            if isinstance(other, _Tensor):
                return _Tensor(self._data * other._data)
            return _Tensor(self._data * other)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __neg__(self):
            return _Tensor(-self._data)

        def __float__(self):
            return float(self._data)

    ft.Tensor = _Tensor

    # nn stubs
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = type("Module", (), {
        "__init__": lambda self: None,
        "parameters": lambda self: iter([]),
    })
    ft.nn = nn_mod

    # optim stubs
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.AdamW = lambda *a, **kw: None
    ft.optim = optim_mod

    ft.cuda = types.SimpleNamespace(is_available=lambda: False)
    ft.isfinite = lambda t: _Tensor(np_.isfinite(t._data).astype(np_.float64))

    return ft


if _USE_FAKE:
    _fake_torch = _build_fake_torch()
    _torch_modules = {
        "torch": _fake_torch,
        "torch.nn": _fake_torch.nn,
        "torch.nn.functional": types.ModuleType("torch.nn.functional"),
        "torch.optim": _fake_torch.optim,
    }


def _inject():
    saved = {}
    for name, mod in _torch_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return saved


def _restore(saved):
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


# ===========================================================================
# Tests
# ===========================================================================


class TestPromptOnlyTruncation:
    """H34 regression: prompt-only rows must trigger RuntimeError."""

    @pytest.fixture(autouse=True)
    def _setup_torch(self):
        if _USE_FAKE:
            self._saved = _inject()
        yield
        if _USE_FAKE and hasattr(self, "_saved"):
            _restore(self._saved)

    def test_prompt_only_row_raises_runtime_error(self):
        """When prompt length >= full sequence length, _tokenize_batch raises.

        We mock the tokenizer so that:
          - The prompt tokenizes to attention_mask = [1, 1, 1]  (3 tokens)
          - The full prompt+completion also tokenizes to length 3
            (simulating truncation that drops all completion tokens).

        This means plen=3 >= L_full=3, which must raise RuntimeError.
        """
        from training.train import MultiAgentGRPOTrainer

        torch = sys.modules["torch"]

        # -- mock model with a parameter on cpu --
        param_mock = MagicMock()
        param_mock.device = "cpu"
        model = MagicMock()
        model.parameters.side_effect = lambda *a, **kw: iter([param_mock])

        # -- mock tokenizer --
        tokenizer = MagicMock()
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 8  # small to trigger truncation
        tokenizer.pad_token = "[PAD]"

        # apply_chat_template: just return the prompt as a string
        tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda prompt, tokenize=False, add_generation_prompt=True: str(prompt)
        )

        # When tokenizing the full text, return 3 tokens (all ones for mask)
        # When tokenizing the prompt-only text, also return 3 tokens (all ones)
        # This simulates the case where the completion was entirely truncated.
        full_ids = torch.Tensor(np.array([[1, 2, 3]], dtype=np.int64))
        full_mask = torch.Tensor(np.array([[1, 1, 1]], dtype=np.float64))

        prompt_ids = torch.Tensor(np.array([[4, 5, 6]], dtype=np.int64))
        prompt_mask = torch.Tensor(np.array([[1, 1, 1]], dtype=np.float64))

        def _tokenize_fn(texts, **kwargs):
            return {
                "input_ids": full_ids.clone(),
                "attention_mask": full_mask.clone(),
            }

        tokenizer.side_effect = _tokenize_fn

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )

        prompts = [[{"role": "user", "content": "hello"}]]
        completions = ["world"]

        with pytest.raises(RuntimeError, match=r"Prompt-only truncation"):
            trainer._tokenize_batch(prompts, completions)

    def test_healthy_rows_do_not_raise(self):
        """When prompt < full length, _tokenize_batch succeeds normally."""
        from training.train import MultiAgentGRPOTrainer

        torch = sys.modules["torch"]

        param_mock = MagicMock()
        param_mock.device = "cpu"
        model = MagicMock()
        model.parameters.side_effect = lambda *a, **kw: iter([param_mock])

        tokenizer = MagicMock()
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = 32
        tokenizer.pad_token = "[PAD]"

        tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda prompt, tokenize=False, add_generation_prompt=True: str(prompt)
        )

        # Full text: 8 tokens, prompt: 3 tokens => plen=3 < L_full=8, healthy
        full_ids = torch.Tensor(np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64))
        full_mask = torch.Tensor(np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64))

        prompt_ids = torch.Tensor(np.array([[4, 5, 6, 0, 0, 0, 0, 0]], dtype=np.int64))
        prompt_mask = torch.Tensor(np.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float64))

        call_count = [0]

        def _tokenize_fn(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # full_texts call
                return {
                    "input_ids": full_ids.clone(),
                    "attention_mask": full_mask.clone(),
                }
            else:
                # rendered_prompts call
                return {
                    "input_ids": prompt_ids.clone(),
                    "attention_mask": prompt_mask.clone(),
                }

        tokenizer.side_effect = _tokenize_fn

        trainer = MultiAgentGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=1e-4,
            kl_coef=0.04,
            clip_range=0.2,
            num_train_epochs_per_step=1,
        )

        prompts = [[{"role": "user", "content": "hello"}]]
        completions = ["world response here"]

        # Should NOT raise
        encoded_full, shifted_labels = trainer._tokenize_batch(prompts, completions)
        assert encoded_full is not None
        assert shifted_labels is not None
